"""
The agent loop that actually uses memory: before responding to a new
message, it retrieves relevant semantic facts and recent episodic
summaries and folds them into the prompt -- the same basic pattern as
RAG, except retrieving from the agent's own memory of the user instead
of from external documents.

At the end of a session, it summarizes the conversation and extracts any
new facts, writing both back to the memory stores -- closing the loop so
the *next* session (a fresh process, a fresh conversation) can retrieve
what happened here.

This is intentionally a simple loop, not a framework: one call to build
a memory-aware prompt, one call to run a turn, one call to end a session.
The goal is to make memory retrieval and memory writing both visible and
inspectable, not hidden behind agent-framework abstractions.

Usage:
    python memory_agent.py --session_id s2 --model gemma3:12b \
        --facts_path ../memory/facts.json \
        --episodes_path ../memory/episodes.jsonl
    (then type messages interactively; type 'exit' to end the session)
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "memory"))
from semantic_memory import SemanticMemoryStore  # noqa: E402
from episodic_memory import EpisodicMemoryStore  # noqa: E402
from extract_facts import extract_and_store  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory of past conversations with this user.

What you know about the user (facts extracted from past sessions):
{facts_block}

Summaries of recent past sessions:
{episodes_block}

Use this information naturally when relevant, without explicitly announcing that you're recalling stored memory (e.g. don't say "According to my records..."). If the user's current message contradicts something you know, trust the current message -- people's circumstances change.
"""

SUMMARIZE_PROMPT_TEMPLATE = """Summarize this conversation in 2-3 sentences, focused on what would be useful to remember for future conversations with this same user.

Conversation:
---
{transcript}
---

Reply with ONLY the summary, no preamble.
"""


def call_ollama(model: str, prompt: str, system: str = None, timeout: int = 60) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return body.get("response", "").strip()


def build_memory_context(facts_store: SemanticMemoryStore, episodes_store: EpisodicMemoryStore,
                          n_recent_episodes: int = 3) -> str:
    facts = facts_store.get_all_facts()
    if facts:
        facts_block = "\n".join(f"- {key}: {entry['value']}" for key, entry in facts.items())
    else:
        facts_block = "(no facts recorded yet)"

    recent = episodes_store.get_recent_episodes(n_recent_episodes)
    if recent:
        episodes_block = "\n".join(f"- {ep['summary']}" for ep in recent)
    else:
        episodes_block = "(no past sessions recorded yet)"

    return SYSTEM_PROMPT_TEMPLATE.format(facts_block=facts_block, episodes_block=episodes_block)


def run_turn(model: str, system_prompt: str, user_message: str) -> str:
    return call_ollama(model, user_message, system=system_prompt)


def end_session(model: str, transcript: list, session_id: str,
                 facts_store: SemanticMemoryStore, episodes_store: EpisodicMemoryStore):
    """Summarizes the session and extracts any new/updated facts, writing
    both back to the memory stores so the next session can retrieve them."""
    transcript_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in transcript)

    summary = call_ollama(model, SUMMARIZE_PROMPT_TEMPLATE.format(transcript=transcript_text))
    episodes_store.add_episode(session_id=session_id, summary=summary)

    extract_and_store(transcript_text, str(facts_store.path), session_id, model)

    return summary


def main(args):
    facts_store = SemanticMemoryStore(args.facts_path)
    episodes_store = EpisodicMemoryStore(args.episodes_path)

    system_prompt = build_memory_context(facts_store, episodes_store)
    print(f"\n=== Session {args.session_id} ===")
    print("(memory context loaded -- type 'exit' to end the session)\n")

    transcript = []
    while True:
        user_message = input("You: ")
        if user_message.strip().lower() == "exit":
            break

        transcript.append({"role": "user", "content": user_message})
        response = run_turn(args.model, system_prompt, user_message)
        transcript.append({"role": "assistant", "content": response})
        print(f"Agent: {response}\n")

    if transcript:
        print("\nEnding session -- summarizing and extracting facts...")
        summary = end_session(args.model, transcript, args.session_id, facts_store, episodes_store)
        print(f"Session summary saved: {summary}")
    else:
        print("No turns taken this session -- nothing to summarize.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-aware agent loop")
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--facts_path", default="../memory/facts.json")
    parser.add_argument("--episodes_path", default="../memory/episodes.jsonl")
    args = parser.parse_args()

    main(args)
