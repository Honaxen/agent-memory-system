"""
A multi-session recall test: runs a scripted "session 1" that states
several facts, lets memory extraction run on it, then runs a fresh
"session 2" (a new process-level state, simulating a genuinely separate
conversation) that asks questions only answerable using memory from
session 1 -- and checks whether the answer is actually correct.

Critically, this also runs the SAME session-2 questions against a
stateless baseline (no memory context at all) to make the comparison
concrete: memory isn't given credit just because the agent said
something plausible-sounding, it's credited only when it produced a
correct answer that the stateless baseline could not have produced.

Usage:
    python multi_session_recall.py \
        --model gemma3:12b \
        --facts_path ../memory/eval_facts.json \
        --episodes_path ../memory/eval_episodes.jsonl \
        --output ../evaluation/results/recall_test.json
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "memory"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
from semantic_memory import SemanticMemoryStore  # noqa: E402
from episodic_memory import EpisodicMemoryStore  # noqa: E402
from extract_facts import extract_and_store  # noqa: E402
from memory_agent import build_memory_context, call_ollama  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"

# Session 1: a scripted transcript stating facts that session 2's
# questions will probe for. Written as a single user turn (a monologue)
# to keep this test simple and reproducible -- no back-and-forth needed
# to state these facts.
SESSION_1_TRANSCRIPT = (
    "User: Hi! Quick intro -- I'm a backend engineer, I mainly work with Go. "
    "I recently adopted a dog named Biscuit. My biggest current goal is "
    "preparing for a half marathon in October."
)

# Session 2: questions that can ONLY be answered correctly using memory
# from session 1 -- a stateless agent has no way to know these answers.
RECALL_QUESTIONS = [
    {"question": "What programming language do I mainly work with?", "expected_answer": "Go"},
    {"question": "What's my dog's name?", "expected_answer": "Biscuit"},
    {"question": "What event am I training for?", "expected_answer": "half marathon"},
]


def judge_answer(model: str, question: str, expected: str, actual: str) -> bool:
    judge_prompt = f"""Does this answer correctly convey the expected information?

Question: {question}
Expected: {expected}
Answer: {actual}

Reply with ONLY "yes" or "no"."""
    result = call_ollama(model, judge_prompt).strip().lower()
    return "yes" in result


def run_session_1(model: str, facts_path: str, episodes_path: str):
    facts_store = SemanticMemoryStore(facts_path)
    episodes_store = EpisodicMemoryStore(episodes_path)

    extract_and_store(SESSION_1_TRANSCRIPT, facts_path, "session_1", model)
    episodes_store.add_episode(
        session_id="session_1",
        summary="User introduced themselves as a backend engineer working mainly in Go, "
                "mentioned adopting a dog named Biscuit, and shared a goal of running a half marathon in October.",
    )


def run_session_2_with_memory(model: str, facts_path: str, episodes_path: str) -> list:
    facts_store = SemanticMemoryStore(facts_path)
    episodes_store = EpisodicMemoryStore(episodes_path)
    system_prompt = build_memory_context(facts_store, episodes_store)

    results = []
    for q in RECALL_QUESTIONS:
        answer = call_ollama(model, q["question"], system=system_prompt)
        results.append({"question": q["question"], "expected_answer": q["expected_answer"], "actual_answer": answer})
    return results


def run_session_2_stateless(model: str) -> list:
    """No memory context at all -- the control condition. If this
    somehow answers correctly too, the memory system isn't actually
    adding anything (the model would just be guessing right, or the
    question wasn't a fair test of memory)."""
    results = []
    for q in RECALL_QUESTIONS:
        answer = call_ollama(model, q["question"])  # no system prompt
        results.append({"question": q["question"], "expected_answer": q["expected_answer"], "actual_answer": answer})
    return results


def main(args):
    # Start from clean memory files for a reproducible test.
    Path(args.facts_path).unlink(missing_ok=True)
    Path(args.episodes_path).unlink(missing_ok=True)

    print("Running session 1 (stating facts)...")
    run_session_1(args.model, args.facts_path, args.episodes_path)

    print("Running session 2 WITH memory...")
    with_memory_results = run_session_2_with_memory(args.model, args.facts_path, args.episodes_path)

    print("Running session 2 WITHOUT memory (stateless baseline)...")
    stateless_results = run_session_2_stateless(args.model)

    print("\nJudging answers...")
    for r in with_memory_results:
        r["correct"] = judge_answer(args.model, r["question"], r["expected_answer"], r["actual_answer"])
    for r in stateless_results:
        r["correct"] = judge_answer(args.model, r["question"], r["expected_answer"], r["actual_answer"])

    with_memory_score = sum(r["correct"] for r in with_memory_results)
    stateless_score = sum(r["correct"] for r in stateless_results)
    total = len(RECALL_QUESTIONS)

    report = {
        "with_memory_results": with_memory_results,
        "stateless_results": stateless_results,
        "with_memory_score": f"{with_memory_score}/{total}",
        "stateless_score": f"{stateless_score}/{total}",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Multi-Session Recall Test ===")
    print(f"With memory:    {with_memory_score}/{total} correct")
    print(f"Without memory: {stateless_score}/{total} correct")
    for r in with_memory_results:
        status = "correct" if r["correct"] else "WRONG"
        print(f"  [{status}] Q: {r['question']}")
        print(f"           A: {r['actual_answer'][:100]}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-session memory recall test, with a stateless control")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--facts_path", default="../memory/eval_facts.json")
    parser.add_argument("--episodes_path", default="../memory/eval_episodes.jsonl")
    parser.add_argument("--output", default="../evaluation/results/recall_test.json")
    args = parser.parse_args()

    main(args)
