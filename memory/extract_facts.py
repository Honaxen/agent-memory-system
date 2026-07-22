"""
Bridges conversation text to semantic_memory.py's fact store: sends a
conversation transcript to an LLM, asks it to extract stable facts about
the user, and upserts each one into the SemanticMemoryStore.

Only extracts facts explicitly stated or clearly implied by the user --
not inferred assumptions. "I live in Tehran" -> location: Tehran is a
direct extraction. "I've been really busy at work" is NOT extracted as
a fact about occupation, job title, or workload, since that would be
guessing at specifics the user didn't actually provide. This restraint
matters because false facts, once stored, get retrieved and presented as
if the agent "remembers" something the user never said.

Usage:
    python extract_facts.py \
        --transcript "User: I just moved to Berlin last month. I work as a nurse." \
        --facts_path ../memory/facts.json \
        --session_id s1 \
        --model gemma3:12b
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from semantic_memory import SemanticMemoryStore  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"

EXTRACTION_PROMPT_TEMPLATE = """Extract stable facts about the user from this conversation transcript. Only extract facts the user explicitly stated or clearly and directly implied -- do not guess, infer personality traits, or fill in details the user didn't actually provide.

Transcript:
---
{transcript}
---

Return a JSON object where each key is a short fact category (e.g. "location", "occupation", "favorite_language") and each value is the fact itself as a short string. If no clear, stable facts are present, return an empty JSON object {{}}.

Reply with ONLY the JSON object, no other text.
"""


def call_ollama(model: str, prompt: str, timeout: int = 60) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return body.get("response", "").strip()


def parse_facts(raw_output: str) -> dict:
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_and_store(transcript: str, facts_path: str, session_id: str, model: str) -> dict:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(transcript=transcript)
    raw_output = call_ollama(model, prompt)
    extracted = parse_facts(raw_output)

    store = SemanticMemoryStore(facts_path)
    updated = {}
    for key, value in extracted.items():
        entry = store.upsert_fact(key, str(value), source_session=session_id)
        updated[key] = entry

    return {"extracted": extracted, "updated_entries": updated}


def main(args):
    result = extract_and_store(args.transcript, args.facts_path, args.session_id, args.model)

    print(f"\nExtracted {len(result['extracted'])} fact(s) from session {args.session_id}:")
    for key, value in result["extracted"].items():
        print(f"  {key}: {value}")

    if not result["extracted"]:
        print("  (none found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract stable facts from a conversation transcript")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--facts_path", default="facts.json")
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--model", default="gemma3:12b")
    args = parser.parse_args()

    main(args)
