"""
Keeps episodic memory from growing unbounded: once the number of stored
episodes exceeds a threshold, the oldest ones (beyond a "keep recent as-is"
window) are merged into a single consolidated summary via an LLM call,
replacing many old entries with one.

This mirrors how human memory works in a rough, practical sense -- recent
events stay detailed, older ones blur into a general summary rather than
being retained individually forever. The alternative (keeping every
session's summary forever) would make memory retrieval slower and, more
importantly, less useful: recalling "3 months ago, user asked about
decorators; 2 months ago, user asked about list comprehensions; ..." in
full is worse for most purposes than "user has been learning Python
fundamentals over the past few months."

Consolidation only touches episodes OLDER than the keep-recent window --
recent episodes are never touched, so nothing about last week's
conversation gets blurred by this process.

Usage:
    python consolidate_memory.py \
        --episodes_path ../memory/episodes.jsonl \
        --keep_recent 5 \
        --model gemma3:12b
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "memory"))
from episodic_memory import EpisodicMemoryStore  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"

CONSOLIDATION_PROMPT_TEMPLATE = """The following are summaries of several older conversation sessions with the same user, in chronological order. Merge them into a single consolidated summary (3-5 sentences) that preserves the useful, still-relevant information and drops anything that's now redundant or no longer meaningful.

Session summaries:
---
{summaries}
---

Reply with ONLY the consolidated summary, no preamble.
"""


def call_ollama(model: str, prompt: str, timeout: int = 60) -> str:
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return body.get("response", "").strip()


def consolidate(episodes_path: str, keep_recent: int, model: str, min_to_consolidate: int = 3) -> dict:
    store = EpisodicMemoryStore(episodes_path)
    all_episodes = store.get_all_episodes()

    if len(all_episodes) <= keep_recent:
        return {"consolidated": False, "reason": f"only {len(all_episodes)} episodes, at or below keep_recent={keep_recent}"}

    old_episodes = all_episodes[:-keep_recent]
    recent_episodes = all_episodes[-keep_recent:]

    if len(old_episodes) < min_to_consolidate:
        return {"consolidated": False,
                "reason": f"only {len(old_episodes)} old episode(s), below min_to_consolidate={min_to_consolidate}"}

    summaries_text = "\n".join(f"- [{ep['created_at'][:10]}] {ep['summary']}" for ep in old_episodes)
    consolidated_summary = call_ollama(model, CONSOLIDATION_PROMPT_TEMPLATE.format(summaries=summaries_text))

    consolidated_episode = {
        "session_id": f"consolidated_{old_episodes[0]['session_id']}_to_{old_episodes[-1]['session_id']}",
        "summary": consolidated_summary,
        "metadata": {
            "consolidated_from": [ep["session_id"] for ep in old_episodes],
            "original_count": len(old_episodes),
        },
        "created_at": old_episodes[-1]["created_at"],  # keep chronological position of the most recent merged episode
    }

    new_episode_list = [consolidated_episode] + recent_episodes
    store.replace_all(new_episode_list)

    return {
        "consolidated": True,
        "episodes_before": len(all_episodes),
        "episodes_after": len(new_episode_list),
        "merged_count": len(old_episodes),
        "consolidated_summary": consolidated_summary,
    }


def main(args):
    result = consolidate(args.episodes_path, args.keep_recent, args.model, args.min_to_consolidate)

    if not result["consolidated"]:
        print(f"No consolidation performed: {result['reason']}")
        return

    print(f"Consolidated {result['merged_count']} old episode(s) into 1.")
    print(f"Episode count: {result['episodes_before']} -> {result['episodes_after']}")
    print(f"\nConsolidated summary:\n{result['consolidated_summary']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate old episodic memory into merged summaries")
    parser.add_argument("--episodes_path", default="../memory/episodes.jsonl")
    parser.add_argument("--keep_recent", type=int, default=5, help="Number of most recent episodes to leave untouched")
    parser.add_argument("--min_to_consolidate", type=int, default=3, help="Minimum old episodes needed to bother consolidating")
    parser.add_argument("--model", default="gemma3:12b")
    args = parser.parse_args()

    main(args)
