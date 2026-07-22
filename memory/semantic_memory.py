"""
Semantic memory: extracted facts about the user that should persist and
be updated over time, as opposed to episodic memory's per-session
summaries. "User lives in Tehran" is a fact worth carrying forward
indefinitely; "user asked about Python decorators in session 3" is not --
that's episodic memory's job.

Facts are extracted from conversation text by an LLM call, then merged
into the existing fact store with explicit conflict handling: a new fact
that contradicts an existing one (e.g. "lives in Tehran" vs. a new "lives
in Berlin") doesn't just get appended alongside the old one -- it
replaces it, with the old value kept in a history list rather than
silently discarded. Silently accumulating contradictory facts would make
the store internally inconsistent and useless for retrieval.

Usage as a library:
    from semantic_memory import SemanticMemoryStore
    store = SemanticMemoryStore("facts.json")
    store.upsert_fact("location", "Tehran", source_session="s1")
    store.upsert_fact("location", "Berlin", source_session="s5")  # conflict -> replaces, keeps history
    print(store.get_fact("location"))  # "Berlin"
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class SemanticMemoryStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({})

    def _load(self) -> dict:
        with open(self.path, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else {}

    def _save(self, data: dict):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def upsert_fact(self, key: str, value: str, source_session: str = None) -> dict:
        """
        Inserts a new fact or updates an existing one. If the key already
        exists with a different value, the old value is moved into a
        'history' list on the fact entry rather than being lost -- this
        keeps a record of what changed and when, useful for debugging
        why the agent's understanding of the user shifted.
        """
        data = self._load()
        now = datetime.now(timezone.utc).isoformat()

        existing = data.get(key)
        if existing is None:
            data[key] = {
                "value": value,
                "updated_at": now,
                "source_session": source_session,
                "history": [],
            }
        elif existing["value"] != value:
            data[key] = {
                "value": value,
                "updated_at": now,
                "source_session": source_session,
                "history": existing["history"] + [{
                    "value": existing["value"],
                    "replaced_at": now,
                }],
            }
        # If the new value matches the existing one exactly, nothing
        # changes -- re-confirming a known fact isn't a conflict.

        self._save(data)
        return data[key]

    def get_fact(self, key: str) -> str | None:
        data = self._load()
        entry = data.get(key)
        return entry["value"] if entry else None

    def get_all_facts(self) -> dict:
        return self._load()

    def get_fact_history(self, key: str) -> list:
        data = self._load()
        entry = data.get(key)
        return entry["history"] if entry else []

    def delete_fact(self, key: str):
        data = self._load()
        data.pop(key, None)
        self._save(data)
