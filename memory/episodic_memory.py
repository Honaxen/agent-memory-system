"""
Episodic memory: stores a summary of each conversation session, not the
raw transcript. A session that ran for 40 turns shouldn't cost 40 turns
of context on every future retrieval -- the summary is what gets carried
forward, generated once at the end of the session.

Storage is a flat JSONL file (one JSON object per line) -- simple,
appendable, and diff-friendly, matching the file-based-over-database
pattern used elsewhere in this portfolio (llm-mlops-pipeline's registry,
for instance) for a portfolio-scale project.

Usage as a library:
    from episodic_memory import EpisodicMemoryStore
    store = EpisodicMemoryStore("episodes.jsonl")
    store.add_episode(session_id="s1", summary="User asked about Python decorators...")
    recent = store.get_recent_episodes(n=3)
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class EpisodicMemoryStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add_episode(self, session_id: str, summary: str, metadata: dict = None) -> dict:
        """
        Appends a new episode. metadata is a free-form dict for anything
        session-specific worth keeping alongside the summary (e.g. topic
        tags) -- kept separate from the summary text itself so retrieval
        logic can filter on structured fields without re-parsing prose.
        """
        episode = {
            "session_id": session_id,
            "summary": summary,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(episode) + "\n")
        return episode

    def get_all_episodes(self) -> list:
        if self.path.stat().st_size == 0:
            return []
        with open(self.path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_recent_episodes(self, n: int = 5) -> list:
        """Most recent n episodes, newest last -- matches the natural
        reading order for "here's what happened recently"."""
        all_episodes = self.get_all_episodes()
        return all_episodes[-n:]

    def get_episode_by_session_id(self, session_id: str) -> dict | None:
        for episode in self.get_all_episodes():
            if episode["session_id"] == session_id:
                return episode
        return None

    def count(self) -> int:
        return len(self.get_all_episodes())

    def replace_all(self, episodes: list):
        """
        Overwrites the entire store with a new list of episodes --
        used by consolidation/ when merging or pruning old episodes.
        A full rewrite rather than an in-place edit, since JSONL doesn't
        support efficient line-level updates.
        """
        with open(self.path, "w") as f:
            for episode in episodes:
                f.write(json.dumps(episode) + "\n")
