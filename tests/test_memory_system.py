"""
Unit tests for the pure logic in this project: semantic memory's conflict
resolution and history tracking, episodic memory's storage/retrieval,
consolidation's threshold logic, and memory-context building. No Ollama
calls are made -- extract_facts.py's and memory_agent.py's actual LLM
calls are a manual/integration concern, verified in this session's real
runs (logged in README.md) rather than mocked here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "memory"))
sys.path.insert(0, str(Path(__file__).parent.parent / "consolidation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from semantic_memory import SemanticMemoryStore  # noqa: E402
from episodic_memory import EpisodicMemoryStore  # noqa: E402
from consolidate_memory import consolidate  # noqa: E402
from memory_agent import build_memory_context  # noqa: E402


# --- SemanticMemoryStore tests ---

@pytest.fixture
def facts_store(tmp_path):
    return SemanticMemoryStore(str(tmp_path / "facts.json"))


def test_upsert_new_fact_has_no_history(facts_store):
    entry = facts_store.upsert_fact("location", "Tehran", source_session="s1")
    assert entry["value"] == "Tehran"
    assert entry["history"] == []


def test_upsert_conflicting_fact_replaces_value_and_keeps_history(facts_store):
    facts_store.upsert_fact("location", "Tehran", source_session="s1")
    entry = facts_store.upsert_fact("location", "Berlin", source_session="s5")

    assert entry["value"] == "Berlin"
    assert len(entry["history"]) == 1
    assert entry["history"][0]["value"] == "Tehran"


def test_upsert_same_value_again_does_not_add_history():
    # Re-confirming a known fact isn't a conflict -- shouldn't pollute history
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SemanticMemoryStore(f"{tmpdir}/facts.json")
        store.upsert_fact("location", "Berlin", source_session="s1")
        entry = store.upsert_fact("location", "Berlin", source_session="s2")
        assert entry["history"] == []


def test_get_fact_returns_none_for_unknown_key(facts_store):
    assert facts_store.get_fact("nonexistent") is None


def test_get_all_facts_returns_everything(facts_store):
    facts_store.upsert_fact("location", "Berlin")
    facts_store.upsert_fact("occupation", "nurse")
    all_facts = facts_store.get_all_facts()
    assert set(all_facts.keys()) == {"location", "occupation"}


def test_delete_fact_removes_it(facts_store):
    facts_store.upsert_fact("location", "Berlin")
    facts_store.delete_fact("location")
    assert facts_store.get_fact("location") is None


# --- EpisodicMemoryStore tests ---

@pytest.fixture
def episodes_store(tmp_path):
    return EpisodicMemoryStore(str(tmp_path / "episodes.jsonl"))


def test_add_episode_increases_count(episodes_store):
    episodes_store.add_episode("s1", "First session summary.")
    assert episodes_store.count() == 1


def test_get_recent_episodes_returns_newest_last(episodes_store):
    episodes_store.add_episode("s1", "First.")
    episodes_store.add_episode("s2", "Second.")
    episodes_store.add_episode("s3", "Third.")

    recent = episodes_store.get_recent_episodes(n=2)
    assert len(recent) == 2
    assert recent[-1]["session_id"] == "s3"  # newest last


def test_get_episode_by_session_id_finds_correct_entry(episodes_store):
    episodes_store.add_episode("s1", "First.")
    episodes_store.add_episode("s2", "Second.")

    found = episodes_store.get_episode_by_session_id("s2")
    assert found["summary"] == "Second."


def test_get_episode_by_session_id_returns_none_when_missing(episodes_store):
    assert episodes_store.get_episode_by_session_id("nonexistent") is None


# --- consolidation threshold logic tests ---

def test_consolidate_skips_when_at_or_below_keep_recent(tmp_path):
    episodes_path = str(tmp_path / "episodes.jsonl")
    store = EpisodicMemoryStore(episodes_path)
    store.add_episode("s1", "One.")
    store.add_episode("s2", "Two.")

    result = consolidate(episodes_path, keep_recent=5, model="unused", min_to_consolidate=1)
    assert result["consolidated"] is False
    assert "keep_recent" in result["reason"]


def test_consolidate_skips_when_below_min_to_consolidate(tmp_path):
    episodes_path = str(tmp_path / "episodes.jsonl")
    store = EpisodicMemoryStore(episodes_path)
    for i in range(4):
        store.add_episode(f"s{i}", f"Session {i}.")

    # keep_recent=3 leaves only 1 old episode, below min_to_consolidate=3
    result = consolidate(episodes_path, keep_recent=3, model="unused", min_to_consolidate=3)
    assert result["consolidated"] is False
    assert "min_to_consolidate" in result["reason"]


# --- build_memory_context tests ---

def test_memory_context_handles_empty_stores(tmp_path):
    facts_store = SemanticMemoryStore(str(tmp_path / "facts.json"))
    episodes_store = EpisodicMemoryStore(str(tmp_path / "episodes.jsonl"))

    context = build_memory_context(facts_store, episodes_store)
    assert "no facts recorded yet" in context
    assert "no past sessions recorded yet" in context


def test_memory_context_includes_stored_facts(tmp_path):
    facts_store = SemanticMemoryStore(str(tmp_path / "facts.json"))
    episodes_store = EpisodicMemoryStore(str(tmp_path / "episodes.jsonl"))
    facts_store.upsert_fact("location", "Berlin")

    context = build_memory_context(facts_store, episodes_store)
    assert "location: Berlin" in context


def test_memory_context_respects_n_recent_episodes_limit(tmp_path):
    facts_store = SemanticMemoryStore(str(tmp_path / "facts.json"))
    episodes_store = EpisodicMemoryStore(str(tmp_path / "episodes.jsonl"))
    episodes_store.add_episode("s1", "Summary one.")
    episodes_store.add_episode("s2", "Summary two.")
    episodes_store.add_episode("s3", "Summary three.")

    context = build_memory_context(facts_store, episodes_store, n_recent_episodes=1)
    assert "Summary three." in context
    assert "Summary one." not in context
