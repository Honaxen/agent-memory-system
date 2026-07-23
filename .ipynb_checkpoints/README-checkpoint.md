# Agent Memory System

Long-term memory for LLM agents — episodic session summaries, semantic facts extracted and updated over time, memory-aware retrieval in the agent loop, and consolidation to keep memory from growing unbounded. Every result below is from a real run against a real model, not a projection.

---

## What This Project Demonstrates

Every agent in this portfolio (`multi-tool-agent`, `research-agent-langgraph`, `voice-rag-agent`) is stateless — each session starts from zero, with nothing carried over from prior interactions.
This project gives an agent a memory that persists and updates across sessions, and proves it with a real cross-session test, not just an architecture diagram.

| Concern | Solution |
|---|---|
| What happened in past sessions? | Episodic memory: a summary per session, not raw transcripts |
| What stable facts does the agent know about the user? | Semantic memory: extracted facts, updated when they conflict with new information |
| How does memory actually get used during a conversation? | Retrieval in the agent loop: relevant memory pulled in before responding, the same way RAG pulls in documents |
| What stops memory from growing forever? | Consolidation: summarizing old episodes into one, without touching recent ones |
| Does the agent actually remember, or just claim to? | A multi-session test comparing memory-equipped vs. stateless recall on the same questions |

---

## Architecture

```
Session 1 (conversation happens)
  ↓
Episodic Summary + Semantic Fact Extraction
  ↓
Memory Store (episodes.jsonl + facts.json)
  ↓
Session 2 (a genuinely separate process)
  ↓
Memory Retrieval → Agent Response
  ↓
Consolidation (as memory grows)
  ↓
Multi-Session Evaluation (vs. a stateless baseline)
```

---

## Project Structure

```
agent-memory-system/
├── memory/
│   ├── episodic_memory.py      — JSONL-based per-session summary store
│   ├── semantic_memory.py      — key-value fact store with conflict resolution + history
│   └── extract_facts.py        — LLM-based fact extraction from transcripts
├── agent/
│   └── memory_agent.py         — agent loop: retrieve memory → respond → summarize + extract at session end
├── consolidation/
│   └── consolidate_memory.py   — merges old episodes once a threshold is exceeded
├── evaluation/
│   └── multi_session_recall.py — memory-equipped vs. stateless recall test
├── tests/
│   └── test_memory_system.py   — 15/15 passing
├── docs/
│   └── architecture.md
└── requirements.txt
```

---

## Getting Started

```bash
pip install -r requirements.txt
ollama serve
```

### 1. Extract facts from a conversation

```bash
python memory/extract_facts.py \
  --transcript "User: I just moved to Berlin last month. I work as a nurse. I've been really busy lately." \
  --facts_path memory/facts.json \
  --session_id s1 \
  --model gemma3:12b
```

**Actual output from this run:**
```
Extracted 3 fact(s) from session s1:
  location: Berlin
  occupation: nurse
  move_date: last month
```
Notice what's *not* there: no invented fact about stress, personality, or workload from "I've been really busy lately" — the extraction stayed restrained to what was actually stated.

### 2. Run a memory-aware session

```bash
python agent/memory_agent.py --session_id s2 --model gemma3:12b \
  --facts_path memory/facts.json \
  --episodes_path memory/episodes.jsonl
```

**Actual conversation from this run**, with zero mention of location in the current session:
```
You: were am i?
Agent: You're in Berlin! As a nurse who recently moved last month, I hope
       you're settling in well. 😊 Is there anything I can help you with
       regarding your new location?
```

### 3. Consolidate old memory

```bash
python consolidation/consolidate_memory.py \
  --episodes_path memory/episodes.jsonl \
  --keep_recent 5 \
  --model gemma3:12b
```

Verified on a synthetic 4-episode set with `keep_recent=1`: the 3 oldest episodes merged into one consolidated summary, the most recent left untouched — episode count went from 4 to 2.

### 4. Run the multi-session recall test

```bash
python evaluation/multi_session_recall.py \
  --model gemma3:12b \
  --facts_path memory/eval_facts.json \
  --episodes_path memory/eval_episodes.jsonl \
  --output evaluation/results/recall_test.json
```

**Actual output from this run:**
```
With memory:    2/3 correct (per automated judge)
Without memory: 1/3 correct
```
Manually inspecting the transcripts found the automated judge scored one answer wrong that was actually correct — the memory-equipped agent answered "You mentioned you mainly work with Go!", clearly pulled from stored memory, but the judge marked it incorrect. By hand, the real result was **3/3 with memory vs. 1/3 without**. Reported here as-is rather than smoothed over — an LLM-as-judge verdict needs spot-checking, not blind trust.

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Stack

Python · Ollama · pytest

---

## What I Learned

**Separating episodic and semantic memory wasn't just theoretical — the two serve genuinely different retrieval needs.**
"User discussed moving to Berlin in session 3" (episodic) and "location: Berlin" (semantic) answer different questions. A prompt asking "where do I live" needs the fact, directly; a prompt asking "what have we talked about lately" needs the summary. Conflating them into one store would have made retrieval worse at both jobs.

**Fact extraction restraint has to be designed in, not hoped for.**
It would have been easy to let the extraction prompt infer loosely ("busy lately" → "stressed," "high workload," etc.). Explicitly instructing the model to extract only clearly stated facts, and verifying that restraint against a real transcript, was what actually prevented the memory store from filling up with invented details presented as remembered fact.

**A stateless baseline is what makes "the agent remembers" a checkable claim, not an assumption.**
Running the identical session-2 questions against both a memory-equipped agent and a no-memory control is what turns "the agent answered correctly" into "the agent answered correctly *because* of memory, not by coincidence or a lucky guess."

**An LLM-as-judge error, caught by actually reading the transcripts, was the most valuable finding in this whole project.**
The automated judge said 2/3; manual inspection said 3/3. That one-answer discrepancy is a small, concrete instance of a lesson this portfolio keeps re-learning: automated grading is a convenience that still needs a human to check its work, especially on the exact claims a project is built to prove.

**Consolidation only needs to touch what's actually old.**
Leaving recent episodes completely untouched, and only merging past a fixed window, meant the synthetic test could verify the exact boundary (3 old episodes merged, 1 recent left alone) without any ambiguity about what should or shouldn't have changed.

---

## Related Projects

- [multi-tool-agent](https://github.com/Honaxen/multi-tool-agent) — a stateless agent this project's memory layer could be added to
- [document-agent](https://github.com/Honaxen/document-agent) — retrieval over external documents; this project retrieves over the agent's own memory instead
- [llm-eval-statistics](https://github.com/Honaxen/llm-eval-statistics) · [llm-safety-redteam](https://github.com/Honaxen/llm-safety-redteam) — the same "LLM-as-judge needs spot-checking" lesson, found independently here again

---

## Author

[Honaxen](https://github.com/Honaxen)