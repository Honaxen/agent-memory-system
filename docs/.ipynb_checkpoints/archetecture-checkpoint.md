# Architecture

## Overview

This project gives an agent memory that persists and updates across
sessions, in four stages:

```
Session 1 (conversation happens)
    |
    v
Episodic Summary (memory/episodic_memory.py)     one summary per session, not raw transcript
    |
Semantic Fact Extraction (memory/extract_facts.py)  stable facts pulled from the conversation
    |
    v
Memory Store (episodes.jsonl + facts.json)
    |
    v
Session 2 (a genuinely separate process/conversation)
    |
    v
Memory Retrieval (agent/memory_agent.py)          facts + recent summaries pulled into context
    |
    v
Agent Response                                     informed by session 1, without session 1 present
    |
    v
Consolidation (consolidation/consolidate_memory.py) old episodes merged as memory grows
    |
    v
Multi-Session Evaluation (evaluation/)              does this actually work, measured against a stateless baseline?
```

Every stateless agent elsewhere in this portfolio (`multi-tool-agent`,
`research-agent-langgraph`, `voice-rag-agent`) starts every session with
nothing carried over. This project's entire purpose is the gap between
"session 2 has no access to session 1's conversation" and "session 2's
agent still knows what session 1 established" -- and proving that gap is
actually closed, not just claimed.

---

## Stage 1: Episodic Memory

`memory/episodic_memory.py` stores one summary per session in a flat
JSONL file, not the raw transcript. A 40-turn session shouldn't cost 40
turns of context on every future retrieval -- the summary, generated
once when the session ends, is what gets carried forward.

JSONL was chosen over a database for the same reason `llm-mlops-pipeline`'s
model registry uses a flat JSON file: simple, appendable, diff-friendly,
and entirely sufficient at the scale a single-user portfolio project
actually needs.

---

## Stage 2: Semantic Memory + Fact Extraction

`memory/semantic_memory.py` stores stable facts about the user (location,
occupation, etc.) as a key-value store, with explicit conflict handling:
a new fact that contradicts an old one *replaces* it, while the old value
moves into a `history` list rather than being silently lost. Facts are
extracted from conversation text by `memory/extract_facts.py`, which is
deliberately conservative -- only extracting what the user explicitly
stated or clearly implied. In this session's real run, the model
correctly extracted `location: Berlin` and `occupation: nurse` from a
transcript, while correctly declining to invent a fact from "I've been
really busy lately" -- exactly the restraint the prompt asked for, not
just a hoped-for outcome.

---

## Stage 3: Memory-Aware Agent Loop

`agent/memory_agent.py` builds a system prompt from both memory stores
before every turn -- the same basic shape as RAG, retrieving from the
agent's own memory of the user instead of external documents -- and, at
session end, summarizes the conversation and extracts new facts, writing
both back so a genuinely separate future session (a fresh process) can
retrieve them.

This was verified directly, not assumed: a real session 2 was asked "were
am i?" with zero mention of location in that session, and the agent
answered "You're in Berlin... As a nurse who recently moved," correctly
pulling both facts stored from session 1's fact extraction.

---

## Stage 4: Consolidation

`consolidation/consolidate_memory.py` prevents episodic memory from
growing without bound. Once the number of stored episodes exceeds a
"keep recent as-is" window, everything older is merged via an LLM call
into a single consolidated summary, replacing many old entries with one.
Recent episodes are never touched -- only episodes older than the window
get blurred together, so nothing about a recent conversation is lost to
this process.

Verified with a synthetic 4-episode test: with `keep_recent=1`, the three
oldest episodes were merged into one consolidated summary while the most
recent episode was left untouched, taking the total from 4 down to 2.

---

## Stage 5: Multi-Session Evaluation

`evaluation/multi_session_recall.py` is the stage that actually proves
the rest of this works, rather than asserting it. It scripts a session 1
that states several facts, then runs the *exact same* session-2 questions
against both the memory-equipped agent and a stateless baseline with
zero memory context -- the same questions, so the only variable is
whether memory was available.

The real run scored 2/3 "correct" per an automated LLM judge for the
memory-equipped agent versus 1/3 for the stateless baseline. Manually
inspecting the transcripts revealed the judge itself made an error: the
memory-equipped agent's answer to "what language do I work with" was
"You mentioned you mainly work with Go!" -- correct, and clearly sourced
from memory -- but the automated judge scored it wrong. By hand, the real
result was 3/3 correct with memory versus 1/3 without. This is reported
as-is rather than smoothed over, because it's a second, independent
instance of the same lesson `llm-eval-statistics` and `llm-safety-redteam`
already surfaced elsewhere in this portfolio: an LLM-as-judge is a
convenience, not a source of ground truth, and its verdicts need
spot-checking, not blind trust.

---

## Why This Order

- Fact extraction (Stage 2) needs episodic storage (Stage 1) established
  first, since both write to memory that Stage 3 will later read from.
- The agent loop (Stage 3) needs both memory stores populated before
  retrieval means anything -- there's nothing to retrieve from an empty
  store.
- Consolidation (Stage 4) only matters once memory has actually
  accumulated -- it's a maintenance concern for a system that's already
  working, not a prerequisite for it working at all.
- The evaluation (Stage 5) had to come last because it needed every
  other piece in place to run a genuine two-session test with a
  meaningful stateless comparison -- and, in this case, it's also what
  caught a real flaw (the judge's own error) that wouldn't have surfaced
  without actually running the full loop end to end.