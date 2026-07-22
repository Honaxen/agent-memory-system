# Agent Memory System

Work in progress -- this README is a placeholder and will be replaced once the project is complete.

Long-term memory for LLM agents -- episodic session summaries, semantic facts extracted and updated over time, memory-aware retrieval in the agent loop, and consolidation to keep memory from growing unbounded.

---

## What This Project Will Demonstrate

Every agent in this portfolio (multi-tool-agent, research-agent-langgraph, voice-rag-agent) is stateless -- each session starts from zero, with nothing carried over from prior interactions.
This project gives an agent a memory that persists and updates across sessions.

Concern -> Solution (planned)
- What happened in past sessions?                    -> Episodic memory: a summary per session, not raw transcripts
- What stable facts does the agent know about the user? -> Semantic memory: extracted facts, updated when they conflict with new information
- How does memory actually get used during a conversation? -> Retrieval in the agent loop: relevant memory pulled in before responding, the same way RAG pulls in documents
- What stops memory from growing forever?              -> Consolidation: summarizing or dropping low-value memory over time
- Does the agent actually remember, or just claim to?   -> A multi-session evaluation that tests recall across separate conversations, not within one

---

## Planned Architecture

Session 1 -> Episodic Summary -> Semantic Fact Extraction -> Memory Store
                                                                  |
Session 2 -> Memory Retrieval (agent/) -> Agent Response          |
                                                                  |
Session N -> Consolidation (consolidation/) -> pruned/summarized memory
                                                                  |
                                             Multi-Session Evaluation (evaluation/)

---

## Project Structure

agent-memory-system/
  memory/           - episodic summaries + semantic fact store
  agent/             - agent loop with memory retrieval
  consolidation/     - memory pruning/summarization over time
  evaluation/        - multi-session recall test
  tests/
  docs/

---

## Stack

Python - Ollama - FAISS - pytest

---

## Status

- [ ] Episodic memory (per-session summaries)
- [ ] Semantic memory (extracted facts, conflict resolution)
- [ ] Memory-aware agent loop
- [ ] Consolidation (pruning/summarization)
- [ ] Multi-session recall evaluation

---

## Related Projects

- [multi-tool-agent](https://github.com/Honaxen/multi-tool-agent) -- a stateless agent this project's memory layer could be added to
- [document-agent](https://github.com/Honaxen/document-agent) -- retrieval over external documents; this project retrieves over the agent's own memory instead

---

## Author

[Honaxen](https://github.com/Honaxen)
