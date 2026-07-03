# FinOps Autonomous Auditor

## What this is
A multi-agent invoice-auditing system. Agents: DB Investigator (text-to-SQL
vendor + duplicate checks), Policy Assessor (RAG over policy docs), Critic
(kills false-positive flags, recalibrates risk). Orchestrated with LangGraph.

## Build principles
- Build in phases. Do NOT scaffold all agents at once.
- Phase 1 is deterministic SQL checks with NO LLM. Get ground truth solid first.
- Input is pre-parsed invoice JSON. OCR/extraction is LAST, or out of scope.
- Every agent reads/writes the shared AuditState (Pydantic).
- Keep the policy RAG corpus small (5-10 docs).

## Stack
Python, LangGraph, ChromaDB, SQLite, Pydantic, Streamlit.

## Evaluation is the point
The headline metric is false-positive rate before vs. after the critic agent.
Build the labeled test set early; this project is judged on rigor, not features.

## Reference
Full architecture, file structure, and phase plan are in PLAN.md. Read it before
starting a new phase. The current phase is tracked at the top of that file.