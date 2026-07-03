# FinOps Autonomous Auditor — Architecture & Plan

> **CURRENT PHASE: 5** — Streamlit frontend (app.py) over the Phase 1-4 pipeline.
> Extraction/OCR is documented as future work, not built (per this phase's own
> "leave as JSON-only" option). LLM calls in this project go through **OpenAI**,
> not Anthropic (no Anthropic API key available).
> Update this line as phases complete. Claude Code should read this file before starting any new phase.

---

## What this project is

A multi-agent invoice-auditing system that ingests an invoice, checks it against
a mock ERP database and a set of company expense policies, and returns an
approve / flag / reject verdict — with a critic agent that recalibrates flags to
kill false positives. Orchestrated with LangGraph.

This is the autonomous, multi-agent version of the expense-auditing problem
already built in production at KVC (Copilot Studio + AI Builder + SAP Concur).
The portfolio angle is the independent-verification loop, not the feature count.

---

## Build principles (do not violate)

- **Build in phases.** Do NOT scaffold all agents at once. Each phase must work
  and be verified before the next begins.
- **Phase 1 has NO LLM.** Deterministic SQL first, so ground truth is solid and
  checkable before any agentic layer is added.
- **Input is pre-parsed invoice JSON.** OCR / extraction is the LAST phase, or
  explicitly out of scope. Do not start with OCR.
- **Every agent reads/writes the shared `AuditState`** (Pydantic). No agent holds
  private state.
- **Keep the policy RAG corpus small** (5–10 clean docs). Corpus mismatch was the
  documented failure mode in the prior audit-tool project — don't repeat it.
- **Evaluation is the deliverable, not a nice-to-have.** See the Evaluation
  section. The headline metric is false-positive rate before vs. after the critic.

---

## Stack

Python · LangGraph · ChromaDB · SQLite · Pydantic · Streamlit

---

## Repo structure

```
finops-auditor/
├── data/
│   ├── mock_erp.db          # SQLite: vendors, paid_invoices tables
│   ├── policies/            # 5-10 policy docs for RAG
│   └── invoices/            # test invoices as JSON (OCR later)
├── agents/
│   ├── extraction.py        # JSON → validated Pydantic schema (Phase 5)
│   ├── db_investigator.py   # text-to-SQL vendor + duplicate checks
│   ├── policy_assessor.py   # RAG over policy docs
│   └── critic.py            # recalibrates flags, kills false positives
├── graph/
│   ├── state.py             # shared AuditState (Pydantic)
│   └── workflow.py          # LangGraph graph definition
├── eval/
│   ├── test_cases.py        # labeled invoices + expected verdicts
│   └── run_benchmark.py     # accuracy + false-positive rate
├── app.py                   # Streamlit frontend
└── README.md
```

Note: the filesystem is the source of truth for structure. This tree is the
*intended* target — when it drifts, trust the actual directory, not this diagram.

---

## Shared state object

The heart of the design. Every agent reads from and writes to this. Get it right
before building agents around it.

```python
class AuditState(BaseModel):
    raw_invoice: dict
    extracted: InvoiceSchema | None = None      # extraction fills (Phase 5)
    db_findings: DBFindings | None = None        # investigator fills
    policy_flags: list[Flag] = []                # assessor fills
    final_verdict: Verdict | None = None         # critic fills
    critic_iterations: int = 0                   # loop guard
```

---

## Phase plan

Each phase leaves you with something that runs. Nothing after Phase 1 is
load-bearing for a demo — it's all additive.

### Phase 1 — deterministic core, no LLM
Build `mock_erp.db` (a `vendors` table and a `paid_invoices` table). Write the DB
Investigator checks as plain SQL — approved-vendor check, and duplicate-amount-
this-month check. No agent, no LLM. Write a few test invoices as JSON where the
correct answer is known, plus a script that runs the checks against them so the
SQL is verifiably correct. This is the ground-truth foundation; if it isn't
right, an LLM won't fix it.

### Phase 2 — wrap the DB check as a text-to-SQL agent
Let the LLM generate SQL from the invoice + a natural-language check description,
execute it, and retry on SQL error or unexpected-empty result. This is the
execute → verify → iterate loop, in LangGraph.

### Phase 3 — add the critic loop
Critic reviews the investigator's flags, kills false positives (the $0.05
rounding-error → "fraud" case), adjusts risk score, can route back to rewrite.
Cap iterations with `critic_iterations`. This is the differentiating agent — make
its rejection logic real, not cosmetic. Interviewers will probe it.

### Phase 4 — add policy RAG
Bring in the Policy Assessor: ChromaDB over 5–10 policy docs, checks line items
against policy rules (e.g. "software subscriptions over $500 require VP
approval"). Reuse the ChromaDB stack from the prior audit tool. Keep the corpus
small and clean.

### Phase 5 — extraction + frontend
Add the OCR/extraction agent last, OR leave it as JSON-only and document
"extraction layer designed, OCR is future work" as an honest scope boundary. Wire
up the Streamlit frontend.

---

## Evaluation (what makes this *the* project, not a demo)

Build a labeled test set where the correct verdict for each invoice is known.
Report:

- **Verdict accuracy** — does the system reach the right approve/flag/reject call?
- **False-positive rate, before vs. after the critic** — the money metric. It
  quantifies exactly what the critic agent buys you. Target sentence for the
  writeup: "The critic reduced false-positive fraud flags from X% to Y%."
- **Failure modes** — which invoices it still gets wrong, and why.

Build the labeled test set early (by Phase 2), not at the end. The whole writeup
is built around the false-positive delta — it's the FinOps equivalent of the
"RAG actually underperformed" finding from the prior project: a result that reads
as genuine investigation rather than a happy-path demo.
