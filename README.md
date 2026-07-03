---
title: FinOps Autonomous Auditor
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# FinOps Autonomous Auditor

A multi-agent invoice-auditing pipeline that catches genuine risk (unapproved
vendors, duplicate payments, missing spend approvals) without drowning humans
in false positives — the differentiator being an independent Critic agent
that reviews every automated flag against real evidence before it becomes a
verdict.

**Live demo:** https://huggingface.co/spaces/tankaihooi/finops-auditor

## Headline result

On a labeled set of 10 invoices (5 of them genuinely clean):

| | Naive rules (pre-critic) | Critic-reviewed |
|---|---|---|
| Verdict accuracy | 90% | **100%** |
| False-positive rate | 20% | **0%** |

The naive baseline — the same mechanical rules a first pass at this problem
would use — flags a legitimate biweekly consulting retainer as a duplicate
payment because it's paid twice in the same calendar month. The critic
reviews the vendor's 6-month payment history, recognizes the consistent
14-day cadence, and dismisses it. That one save is the whole point of the
architecture: **catching what rules miss is easy; not crying wolf on what
rules get wrong is the hard part.**

## Architecture

```
invoice JSON
     |
     v
┌──────────────────────┐
│ DB Investigator      │   generates its own SQL against a read-only
│ (text-to-SQL, LLM)   │   connection; retries on SQL error/empty result
│                      │   -> vendor_approved, is_duplicate
└──────────────────────┘
          |
          v
┌──────────────────────┐
│ Flag Raiser          │   deliberately naive: flags any unapproved vendor
│ (deterministic)      │   or same-vendor+amount+month match, no judgment
│                      │   -> raw_flags
└──────────────────────┘
          |
          v
┌──────────────────────┐
│ Policy Assessor      │   RAG over 6 policy docs (ChromaDB); flags spend
│ (RAG, LLM)           │   over a category's approval threshold, unless the
│                      │   invoice's own description already shows approval
│                      │   -> policy_flags
└──────────────────────┘
          |
          v
┌──────────────────────┐
│ Critic               │   confirms genuine risk / dismisses false positives
│ (LLM, loops)         │   using vendor payment history; can pull one more
│                      │   round of evidence before finalizing
│                      │   -> reviewed_flags + final_verdict
└──────────────────────┘
          |
          v
   approve / flag / reject
```

Orchestrated with LangGraph. The final verdict is **derived mechanically**
from the critic's own confirm/dismiss decisions (confirmed-high → reject, any
other confirmed → flag, all dismissed → approve) rather than a separate
freeform judgment call — so the verdict can never contradict the reasoning
behind it.

## Quickstart

```bash
git clone https://github.com/tankaihooi/finops-auditor.git
cd finops-auditor
pip install -r requirements.txt
cp .env.example .env               # then fill in OPENAI_API_KEY=

python3 scripts/seed_db.py         # build the mock ERP database
python3 scripts/build_policy_index.py  # embed + index the policy docs

streamlit run app.py               # interactive UI
# or:
python3 eval/run_benchmark.py      # headline metric from the CLI
```

## Project structure

```
data/
├── mock_erp.db          # SQLite: vendors (approved list) + paid_invoices (history)
├── invoices/             # 10 labeled test invoices (JSON), each with expected results
├── policies/             # 6 small policy docs (RAG corpus)
└── policy_index/         # ChromaDB index built from policies/ (gitignored, rebuilt on demand)
db/
└── checks.py             # Phase 1 ground truth: plain-SQL vendor/duplicate checks
agents/
├── db_investigator.py    # text-to-SQL: generate → execute → verify → retry
├── flagging.py           # naive raw-flag baseline + mechanical verdict rollup
├── policy_assessor.py    # RAG: retrieve → LLM violation check → Flag(s)
└── critic.py             # confirm/dismiss flags using vendor history, loops on evidence
graph/
├── state.py              # shared AuditState (Pydantic) every agent reads/writes
└── workflow.py           # LangGraph pipeline definition
eval/
├── run_checks.py         # verifies the Phase 1 SQL checks
├── run_investigator.py   # verifies the text-to-SQL agent against ground truth
└── run_benchmark.py      # the headline metric: false-positive rate raw vs. critic
scripts/
├── seed_db.py             # builds data/mock_erp.db
└── build_policy_index.py  # embeds + indexes data/policies/
app.py                     # Streamlit frontend
```

## How each piece works

**DB Investigator** — given a natural-language check description, an LLM
writes a SQLite query, we execute it against a read-only connection, and feed
back the error (or an unexpected empty result) for up to 3 retries. This is
the execute → verify → iterate loop, not a single LLM call trusted blindly.

**Flag Raiser** — pure Python, no LLM. Flags *any* unapproved vendor (severity
`medium`) or *any* same-vendor+amount match in the same calendar month
(severity `high`). This naive baseline is the "before" side of the headline
metric.

**Policy Assessor** — embeds the invoice (vendor, amount, description) and
retrieves the top-3 relevant docs from 6 small policy documents (software
subscriptions, professional services, equipment, marketing, travel, plus a
recurring-payment exemption). An invoice whose own description already shows
approval evidence (a ticket number, a named approver) isn't flagged in the
first place.

**Critic** — the differentiator. Reviews every raw flag (DB + policy) against
the vendor's full payment history and decides confirm or dismiss, with a
rationale. Can request one extra round of evidence via a freeform text-to-SQL
query before finalizing (capped at 2 iterations, enforced in code regardless
of what the model asks for).

**Streamlit UI** (`app.py`) — a thin layer over the same pipeline the eval
scripts use, so there's no separate app logic to keep in sync: audit any
labeled invoice or a custom one and see every stage, or rerun the labeled
benchmark and watch the false-positive-rate delta update live.

## Evaluation

The 10 labeled invoices in `data/invoices/` carry two kinds of ground truth:
a check-level `expected` block (what the DB Investigator's SQL should return)
and an `expected_verdict` (the correct business outcome). `eval/run_benchmark.py`
reruns all of them through the full pipeline and reports both the naive and
critic-reviewed verdict against that ground truth — the numbers in the
headline table above.

**A real failure mode surfaced during testing:** retrieval over a small,
generic policy corpus doesn't cleanly separate relevant from irrelevant docs
(checked directly — distances for genuinely unrelated invoices were
comparable to relevant matches). Left unguarded, the Policy Assessor would
cite a retrieved-but-irrelevant doc as "violated," or infer a spend category
from a vendor's name alone (a vendor named "...Software" isn't automatically
a software subscription). Fixed via a restricted violation-policy enum
(the recurring-payment exemption doc can never itself be "violated") and
explicit prompt guidance to match on the invoice's description, not the
vendor's name — not via a distance threshold, which doesn't work here.

## What's not built

- **Extraction/OCR.** Every input path — CLI eval scripts and the UI — takes
  pre-parsed invoice JSON. Turning a raw scanned invoice into that structured
  JSON is a distinct, well-scoped problem (layout parsing, field extraction,
  confidence scoring) that's deliberately out of scope here so the agentic
  and evaluation work could be the focus. Documented as future work, not a
  gap that was missed.
- Multi-tenant/auth, a real ERP integration (vs. the mock SQLite DB), and
  human-in-the-loop approval workflows for `flag` verdicts are also out of
  scope — this is a focused demonstration of the audit pipeline and its
  evaluation methodology, not a production finance tool.

## Deploying to Hugging Face Spaces

The app is a single `app.py` + `requirements.txt`, so it deploys as-is. Set
`OPENAI_API_KEY` as a Space secret (Settings → Variables and secrets) — never
commit it. The policy index isn't checked into git (it's a non-deterministic
binary artifact); the app builds it automatically on first load, which costs
one small round of embedding calls and a few seconds.

This repo is dual-hosted (GitHub + a Hugging Face Space) from the same
`README.md` — the YAML block at the very top is Spaces config, required
there to select the Streamlit runtime and entry point; it just renders as
plain text on GitHub.

## Stack

Python · OpenAI (`gpt-4.1`, `text-embedding-3-small`) · LangGraph · ChromaDB ·
SQLite · Pydantic · Streamlit
