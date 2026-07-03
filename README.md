# FinOps Autonomous Auditor

See `PLAN.md` for full architecture and phase plan, `CLAUDE.md` for build principles.

> **LLM provider:** this project uses **OpenAI**, not Anthropic — no Anthropic API key
> is available. Any LLM-calling code (`agents/db_investigator.py` and later agents)
> goes through the `openai` SDK. Requires `OPENAI_API_KEY` in the environment.

## Phase 1 — deterministic core, no LLM

Two plain-SQL checks against a mock ERP database — no agents, no LLM calls.

```
data/
├── mock_erp.db      # SQLite: vendors (approved list) + paid_invoices (history)
└── invoices/         # labeled test invoices (JSON), each with an "expected" block
db/
└── checks.py         # is_vendor_approved, is_duplicate_amount
eval/
└── run_checks.py     # runs both checks against every test invoice, prints pass/fail
scripts/
└── seed_db.py         # builds + seeds data/mock_erp.db
```

### Run it

```bash
pip install -r requirements.txt
python3 scripts/seed_db.py   # (re)build the mock ERP database
python3 eval/run_checks.py   # verify the SQL checks against labeled invoices
```

### Checks

- **`is_vendor_approved(vendor)`** — is the vendor on the `vendors` table (case-insensitive)?
- **`is_duplicate_amount(vendor, amount, invoice_date)`** — has this vendor already been
  paid this exact amount (±0.001 tolerance) in the same calendar month as `invoice_date`?
  Duplicate scope is per-vendor, not global — two different vendors billing the same
  round amount in the same month isn't a duplicate-invoice signal.

### Test invoices

8 labeled invoices in `data/invoices/` cover: a clean pass, an unapproved vendor, a
same-vendor/same-amount/same-month duplicate, a same-amount-different-month case (to
verify month-scoping doesn't false-positive), a combined unapproved+duplicate case, and
(Phase 3) a legitimate recurring retainer that the naive duplicate check misflags.

## Phase 2 — text-to-SQL DB Investigator agent

Wraps the Phase 1 checks as an LLM-driven text-to-SQL loop, orchestrated with
LangGraph. Given an invoice and a natural-language check description, the model
writes a SQLite `SELECT`, we execute it against a **read-only** connection, and
retry with the error fed back on: non-`SELECT` SQL, a SQL execution error, or an
unexpected empty result (capped at 3 attempts per check).

```
graph/
├── state.py           # shared AuditState (Pydantic) + supporting models
└── workflow.py         # LangGraph graph: single db_investigator node
agents/
└── db_investigator.py  # text-to-SQL agent: generate -> execute -> verify -> retry
eval/
└── run_investigator.py # runs the agent against the Phase 1 labeled invoices, pass/fail
```

### Run it

```bash
cp .env.example .env   # then fill in OPENAI_API_KEY= in .env (gitignored, never committed)
python3 eval/run_investigator.py                       # full labeled-invoice eval
python3 graph/workflow.py data/invoices/inv_001_clean.json  # single invoice via the graph
```

`agents/db_investigator.py` loads `.env` automatically via `python-dotenv` — no need to
`export` the key in your shell. `.env` is in `.gitignore`.

## Phase 3 (current) — critic loop

Reviews the DB Investigator's flags before they become a verdict. The pipeline is
now `db_investigator -> flag_raiser -> critic` (LangGraph), where:

- **`flag_raiser`** (deterministic, no LLM) is deliberately naive/aggressive — it
  flags *any* unapproved vendor (severity `medium`) or *any* same-vendor+amount
  match in the same calendar month (severity `high`), with zero judgment. This is
  the "before" baseline.
- **`critic`** (LLM) reviews each raw flag against the vendor's full payment
  history and decides **confirm** (genuine risk) or **dismiss** (false positive —
  e.g. a biweekly retainer that legitimately repeats the same amount). It can
  request one extra round of evidence via a freeform text-to-SQL query before
  finalizing (capped at `MAX_CRITIC_ITERATIONS = 2`). The final `approve`/`flag`/
  `reject` verdict is **derived mechanically from the critic's own flag
  decisions** (confirmed-high → reject, any other confirmed → flag, all
  dismissed → approve) — not a separate freeform LLM judgment — so the verdict
  can never contradict the reasoning behind it.

```
agents/
├── flagging.py   # raise_raw_flags (naive baseline) + verdict_from_flags (mechanical rollup)
└── critic.py     # LLM critic: confirm/dismiss each flag + rationale, using vendor history
eval/
└── run_benchmark.py  # the headline metric: false-positive rate raw vs. critic-reviewed
```

### Run it

```bash
python3 eval/run_benchmark.py                              # false-positive rate: raw vs. critic
python3 graph/workflow.py data/invoices/inv_008_recurring_retainer_false_positive.json
```

### Current result

On the 8 labeled invoices (4 of which are truly clean):

| | Raw (pre-critic) | Critic-reviewed |
|---|---|---|
| Verdict accuracy | 88% | **100%** |
| False-positive rate | 25% | **0%** |

The one case the raw baseline gets wrong: a legitimate 6-month biweekly $2,400
retainer to Meridian Consulting Group, which the naive same-vendor+amount+month
rule flags as a duplicate. The critic reviews the vendor's payment history, sees
the consistent 14-day cadence going back 6+ months, and dismisses the flag.

Next: Phase 4 adds the Policy Assessor (RAG over policy docs) — see `PLAN.md`.
