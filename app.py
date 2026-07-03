"""
Streamlit frontend for the FinOps Autonomous Auditor (Phase 5).

Two views:
  - Audit an invoice: run a single invoice (a labeled test case, or a custom
    one built from a form) through the full pipeline and see every stage's
    output - DB findings + generated SQL, raw flags, policy flags, and the
    critic's confirm/dismiss review with rationale.
  - Evaluation: reruns the Phase 1-4 labeled benchmark and shows the
    false-positive-rate / accuracy delta the critic buys you - the project's
    headline metric (see CLAUDE.md).

Extraction/OCR (turning a raw scanned invoice into the structured JSON this
app consumes) is out of scope - documented here as future work. Input is
always the same pre-parsed invoice JSON used throughout Phases 1-4.

Requires OPENAI_API_KEY - set as a Space secret on Hugging Face, or in a local
.env file.
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="FinOps Autonomous Auditor", page_icon="🧾", layout="wide")

if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is not set. On Hugging Face Spaces: Settings -> "
        "Variables and secrets -> add a secret named OPENAI_API_KEY. "
        "Locally: put it in a .env file. Then reload this page."
    )
    st.stop()

POLICY_INDEX_PATH = ROOT / "data" / "policy_index"
if not POLICY_INDEX_PATH.exists():
    with st.spinner("First run: building the policy RAG index..."):
        from scripts.build_policy_index import build_index

        build_index()

from agents.flagging import verdict_from_flags
from graph.state import AuditState
from graph.workflow import build_graph

INVOICES_DIR = ROOT / "data" / "invoices"
VERDICT_COLOR = {"approve": "green", "flag": "orange", "reject": "red"}

st.title("🧾 FinOps Autonomous Auditor")
st.caption(
    "Deterministic DB checks → text-to-SQL DB Investigator → RAG Policy Assessor → "
    "Critic. Input is pre-parsed invoice JSON — extraction/OCR from a raw scanned "
    "invoice is **future work**, not built here."
)

tab_audit, tab_eval = st.tabs(["Audit an invoice", "Evaluation (headline metric)"])


def render_result(result: dict) -> None:
    verdict = result["final_verdict"]
    color = VERDICT_COLOR.get(verdict.decision, "gray")
    st.markdown(f"### Verdict: :{color}[{verdict.decision.upper()}]")
    st.write(verdict.rationale)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("DB Investigator")
        db = result["db_findings"]
        st.write(f"**Vendor approved:** {db.vendor_approved.result}  (attempts={db.vendor_approved.attempts})")
        st.code(db.vendor_approved.generated_sql, language="sql")
        st.write(f"**Duplicate this month:** {db.is_duplicate.result}  (attempts={db.is_duplicate.attempts})")
        st.code(db.is_duplicate.generated_sql, language="sql")

    with col2:
        st.subheader("Raw flags (pre-critic)")
        all_raw = result["raw_flags"] + result["policy_flags"]
        if not all_raw:
            st.write("_None raised._")
        for f in all_raw:
            st.write(f"- **[{f.source}] {f.check}** (severity={f.severity}): {f.message}")

    st.subheader("Critic review")
    if not result["reviewed_flags"]:
        st.write("_No flags were raised, so the critic had nothing to review._")
    for f in result["reviewed_flags"]:
        icon = "✅ dismissed" if f.status == "dismissed" else "⚠️ confirmed"
        st.write(f"- {icon} — **{f.check}** (severity={f.severity}): {f.rationale}")


with tab_audit:
    st.subheader("Pick a labeled test invoice, or build a custom one")

    invoice_files = sorted(INVOICES_DIR.glob("*.json"))
    labels = ["(custom invoice)"] + [f.name for f in invoice_files]
    choice = st.selectbox("Invoice", labels)

    if choice == "(custom invoice)":
        vendor = st.text_input("Vendor", "Acme Office Supplies")
        amount = st.number_input("Amount ($)", min_value=0.0, value=500.0, step=0.01)
        invoice_date = st.date_input("Invoice date")
        description = st.text_area("Description", "")
        invoice = {
            "invoice_id": "CUSTOM-1",
            "vendor": vendor,
            "amount": amount,
            "invoice_date": str(invoice_date),
            "description": description,
        }
    else:
        raw = json.loads((INVOICES_DIR / choice).read_text())
        expected_verdict = raw.get("expected_verdict")
        note = raw.get("note")
        invoice = {k: v for k, v in raw.items() if k not in ("expected", "expected_verdict", "note")}
        st.json(invoice)
        if expected_verdict:
            st.caption(f"Expected verdict (ground truth): **{expected_verdict}**")
        if note:
            st.caption(note)

    if st.button("Run audit", type="primary"):
        try:
            with st.spinner("Running db_investigator → flag_raiser → policy_assessor → critic..."):
                app = build_graph()
                result = app.invoke(AuditState(raw_invoice=invoice))
            render_result(result)
        except Exception as e:
            st.error(f"Audit failed: {e}")

with tab_eval:
    st.subheader("Phase 1-4 labeled benchmark")
    st.write(
        "Reruns every labeled invoice through the full pipeline and compares the "
        "naive pre-critic verdict against the critic-reviewed verdict, using each "
        "invoice's known-correct `expected_verdict`. This is the project's headline "
        "metric (CLAUDE.md): false-positive rate before vs. after the critic."
    )

    if st.button("Run benchmark (10 invoices, ~1-2 min)"):
        app = build_graph()
        files = sorted(INVOICES_DIR.glob("*.json"))
        rows = []
        progress = st.progress(0.0)
        for i, path in enumerate(files):
            invoice = json.loads(path.read_text())
            expected = invoice["expected_verdict"]
            clean = {k: v for k, v in invoice.items() if k not in ("expected", "expected_verdict", "note")}
            result = app.invoke(AuditState(raw_invoice=clean))
            raw_verdict = verdict_from_flags(result["raw_flags"] + result["policy_flags"])
            critic_verdict = result["final_verdict"].decision
            rows.append(
                {
                    "invoice": path.name,
                    "expected": expected,
                    "raw (pre-critic)": raw_verdict,
                    "critic-reviewed": critic_verdict,
                }
            )
            progress.progress((i + 1) / len(files))
        st.session_state["benchmark_rows"] = rows

    rows = st.session_state.get("benchmark_rows")
    if rows:
        st.dataframe(rows, use_container_width=True)

        clean_rows = [r for r in rows if r["expected"] == "approve"]

        def fp_rate(key: str) -> float:
            if not clean_rows:
                return 0.0
            return sum(1 for r in clean_rows if r[key] != "approve") / len(clean_rows)

        def accuracy(key: str) -> float:
            return sum(1 for r in rows if r[key] == r["expected"]) / len(rows)

        c1, c2 = st.columns(2)
        c1.metric(
            "Verdict accuracy",
            f"{accuracy('critic-reviewed'):.0%}",
            delta=f"{(accuracy('critic-reviewed') - accuracy('raw (pre-critic)')) * 100:+.0f} pts vs. raw",
        )
        c2.metric(
            "False-positive rate",
            f"{fp_rate('critic-reviewed'):.0%}",
            delta=f"{(fp_rate('critic-reviewed') - fp_rate('raw (pre-critic)')) * 100:+.0f} pts vs. raw",
            delta_color="inverse",
        )

st.divider()
st.caption(
    "Architecture details: PLAN.md. Build principles: CLAUDE.md. Source: "
    "https://github.com/tankaihooi/finops-auditor"
)
