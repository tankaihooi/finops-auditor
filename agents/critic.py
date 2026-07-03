"""
Critic agent (Phase 3, extended in Phase 4): reviews all raised flags - both
the DB Investigator's raw flags (vendor approval, duplicate payment) and the
Policy Assessor's RAG-based flags (spend-category approval requirements) -
against the flagged vendor's full payment history, and decides per flag
whether it's a genuine risk (confirm) or a false positive (dismiss) - e.g. a
biweekly retainer that a naive same-vendor/amount/month rule misreads as a
duplicate payment. Produces the final verdict.

Can request one additional round of evidence via a freeform text-to-SQL query
before finalizing, capped by MAX_CRITIC_ITERATIONS (loop guard lives in both
this module and graph/workflow.py's routing).

Requires OPENAI_API_KEY in the environment.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agents.db_investigator import (
    DB_PATH,
    MODEL,
    get_vendor_history,
    open_readonly_connection,
    run_freeform_query,
)
from agents.flagging import verdict_from_flags
from graph.state import DBFindings, Flag, Verdict

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MAX_CRITIC_ITERATIONS = 2

CRITIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "need_more_evidence": {
            "type": "boolean",
            "description": "True if an additional targeted query would materially change your decision.",
        },
        "evidence_request": {
            "type": ["string", "null"],
            "description": "If need_more_evidence, a natural-language description of the SQL query to run. Otherwise null.",
        },
        "flag_decisions": {
            "type": "array",
            "description": "One entry per raw flag under review, even if requesting more evidence (best-effort).",
            "items": {
                "type": "object",
                "properties": {
                    "check": {"type": "string", "description": "The 'check' field of the flag being decided."},
                    "decision": {"type": "string", "enum": ["confirm", "dismiss"]},
                    "adjusted_severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "rationale": {"type": "string"},
                },
                "required": ["check", "decision", "adjusted_severity", "rationale"],
                "additionalProperties": False,
            },
        },
        "overall_rationale": {
            "type": "string",
            "description": (
                "One or two sentences summarizing your overall assessment. "
                "The final approve/flag/reject decision is NOT yours to pick "
                "directly - it is computed mechanically from your "
                "flag_decisions (confirmed-high -> reject, any other "
                "confirmed -> flag, all dismissed -> approve). Your leverage "
                "over the outcome is entirely through 'decision' and "
                "'adjusted_severity' on each flag - make sure those reflect "
                "your true judgment."
            ),
        },
    },
    "required": [
        "need_more_evidence",
        "evidence_request",
        "flag_decisions",
        "overall_rationale",
    ],
    "additionalProperties": False,
}


def _build_prompt(
    invoice: dict,
    all_flags: list[Flag],
    vendor_history: list[dict],
    critic_evidence: list[str],
    remaining_iterations: int,
) -> str:
    lines = [
        "You are the Critic agent in an invoice-auditing pipeline. Two "
        "upstream agents raised the flags below using simple, mechanical "
        "rules with no broader context: the DB Investigator (source="
        "'db_investigator') flags any same-vendor+amount payment already "
        "made in the same calendar month as a possible duplicate, or any "
        "vendor not on the approved list. The Policy Assessor (source="
        "'policy_assessor') flags spend that exceeds a category approval "
        "threshold (e.g. software subscriptions over $500) using retrieval "
        "over policy docs, without seeing the vendor's payment history. Both "
        "over-flag: recurring payments (subscriptions, biweekly retainers) "
        "legitimately repeat the same amount, sometimes more than once a "
        "month, and a policy flag may already have approval evidence the "
        "Policy Assessor didn't fully weigh. Your job is to look at the "
        "vendor's full payment history and decide, per flag, whether it is a "
        "genuine risk (confirm) or a false positive (dismiss). A pattern "
        "spanning several months at a consistent interval and amount is "
        "strong evidence of a legitimate recurring payment; a same-amount "
        "repeat with no history outside the current month is not.",
        "",
        "Severity semantics (these drive the final decision mechanically, so "
        "set them deliberately): 'high' means block the payment outright - "
        "reserve it for a confirmed financial-loss risk (e.g. a genuine "
        "duplicate payment with no supporting pattern). 'medium' means send "
        "for human review, not an automatic block - use it for compliance/"
        "process issues like an unapproved vendor or a missing policy "
        "approval when there's no other red flag. 'low' is a minor, "
        "easily-explained note. If you dismiss a flag as a false positive, "
        "its severity no longer affects the outcome, but still set it to "
        "reflect how much residual concern (if any) remains.",
        "",
        f"Invoice under review: vendor={invoice['vendor']!r}, amount={invoice['amount']}, "
        f"date={invoice['invoice_date']}, description={invoice.get('description', '')!r}",
        "",
        "Raw flags from upstream agents:",
    ]
    if not all_flags:
        lines.append("  (none)")
    for f in all_flags:
        lines.append(f"  - source={f.source} check={f.check!r} severity={f.severity}: {f.message}")

    lines.append("")
    lines.append(f"Full payment history for this vendor ({len(vendor_history)} payments):")
    if not vendor_history:
        lines.append("  (no prior payments on record)")
    for row in vendor_history:
        lines.append(f"  - {row['payment_date']}: ${row['amount']:.2f}")

    if critic_evidence:
        lines.append("")
        lines.append("Additional evidence you previously requested:")
        for e in critic_evidence:
            lines.append(f"  - {e}")

    lines.append("")
    if remaining_iterations > 0:
        lines.append(
            f"You may request {remaining_iterations} more round(s) of evidence if it "
            "would materially change your decision. Otherwise finalize now."
        )
    else:
        lines.append("Your evidence budget is exhausted - finalize your decision now.")

    return "\n".join(lines)


def review(
    invoice: dict,
    db_findings: DBFindings,
    raw_flags: list[Flag],
    policy_flags: list[Flag],
    critic_evidence: list[str],
    critic_iterations: int,
    client: OpenAI | None = None,
    db_path: Path = DB_PATH,
) -> dict:
    """Runs one critic pass over the union of raw_flags (DB Investigator) and
    policy_flags (Policy Assessor). Returns a partial AuditState update:
      - {"critic_evidence": [...], "critic_iterations": N}  if looping for more evidence
      - {"reviewed_flags": [...], "final_verdict": Verdict(...)}  once finalized
    """
    client = client or OpenAI()
    all_flags = raw_flags + policy_flags
    conn = open_readonly_connection(db_path)
    try:
        vendor_history = get_vendor_history(conn, invoice["vendor"])
        remaining = MAX_CRITIC_ITERATIONS - critic_iterations
        prompt = _build_prompt(invoice, all_flags, vendor_history, critic_evidence, remaining)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "critic_review",
                    "strict": True,
                    "schema": CRITIC_RESPONSE_SCHEMA,
                },
            },
        )
        result = json.loads(response.choices[0].message.content)

        if result["need_more_evidence"] and remaining > 0:
            new_rows = run_freeform_query(client, conn, result["evidence_request"])
            evidence_summary = f"{result['evidence_request']} -> {new_rows}"
            return {
                "critic_evidence": critic_evidence + [evidence_summary],
                "critic_iterations": critic_iterations + 1,
            }

        by_check = {f.check: f for f in all_flags}
        reviewed: list[Flag] = []
        for decision in result["flag_decisions"]:
            original = by_check.get(decision["check"])
            if original is None:
                continue
            reviewed.append(
                original.model_copy(
                    update={
                        "severity": decision["adjusted_severity"],
                        "status": "confirmed" if decision["decision"] == "confirm" else "dismissed",
                        "rationale": decision["rationale"],
                    }
                )
            )

        return {
            "reviewed_flags": reviewed,
            "final_verdict": Verdict(
                decision=verdict_from_flags(reviewed),
                rationale=result["overall_rationale"],
            ),
        }
    finally:
        conn.close()
