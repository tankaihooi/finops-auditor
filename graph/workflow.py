"""
LangGraph definition.

Phase 3 pipeline: db_investigator -> flag_raiser -> critic (which can loop on
itself, gathering more evidence, up to critic.MAX_CRITIC_ITERATIONS) -> END.

The Policy Assessor node is added in Phase 4 per PLAN.md - do not scaffold it
here ahead of time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import END, StateGraph

from agents.critic import review as critic_review
from agents.db_investigator import investigate
from agents.flagging import raise_raw_flags
from graph.state import AuditState


def db_investigator_node(state: AuditState) -> dict:
    findings = investigate(state.raw_invoice)
    return {"db_findings": findings}


def flag_raiser_node(state: AuditState) -> dict:
    flags = raise_raw_flags(state.db_findings)
    return {"raw_flags": flags}


def critic_node(state: AuditState) -> dict:
    return critic_review(
        state.raw_invoice,
        state.db_findings,
        state.raw_flags,
        state.critic_evidence,
        state.critic_iterations,
    )


def route_after_critic(state: AuditState) -> str:
    return "done" if state.final_verdict is not None else "loop"


def build_graph():
    graph = StateGraph(AuditState)
    graph.add_node("db_investigator", db_investigator_node)
    graph.add_node("flag_raiser", flag_raiser_node)
    graph.add_node("critic", critic_node)
    graph.set_entry_point("db_investigator")
    graph.add_edge("db_investigator", "flag_raiser")
    graph.add_edge("flag_raiser", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"done": END, "loop": "critic"})
    return graph.compile()


if __name__ == "__main__":
    import json

    invoice_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if invoice_path is None:
        print("Usage: python graph/workflow.py <path-to-invoice.json>")
        sys.exit(1)

    invoice = json.loads(invoice_path.read_text())
    invoice.pop("expected", None)
    invoice.pop("expected_verdict", None)

    app = build_graph()
    result = app.invoke(AuditState(raw_invoice=invoice))

    print("Raw flags (pre-critic):")
    for f in result["raw_flags"]:
        print(f"  - [{f.severity}] {f.check}: {f.message}")

    print("\nReviewed flags (post-critic):")
    for f in result["reviewed_flags"]:
        print(f"  - [{f.status}] {f.check} (severity={f.severity}): {f.rationale}")

    print(f"\nFinal verdict: {result['final_verdict'].decision} - {result['final_verdict'].rationale}")
