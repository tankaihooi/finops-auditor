"""
Headline metric: false-positive rate before vs. after the Critic agent.

Runs every labeled invoice in data/invoices/ through the full pipeline
(db_investigator -> flag_raiser -> policy_assessor -> critic) and compares two
verdicts against each invoice's "expected_verdict":

  - raw verdict:    the naive, mechanical rollup of every flag raised before
                     critic review (agents.flagging.verdict_from_flags applied
                     to raw_flags + policy_flags) - no judgment, no context.
  - critic verdict:  the Critic agent's final_verdict after reviewing those
                     flags against the vendor's payment history.

A "false positive" is an invoice whose expected_verdict is "approve" but the
system raised an alarm ("flag" or "reject") anyway. This is the FinOps
equivalent of the false-positive fraud flag CLAUDE.md's Evaluation section
asks for: report the false-positive rate before vs. after the critic.

Requires OPENAI_API_KEY in the environment.

Run with: python eval/run_benchmark.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.flagging import verdict_from_flags
from graph.state import AuditState
from graph.workflow import build_graph

INVOICES_DIR = Path(__file__).resolve().parent.parent / "data" / "invoices"


def _load_invoices() -> list[dict]:
    invoices = []
    for path in sorted(INVOICES_DIR.glob("*.json")):
        invoice = json.loads(path.read_text())
        invoice["_path"] = path.name
        invoices.append(invoice)
    return invoices


def run() -> bool:
    invoices = _load_invoices()
    if not invoices:
        print(f"No test invoices found in {INVOICES_DIR}")
        return False

    app = build_graph()

    rows = []
    for invoice in invoices:
        expected_verdict = invoice["expected_verdict"]
        clean_invoice = {
            k: v
            for k, v in invoice.items()
            if k not in ("_path", "expected", "expected_verdict", "note")
        }

        result = app.invoke(AuditState(raw_invoice=clean_invoice))

        raw_verdict = verdict_from_flags(result["raw_flags"] + result["policy_flags"])
        critic_verdict = result["final_verdict"].decision

        rows.append(
            {
                "path": invoice["_path"],
                "vendor": invoice["vendor"],
                "expected_verdict": expected_verdict,
                "raw_verdict": raw_verdict,
                "critic_verdict": critic_verdict,
                "raw_flags": result["raw_flags"],
                "reviewed_flags": result["reviewed_flags"],
                "critic_rationale": result["final_verdict"].rationale,
            }
        )

    print(f"{'invoice':40s} {'expected':10s} {'raw':10s} {'critic':10s}")
    print("-" * 75)
    for r in rows:
        marker_raw = " " if r["raw_verdict"] == r["expected_verdict"] else "*"
        marker_critic = " " if r["critic_verdict"] == r["expected_verdict"] else "*"
        print(
            f"{r['path']:40s} {r['expected_verdict']:10s} "
            f"{r['raw_verdict']:9s}{marker_raw} {r['critic_verdict']:9s}{marker_critic}"
        )
    print("(* = mismatch vs. expected_verdict)\n")

    for r in rows:
        if r["reviewed_flags"]:
            print(f"{r['path']}:")
            for f in r["reviewed_flags"]:
                print(f"    [{f.status}] {f.check} (severity={f.severity}): {f.rationale}")
    print()

    clean_invoices = [r for r in rows if r["expected_verdict"] == "approve"]

    def false_positive_rate(verdict_key: str) -> float:
        if not clean_invoices:
            return 0.0
        false_positives = sum(1 for r in clean_invoices if r[verdict_key] != "approve")
        return false_positives / len(clean_invoices)

    def accuracy(verdict_key: str) -> float:
        correct = sum(1 for r in rows if r[verdict_key] == r["expected_verdict"])
        return correct / len(rows)

    raw_fp_rate = false_positive_rate("raw_verdict")
    critic_fp_rate = false_positive_rate("critic_verdict")
    raw_accuracy = accuracy("raw_verdict")
    critic_accuracy = accuracy("critic_verdict")

    print("=" * 75)
    print(f"Overall verdict accuracy: raw={raw_accuracy:.0%}  critic={critic_accuracy:.0%}  "
          f"({len(rows)} invoices)")
    print(
        f"False-positive rate on clean invoices ({len(clean_invoices)} labeled 'approve'): "
        f"raw={raw_fp_rate:.0%}  critic={critic_fp_rate:.0%}"
    )
    if raw_fp_rate > critic_fp_rate:
        print(
            f"-> The critic reduced false-positive flags from {raw_fp_rate:.0%} to "
            f"{critic_fp_rate:.0%}."
        )
    elif critic_fp_rate > raw_fp_rate:
        print(
            f"-> WARNING: the critic INCREASED the false-positive rate from "
            f"{raw_fp_rate:.0%} to {critic_fp_rate:.0%}."
        )
    else:
        print(f"-> No change in false-positive rate ({raw_fp_rate:.0%}).")
    print("=" * 75)

    return critic_accuracy >= raw_accuracy and critic_fp_rate <= raw_fp_rate


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
