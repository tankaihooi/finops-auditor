"""
Phase 2 verification: runs the text-to-SQL DB Investigator agent against every
labeled test invoice in data/invoices/ and compares its answers to the same
ground-truth "expected" block used in eval/run_checks.py (Phase 1).

This is the first real test of the execute -> verify -> iterate loop: does an
LLM writing its own SQL land on the same answers as the hand-written SQL from
Phase 1? Also reports how many retry attempts each check needed.

Requires OPENAI_API_KEY in the environment.

Run with: python eval/run_investigator.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.db_investigator import SQLGenerationError, investigate

INVOICES_DIR = Path(__file__).resolve().parent.parent / "data" / "invoices"


def run() -> bool:
    invoice_files = sorted(INVOICES_DIR.glob("*.json"))

    if not invoice_files:
        print(f"No test invoices found in {INVOICES_DIR}")
        return False

    all_passed = True
    total_checks = 0
    passed_checks = 0

    for path in invoice_files:
        invoice = json.loads(path.read_text())
        expected = invoice.pop("expected")

        print(f"\n{path.name} — {invoice['vendor']} / ${invoice['amount']:.2f} / {invoice['invoice_date']}")

        try:
            findings = investigate(invoice)
        except SQLGenerationError as e:
            print(f"  [FAIL] agent could not complete checks: {e}")
            all_passed = False
            total_checks += 2
            continue

        results = [
            ("vendor_approved", expected["vendor_approved"], findings.vendor_approved),
            ("is_duplicate", expected["is_duplicate"], findings.is_duplicate),
        ]

        for check_name, expected_val, check_result in results:
            total_checks += 1
            ok = expected_val == check_result.result
            passed_checks += ok
            all_passed &= ok
            status = "PASS" if ok else "FAIL"
            print(
                f"  [{status}] {check_name}: expected={expected_val} actual={check_result.result} "
                f"(attempts={check_result.attempts})"
            )
            print(f"         sql: {check_result.generated_sql}")

    print(f"\n{'=' * 50}")
    print(f"{passed_checks}/{total_checks} checks passed across {len(invoice_files)} invoices")
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
