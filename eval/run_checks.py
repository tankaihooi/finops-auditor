"""
Phase 1 ground-truth verification: runs the plain-SQL checks against every
labeled test invoice in data/invoices/ and prints pass/fail per check.

Each invoice JSON carries an "expected" block with the known-correct answer
for both checks. This script does NOT judge whether an invoice should be
approved - it only verifies the SQL checks themselves return the right
booleans. That's the ground truth the later agentic layers get graded against.

Run with: python eval/run_checks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.checks import get_connection, is_vendor_approved, is_duplicate_amount

INVOICES_DIR = Path(__file__).resolve().parent.parent / "data" / "invoices"


def run() -> bool:
    conn = get_connection()
    invoice_files = sorted(INVOICES_DIR.glob("*.json"))

    if not invoice_files:
        print(f"No test invoices found in {INVOICES_DIR}")
        return False

    all_passed = True
    total_checks = 0
    passed_checks = 0

    for path in invoice_files:
        invoice = json.loads(path.read_text())
        expected = invoice["expected"]

        actual_vendor_approved = is_vendor_approved(conn, invoice["vendor"])
        actual_is_duplicate = is_duplicate_amount(
            conn, invoice["vendor"], invoice["amount"], invoice["invoice_date"]
        )

        results = [
            ("vendor_approved", expected["vendor_approved"], actual_vendor_approved),
            ("is_duplicate", expected["is_duplicate"], actual_is_duplicate),
        ]

        print(f"\n{path.name} — {invoice['vendor']} / ${invoice['amount']:.2f} / {invoice['invoice_date']}")
        for check_name, expected_val, actual_val in results:
            total_checks += 1
            ok = expected_val == actual_val
            passed_checks += ok
            all_passed &= ok
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {check_name}: expected={expected_val} actual={actual_val}")

    conn.close()

    print(f"\n{'=' * 50}")
    print(f"{passed_checks}/{total_checks} checks passed across {len(invoice_files)} invoices")
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
