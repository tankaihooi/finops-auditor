"""
Phase 1 ground-truth checks: plain SQL, no LLM.

Two checks against data/mock_erp.db:
  1. is_vendor_approved       - is the invoice's vendor on the approved list?
  2. is_duplicate_amount      - has this exact amount already been paid to
                                this vendor in the same calendar month as the
                                invoice date?

Duplicate scope is (vendor, amount, month) rather than (amount, month) alone -
two different vendors legitimately billing the same round amount in the same
month is not a duplicate-invoice signal, so vendor is part of the match.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_erp.db"

# Amounts are compared with a small tolerance to avoid floating-point
# equality issues (e.g. 219.99 stored/retrieved as 219.98999999999998).
AMOUNT_TOLERANCE = 0.001


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def is_vendor_approved(conn: sqlite3.Connection, vendor_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM vendors WHERE LOWER(vendor_name) = LOWER(?)",
        (vendor_name,),
    ).fetchone()
    return row is not None


def is_duplicate_amount(
    conn: sqlite3.Connection, vendor_name: str, amount: float, invoice_date: str
) -> bool:
    """invoice_date must be 'YYYY-MM-DD'; matches paid_invoices in the same
    vendor + calendar month with an amount within AMOUNT_TOLERANCE."""
    row = conn.execute(
        """
        SELECT 1 FROM paid_invoices
        WHERE LOWER(vendor_name) = LOWER(?)
          AND ABS(amount - ?) < ?
          AND strftime('%Y-%m', payment_date) = strftime('%Y-%m', ?)
        """,
        (vendor_name, amount, AMOUNT_TOLERANCE, invoice_date),
    ).fetchone()
    return row is not None
