"""
Builds and seeds data/mock_erp.db from scratch.

Run with: python scripts/seed_db.py

Creates two tables:
  - vendors: the approved vendor list
  - paid_invoices: historical payments (vendor may or may not be approved,
    mirrors reality where off-list payments sometimes slip through)

Includes deliberate duplicate-amount cases (same vendor, same amount, same
calendar month) and payments to non-approved vendors so Phase 1's checks have
real positives and negatives to catch.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_erp.db"

APPROVED_VENDORS = [
    "Acme Office Supplies",
    "Staples Business Solutions",
    "CloudNova Hosting",
    "Meridian Consulting Group",
    "BrightPath Legal Services",
    "Apex Logistics",
    "Sterling Catering Co",
    "TechForge Software",
    "Horizon Marketing Partners",
    "Quantum Analytics Inc",
]

# (vendor, amount, payment_date YYYY-MM-DD)
# Includes:
#  - normal, unique payments across several months
#  - two intentional same-month duplicate-amount pairs with NO established
#    multi-month cadence (Acme, Apex) - genuine duplicate-payment risk, no
#    supporting pattern to explain them away
#  - a 6-month biweekly $2,400 retainer to Meridian Consulting Group - a
#    legitimate recurring payment that a naive same-vendor/amount/month check
#    will misflag as a "duplicate" once a second July payment arrives. This is
#    the false-positive case the Phase 3 critic agent is measured against.
#  - payments to vendors NOT in APPROVED_VENDORS (Rogue Consulting LLC,
#    Discount IT Hardware) to give the vendor check real negatives
_MERIDIAN_RETAINER_DATES = [
    "2026-01-02", "2026-01-16", "2026-01-30",
    "2026-02-13", "2026-02-27",
    "2026-03-13", "2026-03-27",
    "2026-04-10", "2026-04-24",
    "2026-05-08", "2026-05-22",
    "2026-06-05", "2026-06-19",
    "2026-07-03",  # last one before the test invoice (2026-07-17) arrives
]

PAID_INVOICES = [
    ("Acme Office Supplies", 482.10, "2026-05-04"),
    ("Acme Office Supplies", 219.99, "2026-06-02"),
    ("Acme Office Supplies", 219.99, "2026-06-18"),  # duplicate amount, same month, no cadence
    ("Staples Business Solutions", 1050.00, "2026-05-11"),
    ("Staples Business Solutions", 875.25, "2026-06-20"),
    ("CloudNova Hosting", 3200.00, "2026-05-01"),
    ("CloudNova Hosting", 3200.00, "2026-06-01"),  # same amount, different month -> NOT a dup
    ("CloudNova Hosting", 3400.00, "2026-07-01"),
    ("Meridian Consulting Group", 12500.00, "2026-04-15"),   # one-off project payment
    ("Meridian Consulting Group", 8600.00, "2026-06-15"),    # one-off project payment
    *[("Meridian Consulting Group", 2400.00, d) for d in _MERIDIAN_RETAINER_DATES],
    ("BrightPath Legal Services", 4750.00, "2026-05-22"),
    ("Apex Logistics", 615.40, "2026-06-09"),
    ("Apex Logistics", 615.40, "2026-06-27"),  # duplicate amount, same month, no cadence
    ("Sterling Catering Co", 340.00, "2026-06-03"),
    ("TechForge Software", 999.00, "2026-07-02"),
    ("Horizon Marketing Partners", 5200.00, "2026-05-30"),
    ("Quantum Analytics Inc", 2100.00, "2026-06-25"),
    ("Rogue Consulting LLC", 3000.00, "2026-06-10"),      # not an approved vendor
    ("Discount IT Hardware", 1499.99, "2026-06-14"),      # not an approved vendor
]


def build_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE vendors (
            vendor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT NOT NULL UNIQUE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE paid_invoices (
            invoice_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name  TEXT NOT NULL,
            amount       REAL NOT NULL,
            payment_date TEXT NOT NULL
        )
        """
    )

    cur.executemany(
        "INSERT INTO vendors (vendor_name) VALUES (?)",
        [(v,) for v in APPROVED_VENDORS],
    )
    cur.executemany(
        "INSERT INTO paid_invoices (vendor_name, amount, payment_date) VALUES (?, ?, ?)",
        PAID_INVOICES,
    )

    conn.commit()
    conn.close()
    print(f"Seeded {db_path} with {len(APPROVED_VENDORS)} vendors and {len(PAID_INVOICES)} paid invoices.")


if __name__ == "__main__":
    build_db()
