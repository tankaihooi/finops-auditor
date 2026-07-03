"""
DB Investigator agent (Phase 2): text-to-SQL wrapper around the Phase 1 checks.

Given an invoice and a natural-language check description, the LLM generates a
SQLite SELECT query, we execute it (read-only connection) and verify the
result, and retry with the error fed back on: (a) non-SELECT SQL, (b) a SQL
execution error, or (c) an unexpected empty result. This is the
execute -> verify -> iterate loop described in PLAN.md's Phase 2.

Requires OPENAI_API_KEY in the environment.
"""

import json
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from graph.state import DBFindings, SQLCheckResult

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = "gpt-4.1"
MAX_RETRIES = 3

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_erp.db"

SCHEMA_DESCRIPTION = """
Tables:
  vendors(vendor_id INTEGER PRIMARY KEY, vendor_name TEXT)
    -- the approved vendor list. A vendor is "approved" iff its name appears here.
  paid_invoices(invoice_id INTEGER PRIMARY KEY, vendor_name TEXT, amount REAL, payment_date TEXT)
    -- historical payments, payment_date is 'YYYY-MM-DD'. vendor_name here is NOT
    -- guaranteed to be in the vendors table (unapproved vendors can still have paid invoices).
""".strip()

SQL_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": (
                "A single read-only SQLite SELECT statement. Use named "
                "placeholders (e.g. :vendor, :amount, :invoice_date) for any "
                "invoice values instead of literals. The query must always "
                "return exactly one row with exactly one column named "
                "'result', containing 1 (true) or 0 (false)."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence explaining the query's logic.",
        },
    },
    "required": ["sql", "reasoning"],
    "additionalProperties": False,
}

FREEFORM_SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": (
                "A single read-only SQLite SELECT statement answering the "
                "request. May return any number of rows/columns - no fixed "
                "shape required, unlike a boolean check query."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence explaining the query's logic.",
        },
    },
    "required": ["sql", "reasoning"],
    "additionalProperties": False,
}

_SELECT_ONLY = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


class SQLGenerationError(RuntimeError):
    """Raised when the LLM fails to produce a working query within MAX_RETRIES."""


def open_readonly_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Read-only connection: even if a model ever proposes a write despite
    instructions, execution fails immediately rather than mutating the DB."""
    db_uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(db_uri, uri=True)


def get_vendor_history(conn: sqlite3.Connection, vendor: str) -> list[dict]:
    """Deterministic (no LLM) lookup: every payment on record for this vendor,
    oldest first. Used to give the Critic agent cadence/pattern evidence."""
    rows = conn.execute(
        "SELECT amount, payment_date FROM paid_invoices "
        "WHERE LOWER(vendor_name) = LOWER(?) ORDER BY payment_date",
        (vendor,),
    ).fetchall()
    return [{"amount": amount, "payment_date": payment_date} for amount, payment_date in rows]


def run_freeform_query(
    client: OpenAI, conn: sqlite3.Connection, description: str, max_retries: int = MAX_RETRIES
) -> list[dict]:
    """Text-to-SQL for an arbitrary read-only data request (not constrained to
    a single boolean result). Used by the Critic agent when it needs evidence
    beyond the standard vendor-history lookup."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a SQL analyst for an invoice-auditing system. Given "
                "a database schema and a data request, write a single "
                "SQLite SELECT query that answers it. Only SELECT statements "
                f"are allowed - no writes.\n\nSchema:\n{SCHEMA_DESCRIPTION}"
            ),
        },
        {"role": "user", "content": description},
    ]

    last_error = "no attempts made"
    for _ in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "freeform_sql",
                    "strict": True,
                    "schema": FREEFORM_SQL_SCHEMA,
                },
            },
        )
        proposal = json.loads(response.choices[0].message.content)
        sql = proposal["sql"]
        messages.append({"role": "assistant", "content": json.dumps(proposal)})

        if not _SELECT_ONLY.match(sql):
            last_error = "Only SELECT statements are allowed."
            messages.append(
                {"role": "user", "content": f"Rejected: {last_error} Rewrite the query."}
            )
            continue

        try:
            cur = conn.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        except sqlite3.Error as e:
            last_error = str(e)
            messages.append(
                {"role": "user", "content": f"SQL error: {last_error} Fix the query."}
            )
            continue

        return [dict(zip(columns, row)) for row in rows]

    raise SQLGenerationError(f"freeform query failed after {max_retries} attempts. Last error: {last_error}")


def _propose_sql(client: OpenAI, messages: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "sql_proposal",
                "strict": True,
                "schema": SQL_PROPOSAL_SCHEMA,
            },
        },
    )
    return json.loads(response.choices[0].message.content)


def _run_check(
    client: OpenAI,
    conn: sqlite3.Connection,
    check_name: str,
    check_description: str,
    params: dict,
) -> SQLCheckResult:
    system_prompt = (
        "You are a SQL analyst for an invoice-auditing system. Given a "
        "database schema and a check description, write a single SQLite "
        "SELECT query that answers the check. Only SELECT statements are "
        "allowed - no writes.\n\n"
        f"Schema:\n{SCHEMA_DESCRIPTION}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Check: {check_description}\n\n"
                f"Available parameters (bind exactly these names): {list(params.keys())}"
            ),
        },
    ]

    last_error = "no attempts made"
    for attempt in range(1, MAX_RETRIES + 1):
        proposal = _propose_sql(client, messages)
        sql = proposal["sql"]
        reasoning = proposal["reasoning"]
        messages.append({"role": "assistant", "content": json.dumps(proposal)})

        if not _SELECT_ONLY.match(sql):
            last_error = "Only SELECT statements are allowed."
            messages.append(
                {"role": "user", "content": f"Rejected: {last_error} Rewrite the query."}
            )
            continue

        try:
            cur = conn.execute(sql, params)
            row = cur.fetchone()
        except sqlite3.Error as e:
            last_error = str(e)
            messages.append(
                {"role": "user", "content": f"SQL error: {last_error} Fix the query."}
            )
            continue

        if row is None:
            last_error = "Query returned zero rows."
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{last_error} The query must always return exactly one "
                        "row (e.g. use SELECT EXISTS(...) AS result or "
                        "SELECT COUNT(*) > 0 AS result). Rewrite it."
                    ),
                }
            )
            continue

        return SQLCheckResult(
            check_name=check_name,
            result=bool(row[0]),
            generated_sql=sql,
            attempts=attempt,
            reasoning=reasoning,
        )

    raise SQLGenerationError(
        f"'{check_name}' check failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def investigate(
    invoice: dict, db_path: Path = DB_PATH, client: OpenAI | None = None
) -> DBFindings:
    """Runs both DB checks for an invoice via the text-to-SQL agent loop."""
    client = client or OpenAI()
    conn = open_readonly_connection(db_path)
    try:
        params = {
            "vendor": invoice["vendor"],
            "amount": invoice["amount"],
            "invoice_date": invoice["invoice_date"],
        }

        vendor_check = _run_check(
            client,
            conn,
            "vendor_approved",
            "Is the vendor named :vendor present in the vendors table (case-insensitive exact match)?",
            {"vendor": params["vendor"]},
        )

        duplicate_check = _run_check(
            client,
            conn,
            "is_duplicate",
            (
                "Has the same vendor (:vendor, case-insensitive) already been "
                "paid the same amount (:amount, within 0.001) in the same "
                "calendar month as :invoice_date, according to paid_invoices?"
            ),
            params,
        )

        return DBFindings(vendor_approved=vendor_check, is_duplicate=duplicate_check)
    finally:
        conn.close()
