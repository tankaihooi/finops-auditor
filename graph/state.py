"""
Shared state object every agent reads from and writes to.

As of Phase 3: `raw_invoice`, `db_findings`, `raw_flags`, `reviewed_flags`,
`critic_evidence`, `critic_iterations`, and `final_verdict` are populated.
`extracted` and `policy_flags` are placeholders for later phases per PLAN.md.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class InvoiceSchema(BaseModel):
    invoice_id: str
    vendor: str
    amount: float
    invoice_date: str
    description: str | None = None


class SQLCheckResult(BaseModel):
    check_name: str
    result: bool
    generated_sql: str
    attempts: int
    reasoning: str


class DBFindings(BaseModel):
    vendor_approved: SQLCheckResult
    is_duplicate: SQLCheckResult


class Flag(BaseModel):
    source: str
    check: str
    severity: Literal["low", "medium", "high"]
    message: str
    status: Literal["open", "confirmed", "dismissed"] = "open"
    rationale: str | None = None


class Verdict(BaseModel):
    decision: Literal["approve", "flag", "reject"]
    rationale: str


class AuditState(BaseModel):
    raw_invoice: dict[str, Any]
    extracted: InvoiceSchema | None = None
    db_findings: DBFindings | None = None
    raw_flags: list[Flag] = Field(default_factory=list)
    reviewed_flags: list[Flag] = Field(default_factory=list)
    critic_evidence: list[str] = Field(default_factory=list)
    policy_flags: list[Flag] = Field(default_factory=list)
    final_verdict: Verdict | None = None
    critic_iterations: int = 0
