"""
Deterministic (no LLM) raw flag generator: turns DB Investigator findings into
naive, aggressive flags. This is the "before" baseline the Critic agent
(agents/critic.py) is measured against - by design it flags any unapproved
vendor or any same-vendor+amount+month match, with no judgment about whether
that's actually a problem (e.g. a legitimate recurring payment).
"""

from graph.state import DBFindings, Flag


def raise_raw_flags(db_findings: DBFindings) -> list[Flag]:
    flags: list[Flag] = []

    if not db_findings.vendor_approved.result:
        # Compliance/process issue by default (needs review) - escalate to
        # "high" only if other evidence (e.g. the Critic finds a fraud
        # signal) warrants blocking the payment outright.
        flags.append(
            Flag(
                source="db_investigator",
                check="vendor_approved",
                severity="medium",
                message="Vendor is not on the approved vendor list.",
            )
        )

    if db_findings.is_duplicate.result:
        # Financial-loss risk by default (block until resolved) - the Critic
        # may downgrade this if vendor history shows a legitimate recurring
        # pattern rather than a genuine double payment.
        flags.append(
            Flag(
                source="db_investigator",
                check="is_duplicate",
                severity="high",
                message="Same vendor and amount already paid this calendar month - possible duplicate payment.",
            )
        )

    return flags


def verdict_from_flags(flags: list[Flag]) -> str:
    """Naive verdict rollup: any non-dismissed high-severity flag -> reject,
    any other non-dismissed flag -> flag, otherwise approve. Used both as the
    pre-critic baseline and as the mechanical part of the critic's own rollup."""
    live = [f for f in flags if f.status != "dismissed"]
    if any(f.severity == "high" for f in live):
        return "reject"
    if live:
        return "flag"
    return "approve"
