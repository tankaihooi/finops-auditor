"""
Policy Assessor agent (Phase 4): RAG over data/policies/*.md.

Given an invoice, retrieves the most relevant policy chunks from the ChromaDB
index (built by scripts/build_policy_index.py) and asks the LLM whether the
invoice violates any of them (e.g. "software subscriptions over $500 require
VP approval"). The LLM is given the invoice's own description as evidence, so
an invoice that already states its approval (ticket number, named approver)
is not flagged - the same false-positive-avoidance principle as the Critic's
review of duplicate flags, just applied at the point flags are raised instead
of after.

Requires OPENAI_API_KEY in the environment.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from graph.state import Flag

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 3

INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "policy_index"
COLLECTION_NAME = "finops_policies"

# Docs that impose an actual category+threshold approval requirement, i.e.
# things that can genuinely be "violated". "recurring_payments" deliberately
# excluded - it's an exemption clause (no new approval needed each cycle),
# not a rule with its own threshold, so it can never itself be a violation.
# Retrieval over a small, generic corpus (per CLAUDE.md's "corpus mismatch"
# warning) will sometimes surface a doc that doesn't really apply to a given
# invoice - constraining the schema to this list, plus the prompt's explicit
# category-match instruction below, is what keeps that noise from turning
# into spurious flags.
VIOLATABLE_POLICIES = [
    "software_subscriptions",
    "professional_services",
    "equipment_purchases",
    "marketing_spend",
    "travel_and_expenses",
]

ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "violations": {
            "type": "array",
            "description": "Zero or more genuine policy violations found. Do not include a policy chunk here if the invoice's own description already shows evidence of the required approval.",
            "items": {
                "type": "object",
                "properties": {
                    "policy": {
                        "type": "string",
                        "enum": VIOLATABLE_POLICIES,
                        "description": "Which category policy is violated.",
                    },
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "message": {"type": "string", "description": "One sentence explaining the violation."},
                },
                "required": ["policy", "severity", "message"],
                "additionalProperties": False,
            },
        },
        "reasoning": {
            "type": "string",
            "description": "One or two sentences on why these policies do or don't apply here.",
        },
    },
    "required": ["violations", "reasoning"],
    "additionalProperties": False,
}


def _get_collection(index_path: Path = INDEX_PATH):
    client = PersistentClient(path=str(index_path))
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name=EMBEDDING_MODEL
    )
    return client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)


def assess(
    invoice: dict, client: OpenAI | None = None, index_path: Path = INDEX_PATH, top_k: int = TOP_K
) -> list[Flag]:
    """Retrieves the most relevant policy docs for this invoice and returns a
    Flag for each genuine violation the LLM finds. Severity semantics match
    agents/critic.py: 'high' = block payment, 'medium' = needs review,
    'low' = minor note."""
    client = client or OpenAI()
    collection = _get_collection(index_path)

    query_text = (
        f"vendor={invoice['vendor']} amount={invoice['amount']} "
        f"description={invoice.get('description', '')}"
    )
    retrieved = collection.query(query_texts=[query_text], n_results=top_k)
    policy_docs = list(zip(retrieved["ids"][0], retrieved["documents"][0]))

    prompt_lines = [
        "You are the Policy Assessor agent in an invoice-auditing pipeline. "
        "The documents below were retrieved by embedding similarity against a "
        "small, generic policy corpus - retrieval is approximate, and for an "
        "invoice that doesn't clearly belong to any of these spend categories "
        "(software subscription, professional services/consulting, equipment/"
        "hardware, marketing/advertising, travel/entertainment), the "
        "'closest' retrieved doc may still not actually apply. Only raise a "
        "violation if the invoice's own DESCRIPTION - what is actually being "
        "purchased or billed - genuinely places it in one of those "
        "categories AND it exceeds that category's dollar threshold. Do not "
        "infer a category from the vendor's name alone (a vendor named "
        "'... Software' or '... Consulting' doesn't mean every invoice from "
        "them is a software subscription or a consulting engagement - it "
        "could just as easily be office supplies or an unrelated expense; "
        "look for actual subscription/renewal/license, SOW/engagement, "
        "equipment, campaign, or travel language in the description itself). "
        "If the invoice doesn't clearly match any category (e.g. general "
        "office supplies, logistics, an unrelated business expense with no "
        "category signal in its description), return zero violations - do "
        "not force a match onto a retrieved doc just because it was "
        "returned.",
        "",
        "The 'recurring_payments' document is an exemption, not a rule - it "
        "can never itself be the violated policy; use it only to decide "
        "whether NOT to raise a software_subscriptions/professional_services/"
        "etc. violation you'd otherwise raise.",
        "",
        "Severity: 'high' means block payment outright, 'medium' means send "
        "for the required approval/sign-off before paying (the normal case "
        "for a missing approval), 'low' is a minor note. If the invoice's "
        "own description already shows evidence of the required approval (a "
        "ticket number, a named approver, a reference to an existing "
        "approved budget/SOW), do NOT flag it - that policy's requirement "
        "has already been satisfied.",
        "",
        f"Invoice: vendor={invoice['vendor']!r}, amount={invoice['amount']}, "
        f"date={invoice['invoice_date']}, description={invoice.get('description', '')!r}",
        "",
        "Retrieved policy documents:",
    ]
    for policy_id, doc_text in policy_docs:
        prompt_lines.append(f"--- {policy_id} ---")
        prompt_lines.append(doc_text)
        prompt_lines.append("")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "\n".join(prompt_lines)}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "policy_assessment",
                "strict": True,
                "schema": ASSESSMENT_SCHEMA,
            },
        },
    )
    result = json.loads(response.choices[0].message.content)

    return [
        Flag(
            source="policy_assessor",
            check=v["policy"],
            severity=v["severity"],
            message=v["message"],
        )
        for v in result["violations"]
    ]
