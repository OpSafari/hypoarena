"""Shared builders for schema fixtures.

Every builder returns a valid record by default and accepts keyword overrides,
so a test can break exactly one invariant at a time instead of restating a whole
payload. Fixtures are synthetic by construction: identifiers are literals and no
builder touches the network or the filesystem.
"""

from __future__ import annotations

from typing import Any

from hypoarena.schema import (
    Citation,
    Claim,
    PredictedRelation,
    Provenance,
    Scope,
)

DOCUMENT_ID = "doc_0123456789ab"
OTHER_DOCUMENT_ID = "doc_ffffffffffff"
CLAIM_ID = "clm_0123456789ab"
OTHER_CLAIM_ID = "clm_ffffffffffff"
AGENT_ID = "agt_0123456789ab"
CORPUS_HASH = "0123456789abcdef"


def sample_scope(**overrides: Any) -> Scope:
    """Return a scope for a cultured-cell population."""
    payload: dict[str, Any] = {"population": "hek293 cells", "conditions": ()}
    payload.update(overrides)
    return Scope(**payload)


def sample_citation(**overrides: Any) -> Citation:
    """Return a citation pointing into the sample synthetic document."""
    payload: dict[str, Any] = {
        "document_id": DOCUMENT_ID,
        "start": 10,
        "end": 25,
        "quote": "kinase binds",
    }
    payload.update(overrides)
    return Citation(**payload)


def sample_provenance(**overrides: Any) -> Provenance:
    """Return agent provenance with a recorded seed and corpus hash."""
    payload: dict[str, Any] = {
        "origin": "agent",
        "agent_id": AGENT_ID,
        "generation": 1,
        "parents": (),
        "seed": 270106,
        "corpus_hash": CORPUS_HASH,
        "notes": None,
    }
    payload.update(overrides)
    return Provenance(**payload)


def sample_claim(**overrides: Any) -> Claim:
    """Return a cited claim about a protein increasing gene expression."""
    payload: dict[str, Any] = {
        "claim_id": CLAIM_ID,
        "statement": "Protein A increases the expression of gene B",
        "subject": "protein A",
        "object": "gene B expression",
        "relation": PredictedRelation.INCREASES,
        "scope": sample_scope(),
        "citations": (sample_citation(),),
        "mechanism": "phosphorylation of the promoter complex",
        "provenance": sample_provenance(),
    }
    payload.update(overrides)
    return Claim(**payload)
