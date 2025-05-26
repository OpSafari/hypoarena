"""Serialization of value objects: golden dicts, roundtrips and rejections."""

from __future__ import annotations

import json

import pytest

from hypoarena.codec import dumps_line
from hypoarena.errors import SchemaError, ValidationError
from hypoarena.schema import (
    Citation,
    Claim,
    Evidence,
    EvidencePolarity,
    PredictedRelation,
    Provenance,
    Scope,
)
from hypoarena.serialize import (
    citation_from_dict,
    citation_to_dict,
    claim_from_dict,
    claim_from_line,
    claim_to_dict,
    claim_to_line,
    evidence_from_dict,
    evidence_from_line,
    evidence_to_dict,
    evidence_to_line,
    provenance_from_dict,
    provenance_to_dict,
    scope_from_dict,
    scope_to_dict,
)


def test_scope_roundtrips_and_pins_its_dict_shape() -> None:
    scope = Scope("hek293 cells", ("hypoxia",))
    assert scope_to_dict(scope) == {
        "population": "hek293 cells",
        "conditions": ["hypoxia"],
    }
    assert scope_from_dict(scope_to_dict(scope)) == scope


def test_scope_from_dict_rejects_unknown_keys_and_bad_payloads() -> None:
    with pytest.raises(SchemaError, match="unknown keys"):
        scope_from_dict({"population": "cells", "extra": 1})
    with pytest.raises(SchemaError, match="missing"):
        scope_from_dict({"conditions": []})
    with pytest.raises(ValidationError):
        scope_from_dict({"population": " ", "conditions": []})


def test_citation_roundtrips_and_pins_its_dict_shape() -> None:
    citation = Citation("doc_0123456789ab", 10, 25, "kinase binds")
    assert citation_to_dict(citation) == {
        "document_id": "doc_0123456789ab",
        "start": 10,
        "end": 25,
        "quote": "kinase binds",
    }
    assert citation_from_dict(citation_to_dict(citation)) == citation


def test_citation_from_dict_enforces_offset_bounds() -> None:
    payload = citation_to_dict(Citation("doc_0123456789ab", 10, 25, "kinase binds"))
    payload["start"] = -1
    with pytest.raises(SchemaError, match="below the minimum"):
        citation_from_dict(payload)
    payload["start"] = 30
    payload["end"] = 25
    with pytest.raises(ValidationError, match="non-empty"):
        citation_from_dict(payload)


def test_provenance_roundtrips_with_explicit_nulls() -> None:
    provenance = Provenance(origin="synthetic", seed=7, corpus_hash="0123456789abcdef")
    encoded = provenance_to_dict(provenance)
    assert encoded == {
        "origin": "synthetic",
        "agent_id": None,
        "generation": 0,
        "parents": [],
        "seed": 7,
        "corpus_hash": "0123456789abcdef",
        "notes": None,
    }
    assert provenance_from_dict(encoded) == provenance


def test_provenance_from_dict_defers_cross_field_rules_to_the_constructor() -> None:
    payload = provenance_to_dict(Provenance(origin="user"))
    payload["origin"] = "agent"
    with pytest.raises(ValidationError, match="agent_id"):
        provenance_from_dict(payload)
    payload["origin"] = "oracle"
    with pytest.raises(ValidationError, match="unknown provenance origin"):
        provenance_from_dict(payload)


def golden_claim() -> Claim:
    return Claim(
        claim_id="clm_0123456789ab",
        statement="Protein A increases gene B expression",
        subject="protein A",
        object="gene B expression",
        relation=PredictedRelation.INCREASES,
        scope=Scope("hek293 cells", ("hypoxia",)),
        citations=(Citation("doc_0123456789ab", 10, 25, "kinase binds"),),
        mechanism="phosphorylation of the promoter complex",
        provenance=Provenance(
            origin="agent",
            agent_id="agt_0123456789ab",
            generation=2,
            parents=("clm_ffffffffffff",),
            seed=270106,
            corpus_hash="0123456789abcdef",
        ),
    )


def test_claim_to_dict_is_golden() -> None:
    assert claim_to_dict(golden_claim()) == {
        "schema_version": "1.0",
        "claim_id": "clm_0123456789ab",
        "statement": "Protein A increases gene B expression",
        "subject": "protein A",
        "object": "gene B expression",
        "relation": "increases",
        "scope": {"population": "hek293 cells", "conditions": ["hypoxia"]},
        "citations": [
            {
                "document_id": "doc_0123456789ab",
                "start": 10,
                "end": 25,
                "quote": "kinase binds",
            }
        ],
        "mechanism": "phosphorylation of the promoter complex",
        "provenance": {
            "origin": "agent",
            "agent_id": "agt_0123456789ab",
            "generation": 2,
            "parents": ["clm_ffffffffffff"],
            "seed": 270106,
            "corpus_hash": "0123456789abcdef",
            "notes": None,
        },
    }


def test_claim_roundtrips_through_dict_and_line() -> None:
    claim = golden_claim()
    assert claim_from_dict(claim_to_dict(claim)) == claim
    assert claim_from_line(claim_to_line(claim)) == claim


def test_claim_from_dict_rejects_unknown_keys_and_bad_versions() -> None:
    payload = claim_to_dict(golden_claim())
    payload["reviewer_notes"] = "surprise"
    with pytest.raises(SchemaError, match="unknown keys"):
        claim_from_dict(payload)
    payload = claim_to_dict(golden_claim())
    payload["schema_version"] = "9.9"
    with pytest.raises(SchemaError, match="unsupported schema version"):
        claim_from_dict(payload)


def test_claim_from_dict_rejects_unknown_relations() -> None:
    payload = claim_to_dict(golden_claim())
    payload["relation"] = "teleports"
    with pytest.raises(SchemaError, match="not a valid PredictedRelation"):
        claim_from_dict(payload)


def test_claim_from_dict_propagates_semantic_validation() -> None:
    payload = claim_to_dict(golden_claim())
    payload["object"] = "Protein A"
    with pytest.raises(ValidationError, match="must differ"):
        claim_from_dict(payload)


def test_claim_line_is_canonical_and_idempotent() -> None:
    line = claim_to_line(golden_claim())
    assert line.endswith("\n")
    assert "\n" not in line[:-1]
    assert dumps_line(json.loads(line)) == line


def golden_evidence() -> Evidence:
    return Evidence(
        evidence_id="evd_0123456789ab",
        statement="ChIP-seq shows binding enrichment at the promoter",
        polarity=EvidencePolarity.SUPPORT,
        strength=0.8,
        citations=(Citation("doc_0123456789ab", 10, 25, "kinase binds"),),
        method="synthetic_finding",
        provenance=Provenance(
            origin="synthetic", seed=270106, corpus_hash="0123456789abcdef"
        ),
        effect_size=1.25,
        sample_size=48,
    )


def test_evidence_to_dict_is_golden() -> None:
    assert evidence_to_dict(golden_evidence()) == {
        "schema_version": "1.0",
        "evidence_id": "evd_0123456789ab",
        "statement": "ChIP-seq shows binding enrichment at the promoter",
        "polarity": "support",
        "strength": 0.8,
        "citations": [
            {
                "document_id": "doc_0123456789ab",
                "start": 10,
                "end": 25,
                "quote": "kinase binds",
            }
        ],
        "method": "synthetic_finding",
        "provenance": {
            "origin": "synthetic",
            "agent_id": None,
            "generation": 0,
            "parents": [],
            "seed": 270106,
            "corpus_hash": "0123456789abcdef",
            "notes": None,
        },
        "effect_size": 1.25,
        "sample_size": 48,
    }


def test_evidence_roundtrips_through_dict_and_line() -> None:
    evidence = golden_evidence()
    assert evidence_from_dict(evidence_to_dict(evidence)) == evidence
    assert evidence_from_line(evidence_to_line(evidence)) == evidence


def test_evidence_from_dict_enforces_strength_bounds() -> None:
    payload = evidence_to_dict(golden_evidence())
    payload["strength"] = 1.5
    with pytest.raises(SchemaError, match="above the maximum"):
        evidence_from_dict(payload)


def test_evidence_from_dict_rejects_unknown_polarity_and_keys() -> None:
    payload = evidence_to_dict(golden_evidence())
    payload["polarity"] = "maybe"
    with pytest.raises(SchemaError, match="not a valid EvidencePolarity"):
        evidence_from_dict(payload)
    payload = evidence_to_dict(golden_evidence())
    payload["reviewer"] = "anon"
    with pytest.raises(SchemaError, match="unknown keys"):
        evidence_from_dict(payload)


def test_evidence_line_is_canonical() -> None:
    line = evidence_to_line(golden_evidence())
    assert dumps_line(json.loads(line)) == line
