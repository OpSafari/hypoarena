"""Golden JSONL payloads for the claim/evidence schema.

The two line literals below are the exact bytes the serializers must produce for
the reference records. They pin key names, nesting, null handling and the
canonical ordering promised by :func:`hypoarena.codec.dumps_line`, so artifacts
written by one version stay readable (and byte-comparable) by the next.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

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
    claim_from_dict,
    claim_from_line,
    claim_to_line,
    evidence_from_line,
    evidence_to_line,
)

SECRET_SHAPED = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)

GOLDEN_CLAIM_LINE = '{"citations":[{"document_id":"doc_0123456789ab","end":25,"quote":"kinase binds","start":10}],"claim_id":"clm_0123456789ab","mechanism":"phosphorylation of the promoter complex","object":"gene B expression","provenance":{"agent_id":"agt_0123456789ab","corpus_hash":"0123456789abcdef","generation":2,"notes":null,"origin":"agent","parents":["clm_ffffffffffff"],"seed":270106},"relation":"increases","schema_version":"1.0","scope":{"conditions":["hypoxia"],"population":"hek293 cells"},"statement":"Protein A increases gene B expression","subject":"protein A"}'

GOLDEN_EVIDENCE_LINE = '{"citations":[{"document_id":"doc_0123456789ab","end":25,"quote":"kinase binds","start":10}],"effect_size":1.25,"evidence_id":"evd_0123456789ab","method":"synthetic_finding","polarity":"support","provenance":{"agent_id":null,"corpus_hash":"0123456789abcdef","generation":0,"notes":null,"origin":"synthetic","parents":[],"seed":270106},"sample_size":48,"schema_version":"1.0","statement":"ChIP-seq shows binding enrichment at the promoter","strength":0.8}'


def reference_claim() -> Claim:
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


def reference_evidence() -> Evidence:
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


def keys_of(node: Any) -> Iterator[str]:
    """Yield every mapping key inside a nested JSON structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from keys_of(value)
    elif isinstance(node, list):
        for item in node:
            yield from keys_of(item)


def test_claim_line_matches_the_golden_bytes() -> None:
    assert claim_to_line(reference_claim()).rstrip("\n") == GOLDEN_CLAIM_LINE


def test_evidence_line_matches_the_golden_bytes() -> None:
    assert evidence_to_line(reference_evidence()).rstrip("\n") == GOLDEN_EVIDENCE_LINE


def test_golden_lines_decode_back_to_the_reference_records() -> None:
    assert claim_from_line(GOLDEN_CLAIM_LINE) == reference_claim()
    assert evidence_from_line(GOLDEN_EVIDENCE_LINE) == reference_evidence()


def test_golden_lines_are_canonically_ordered_and_idempotent() -> None:
    for line in (GOLDEN_CLAIM_LINE, GOLDEN_EVIDENCE_LINE):
        payload = json.loads(line)
        assert list(payload) == sorted(payload)
        assert dumps_line(payload).rstrip("\n") == line


def test_a_jsonl_document_roundtrips_line_by_line() -> None:
    document = GOLDEN_CLAIM_LINE + "\n" + GOLDEN_EVIDENCE_LINE + "\n"
    lines = document.splitlines()
    assert len(lines) == 2
    assert claim_from_line(lines[0], line_number=1) == reference_claim()
    assert evidence_from_line(lines[1], line_number=2) == reference_evidence()


def test_payloads_carry_no_secret_shaped_keys() -> None:
    for line in (GOLDEN_CLAIM_LINE, GOLDEN_EVIDENCE_LINE):
        for key in keys_of(json.loads(line)):
            lowered = key.lower()
            assert not any(marker in lowered for marker in SECRET_SHAPED), key


def test_tampered_golden_payloads_are_rejected() -> None:
    payload = json.loads(GOLDEN_CLAIM_LINE)
    payload["claim_id"] = "clm_not_valid"
    with pytest.raises(ValidationError, match="malformed"):
        claim_from_dict(payload)
    payload = json.loads(GOLDEN_CLAIM_LINE)
    del payload["statement"]
    with pytest.raises(SchemaError, match="missing"):
        claim_from_dict(payload)
    payload = json.loads(GOLDEN_EVIDENCE_LINE)
    payload["strength"] = 4.0
    with pytest.raises(SchemaError, match="above the maximum"):
        evidence_from_line(dumps_line(payload))
