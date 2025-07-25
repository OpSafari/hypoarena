"""Grounding report serialization: roundtrips, integrity and rejections."""

from __future__ import annotations

import json

import pytest

from hypoarena.corpus import Corpus, Document
from hypoarena.errors import SchemaError
from hypoarena.grounding import GroundingFlag, GroundingVerifier
from hypoarena.schema import Citation, Claim, PredictedRelation, Provenance, Scope
from hypoarena.serialize import (
    citation_check_from_dict,
    citation_check_to_dict,
    grounding_counts,
    grounding_report_from_dict,
    grounding_report_to_dict,
    grounding_reports_from_lines,
    grounding_reports_to_lines,
    grounding_signature,
)

DOC = "doc_0123456789ab"
TEXT = "Protein A increases cell growth in HeLa cells. Batch effects were reviewed."
QUOTE = "Protein A increases cell growth"
START = TEXT.index(QUOTE)


def claim(citations: tuple[Citation, ...], claim_id: str = "clm_0123456789ab") -> Claim:
    return Claim(
        claim_id=claim_id,
        statement="Protein A increases cell growth",
        subject="protein A",
        object="cell growth",
        relation=PredictedRelation.INCREASES,
        scope=Scope(population="HeLa cells"),
        citations=citations,
        provenance=Provenance(origin="synthetic", seed=1),
    )


def reports() -> tuple[object, ...]:
    verifier = GroundingVerifier(Corpus([Document(DOC, "Findings", TEXT)]))
    good = Citation(DOC, START, START + len(QUOTE), QUOTE)
    forged = Citation(DOC, START, START + len(QUOTE), "Protein A decreases growth")
    return verifier.verify_claims(
        [
            claim((good,), "clm_0123456789ab"),
            claim((forged,), "clm_ffffffffffff"),
            claim((), "clm_222222222222"),
        ]
    )


def test_citation_checks_roundtrip() -> None:
    check = reports()[0].checks[0]
    assert citation_check_from_dict(citation_check_to_dict(check)) == check


def test_reports_roundtrip_with_their_flags() -> None:
    produced = reports()
    for report in produced:
        restored = grounding_report_from_dict(grounding_report_to_dict(report))
        assert restored == report
    assert [item.flag for item in produced] == [
        GroundingFlag.GROUNDED,
        GroundingFlag.FABRICATED,
        GroundingFlag.UNGROUNDED,
    ]


def test_report_dict_keys_are_golden() -> None:
    payload = grounding_report_to_dict(reports()[0])
    assert sorted(payload) == [
        "checks",
        "claim_id",
        "flag",
        "issues",
        "schema_version",
        "score",
        "statement",
    ]
    assert sorted(payload["checks"][0]) == [
        "claimed_numbers",
        "detail",
        "document_id",
        "end",
        "entity_overlap",
        "issues",
        "quote",
        "quoted_numbers",
        "resolved",
        "start",
    ]


def test_lines_roundtrip_and_pin_the_record_order() -> None:
    produced = reports()
    lines = grounding_reports_to_lines(produced)
    kinds = [json.loads(line)["record"] for line in lines]
    assert kinds == ["meta", "report", "report", "report"]
    assert grounding_reports_from_lines(lines) == produced


def test_header_counts_and_signature_are_verified() -> None:
    produced = reports()
    assert grounding_counts(produced) == {
        "reports": 3,
        "grounded": 1,
        "weakly_grounded": 0,
        "ungrounded": 1,
        "fabricated": 1,
    }
    lines = grounding_reports_to_lines(produced)
    meta = json.loads(lines[0])
    assert meta["signature"] == grounding_signature(produced)
    with pytest.raises(SchemaError, match="counts disagree"):
        grounding_reports_from_lines(lines[:-1])


def test_unknown_records_and_keys_are_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown grounding record type"):
        grounding_reports_from_lines(['{"record":"verdict","flag":"grounded"}'])
    payload = grounding_report_to_dict(reports()[0])
    payload["reviewer"] = "anon"
    with pytest.raises(SchemaError, match="unknown keys"):
        grounding_report_from_dict(payload)


def test_tampered_flags_are_rejected() -> None:
    payload = grounding_report_to_dict(reports()[0])
    payload["flag"] = "perfectly_grounded"
    with pytest.raises(SchemaError, match="not a valid GroundingFlag"):
        grounding_report_from_dict(payload)
    payload = grounding_report_to_dict(reports()[0])
    payload["score"] = 4.0
    with pytest.raises(SchemaError, match="above the maximum"):
        grounding_report_from_dict(payload)
