"""Evidence storage and claim–evidence links."""

from __future__ import annotations

import pytest

from helpers import (
    CLAIM_ID,
    EVIDENCE_ID,
    OTHER_CLAIM_ID,
    OTHER_EVIDENCE_ID,
    sample_claim,
    sample_evidence,
)
from hypoarena.errors import (
    DuplicateIdError,
    GraphInvariantError,
    UnknownReferenceError,
    ValidationError,
)
from hypoarena.graph import HypothesisGraph


def graph_with_claim() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    return graph


def test_add_evidence_stores_and_orders_by_identifier() -> None:
    graph = graph_with_claim()
    graph.add_evidence(sample_evidence())
    graph.add_evidence(sample_evidence(evidence_id=OTHER_EVIDENCE_ID))
    assert graph.evidence_ids == (EVIDENCE_ID, OTHER_EVIDENCE_ID)
    assert len(graph.evidence_items) == 2
    assert graph.evidence(EVIDENCE_ID).strength == 0.8


def test_duplicate_evidence_and_wrong_types_are_rejected() -> None:
    graph = graph_with_claim()
    graph.add_evidence(sample_evidence())
    with pytest.raises(DuplicateIdError) as info:
        graph.add_evidence(sample_evidence(statement="a different finding"))
    assert info.value.kind == "evidence"
    with pytest.raises(ValidationError, match="Evidence objects"):
        graph.add_evidence("evd_0123456789ab")  # type: ignore[arg-type]


def test_link_requires_both_endpoints_to_exist() -> None:
    graph = graph_with_claim()
    graph.add_evidence(sample_evidence())
    with pytest.raises(UnknownReferenceError):
        graph.link_evidence(CLAIM_ID, OTHER_EVIDENCE_ID)
    with pytest.raises(UnknownReferenceError):
        graph.link_evidence(OTHER_CLAIM_ID, EVIDENCE_ID)
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    assert graph.link_count == 1


def test_linking_twice_is_rejected() -> None:
    graph = graph_with_claim()
    graph.add_evidence(sample_evidence())
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    with pytest.raises(GraphInvariantError, match="already linked"):
        graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    assert graph.link_count == 1


def test_evidence_lookup_is_bidirectional() -> None:
    graph = graph_with_claim()
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_evidence(sample_evidence())
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    graph.link_evidence(OTHER_CLAIM_ID, EVIDENCE_ID)
    assert [item.evidence_id for item in graph.evidence_for(CLAIM_ID)] == [EVIDENCE_ID]
    assert graph.claims_for_evidence(EVIDENCE_ID) == (CLAIM_ID, OTHER_CLAIM_ID)


def test_unlink_removes_both_directions_and_reports_missing_links() -> None:
    graph = graph_with_claim()
    graph.add_evidence(sample_evidence())
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    graph.unlink_evidence(CLAIM_ID, EVIDENCE_ID)
    assert graph.link_count == 0
    assert graph.evidence_for(CLAIM_ID) == ()
    with pytest.raises(UnknownReferenceError):
        graph.unlink_evidence(CLAIM_ID, EVIDENCE_ID)


def test_evidence_lookup_reports_unknown_identifiers() -> None:
    with pytest.raises(UnknownReferenceError) as info:
        graph_with_claim().evidence_for("clm_111111111111")
    assert info.value.kind == "claim"
    with pytest.raises(UnknownReferenceError):
        graph_with_claim().claims_for_evidence(EVIDENCE_ID)
