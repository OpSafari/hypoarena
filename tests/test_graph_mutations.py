"""Replacement and removal keep links, edges and indexes consistent."""

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
from hypoarena.errors import UnknownReferenceError, ValidationError
from hypoarena.graph import HypothesisGraph
from hypoarena.schema import ClaimRelation, PredictedRelation


def populated() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_evidence(sample_evidence())
    graph.add_evidence(sample_evidence(evidence_id=OTHER_EVIDENCE_ID))
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    graph.link_evidence(OTHER_CLAIM_ID, EVIDENCE_ID)
    graph.link_evidence(CLAIM_ID, OTHER_EVIDENCE_ID)
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)
    return graph


def test_replace_claim_keeps_links_and_edges() -> None:
    graph = populated()
    revised = sample_claim(
        statement="Protein A increases gene B expression under hypoxia",
        relation=PredictedRelation.ENABLES,
    )
    graph.replace_claim(revised)
    assert graph.claim(CLAIM_ID).relation is PredictedRelation.ENABLES
    assert len(graph.evidence_for(CLAIM_ID)) == 2
    assert graph.edge_count == 1


def test_replace_claim_requires_an_existing_identifier() -> None:
    graph = populated()
    with pytest.raises(UnknownReferenceError):
        graph.replace_claim(sample_claim(claim_id="clm_999999999999"))
    with pytest.raises(ValidationError, match="Claim objects"):
        graph.replace_claim(CLAIM_ID)  # type: ignore[arg-type]
    assert graph.claim(CLAIM_ID).statement == sample_claim().statement


def test_remove_claim_cascades_edges_and_links_but_keeps_evidence() -> None:
    graph = populated()
    removed = graph.remove_claim(CLAIM_ID)
    assert removed.claim_id == CLAIM_ID
    assert graph.has_claim(CLAIM_ID) is False
    assert graph.edge_count == 0
    assert graph.link_count == 1
    assert graph.evidence_ids == (EVIDENCE_ID, OTHER_EVIDENCE_ID)
    assert graph.claims_for_evidence(EVIDENCE_ID) == (OTHER_CLAIM_ID,)


def test_remove_claim_reports_unknown_identifiers() -> None:
    with pytest.raises(UnknownReferenceError):
        populated().remove_claim("clm_999999999999")


def test_remove_evidence_drops_links_in_both_directions() -> None:
    graph = populated()
    graph.remove_evidence(EVIDENCE_ID)
    assert graph.has_evidence(EVIDENCE_ID) is False
    assert graph.link_count == 1
    assert [item.evidence_id for item in graph.evidence_for(CLAIM_ID)] == [
        OTHER_EVIDENCE_ID
    ]


def test_clear_empties_every_index() -> None:
    graph = populated()
    graph.clear()
    assert len(graph) == 0
    assert graph.evidence_ids == ()
    assert graph.edges == ()
    assert graph.link_count == 0
