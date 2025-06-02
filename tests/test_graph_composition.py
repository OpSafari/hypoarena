"""Subgraphs, merging and the conflict rules that guard both."""

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
from hypoarena.errors import GraphInvariantError, UnknownReferenceError
from hypoarena.graph import HypothesisGraph
from hypoarena.schema import ClaimRelation


def left() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_evidence(sample_evidence())
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)
    return graph


def right() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_claim(sample_claim(claim_id="clm_222222222222"))
    graph.add_evidence(sample_evidence(evidence_id=OTHER_EVIDENCE_ID))
    graph.link_evidence(OTHER_CLAIM_ID, OTHER_EVIDENCE_ID)
    graph.add_edge(OTHER_CLAIM_ID, "clm_222222222222", ClaimRelation.REFINES)
    return graph


def test_subgraph_keeps_selected_claims_and_inner_edges() -> None:
    graph = left()
    sliced = graph.subgraph([CLAIM_ID, OTHER_CLAIM_ID])
    assert sliced.claim_ids == graph.claim_ids
    assert sliced.edge_count == 1
    assert sliced.link_count == 1


def test_subgraph_drops_edges_leaving_the_selection() -> None:
    sliced = left().subgraph([CLAIM_ID])
    assert sliced.claim_ids == (CLAIM_ID,)
    assert sliced.edges == ()
    assert sliced.link_count == 1


def test_subgraph_can_skip_evidence() -> None:
    sliced = left().subgraph([CLAIM_ID], include_evidence=False)
    assert sliced.evidence_ids == ()
    assert sliced.link_count == 0


def test_subgraph_rejects_unknown_claims() -> None:
    with pytest.raises(UnknownReferenceError):
        left().subgraph(["clm_999999999999"])


def test_merge_counts_new_records_and_skips_identical_ones() -> None:
    graph = left()
    added = graph.merge(right())
    assert added == 4  # one claim, one evidence item, one link, one edge
    assert graph.claim_ids == (CLAIM_ID, "clm_222222222222", OTHER_CLAIM_ID)
    assert graph.evidence_ids == (EVIDENCE_ID, OTHER_EVIDENCE_ID)
    assert graph.edge_count == 2
    assert graph.link_count == 2
    assert graph.merge(right()) == 0


def test_merge_rejects_conflicting_records() -> None:
    graph = left()
    conflicting = HypothesisGraph()
    conflicting.add_claim(sample_claim(statement="a different statement"))
    with pytest.raises(GraphInvariantError, match="conflicting claim"):
        graph.merge(conflicting)
    conflicting = HypothesisGraph()
    conflicting.add_evidence(sample_evidence(strength=0.2))
    with pytest.raises(GraphInvariantError, match="conflicting evidence"):
        graph.merge(conflicting)


def test_merged_does_not_mutate_either_input() -> None:
    first, second = left(), right()
    combined = first.merged(second)
    assert len(first) == 2 and len(second) == 2
    assert len(combined) == 3
    assert combined.link_count == 2
    assert combined.validate() is None
