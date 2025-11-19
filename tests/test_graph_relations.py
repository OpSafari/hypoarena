"""Typed claim relations: insertion, lookup, ordering and rejection."""

from __future__ import annotations

import pytest

from helpers import CLAIM_ID, OTHER_CLAIM_ID, sample_claim
from hypoarena.errors import GraphInvariantError, UnknownReferenceError
from hypoarena.graph import ClaimEdge, HypothesisGraph
from hypoarena.schema import ClaimRelation

THIRD_CLAIM_ID = "clm_222222222222"


def chain() -> tuple[HypothesisGraph, ClaimEdge]:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_claim(sample_claim(claim_id=THIRD_CLAIM_ID))
    edge = graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)
    return graph, edge


def test_add_edge_returns_the_record_and_indexes_it() -> None:
    graph, edge = chain()
    assert edge.source == CLAIM_ID
    assert graph.has_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS) is True
    assert graph.edge_between(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS) == edge
    assert graph.edge_count == 1


def test_edges_require_existing_claims() -> None:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    with pytest.raises(UnknownReferenceError):
        graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)


def test_duplicate_and_symmetric_contradictions_are_rejected() -> None:
    graph, _ = chain()
    with pytest.raises(GraphInvariantError, match="already exists"):
        graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.CONTRADICTS)
    with pytest.raises(GraphInvariantError, match="symmetric"):
        graph.add_edge(OTHER_CLAIM_ID, CLAIM_ID, ClaimRelation.CONTRADICTS)
    assert graph.edge_count == 2


def test_different_relations_between_the_same_pair_coexist() -> None:
    graph, _ = chain()
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.REFINES)
    assert graph.edge_count == 2
    assert {edge.relation for edge in graph.edges} == {
        ClaimRelation.ENTAILS,
        ClaimRelation.REFINES,
    }


def test_edges_are_returned_in_canonical_order() -> None:
    graph, _ = chain()
    graph.add_edge(THIRD_CLAIM_ID, CLAIM_ID, ClaimRelation.REFINES)
    graph.add_edge(CLAIM_ID, THIRD_CLAIM_ID, ClaimRelation.ENTAILS, note="derived")
    keys = [edge.key() for edge in graph.edges]
    assert keys == sorted(keys)
    assert len(keys) == 3


def test_remove_edge_updates_both_indexes() -> None:
    graph, edge = chain()
    graph.remove_edge(edge)
    assert graph.edge_count == 0
    assert graph.edges == ()
    with pytest.raises(UnknownReferenceError):
        graph.remove_edge(edge)


def test_edge_between_reports_missing_relations() -> None:
    graph, _ = chain()
    with pytest.raises(UnknownReferenceError) as info:
        graph.edge_between(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.CONTRADICTS)
    assert info.value.kind == "edge"
