"""Traversal helpers: filtered views, reachability and root/leaf sets."""

from __future__ import annotations

import pytest

from helpers import CLAIM_ID, OTHER_CLAIM_ID, sample_claim
from hypoarena.errors import UnknownReferenceError
from hypoarena.graph import HypothesisGraph
from hypoarena.schema import ClaimRelation

MID_ID = "clm_222222222222"
LEAF_ID = "clm_333333333333"


def entailment_chain() -> HypothesisGraph:
    graph = HypothesisGraph()
    for claim_id in (CLAIM_ID, MID_ID, LEAF_ID, OTHER_CLAIM_ID):
        graph.add_claim(sample_claim(claim_id=claim_id))
    graph.add_edge(CLAIM_ID, MID_ID, ClaimRelation.ENTAILS)
    graph.add_edge(MID_ID, LEAF_ID, ClaimRelation.ENTAILS)
    graph.add_edge(OTHER_CLAIM_ID, MID_ID, ClaimRelation.CONTRADICTS)
    return graph


def test_outgoing_and_incoming_filter_by_relation() -> None:
    graph = entailment_chain()
    assert [edge.target for edge in graph.outgoing(CLAIM_ID)] == [MID_ID]
    assert [edge.source for edge in graph.incoming(MID_ID)] == [
        CLAIM_ID,
        OTHER_CLAIM_ID,
    ]
    assert graph.incoming(MID_ID, ClaimRelation.CONTRADICTS)[0].source == OTHER_CLAIM_ID
    assert graph.outgoing(MID_ID, ClaimRelation.CONTRADICTS) == ()


def test_traversal_requires_known_claims() -> None:
    graph = entailment_chain()
    with pytest.raises(UnknownReferenceError):
        graph.outgoing("clm_999999999999")
    with pytest.raises(UnknownReferenceError):
        graph.incoming("clm_999999999999")


def test_targets_sources_and_neighbors_are_sorted_and_unique() -> None:
    graph = entailment_chain()
    assert graph.targets(CLAIM_ID) == (MID_ID,)
    assert graph.sources(MID_ID) == tuple(sorted([CLAIM_ID, OTHER_CLAIM_ID]))
    assert graph.neighbors(MID_ID) == tuple(sorted({CLAIM_ID, OTHER_CLAIM_ID, LEAF_ID}))


def test_has_path_follows_direction_only() -> None:
    graph = entailment_chain()
    assert graph.has_path(CLAIM_ID, LEAF_ID) is True
    assert graph.has_path(LEAF_ID, CLAIM_ID) is False
    assert graph.has_path(CLAIM_ID, CLAIM_ID) is False


def test_has_path_respects_the_relation_filter() -> None:
    graph = entailment_chain()
    assert graph.has_path(OTHER_CLAIM_ID, LEAF_ID) is True
    assert (
        graph.has_path(CLAIM_ID, LEAF_ID, relations=frozenset({ClaimRelation.ENTAILS}))
        is True
    )
    assert (
        graph.has_path(
            OTHER_CLAIM_ID, LEAF_ID, relations=frozenset({ClaimRelation.ENTAILS})
        )
        is False
    )


def test_roots_and_leaves_partition_the_chain_ends() -> None:
    graph = entailment_chain()
    assert graph.roots() == tuple(sorted([CLAIM_ID, OTHER_CLAIM_ID]))
    assert graph.leaves() == (LEAF_ID,)
