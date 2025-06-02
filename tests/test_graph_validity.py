"""The validity sweep accepts sound graphs and names every corruption.

The corruption cases poke private indexes directly on purpose: the sweep exists
to catch exactly that class of bug, so the tests have to be able to produce it.
"""

from __future__ import annotations

import pytest

from helpers import (
    CLAIM_ID,
    EVIDENCE_ID,
    OTHER_CLAIM_ID,
    sample_claim,
    sample_evidence,
)
from hypoarena.errors import GraphInvariantError
from hypoarena.graph import ClaimEdge, HypothesisGraph
from hypoarena.schema import ClaimRelation


def sound_graph() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_evidence(sample_evidence())
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)
    return graph


def test_a_sound_graph_passes_validation() -> None:
    graph = sound_graph()
    assert graph.validate() is None


def test_an_empty_graph_passes_validation() -> None:
    assert HypothesisGraph().validate() is None


def test_misindexed_claims_are_detected() -> None:
    graph = sound_graph()
    graph._claims["clm_999999999999"] = graph._claims.pop(CLAIM_ID)
    with pytest.raises(GraphInvariantError, match="claim index disagrees"):
        graph.validate()


def test_dangling_links_are_detected() -> None:
    graph = sound_graph()
    del graph._evidence[EVIDENCE_ID]
    with pytest.raises(GraphInvariantError, match="missing evidence"):
        graph.validate()


def test_asymmetric_link_indexes_are_detected() -> None:
    graph = sound_graph()
    graph._backlinks[EVIDENCE_ID].remove(CLAIM_ID)
    with pytest.raises(GraphInvariantError, match="backlink index is missing"):
        graph.validate()
    graph = sound_graph()
    graph._links[CLAIM_ID].append("evd_999999999999")
    with pytest.raises(GraphInvariantError, match="missing evidence"):
        graph.validate()


def test_duplicate_links_are_detected() -> None:
    graph = sound_graph()
    graph._links[CLAIM_ID].append(EVIDENCE_ID)
    with pytest.raises(GraphInvariantError, match="duplicate evidence links"):
        graph.validate()


def test_edges_pointing_at_removed_claims_are_detected() -> None:
    graph = sound_graph()
    del graph._claims[OTHER_CLAIM_ID]
    with pytest.raises(GraphInvariantError, match="missing claim"):
        graph.validate()


def test_duplicate_edges_and_broken_edge_indexes_are_detected() -> None:
    graph = sound_graph()
    duplicate = ClaimEdge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)
    graph._edges[CLAIM_ID].append(duplicate)
    with pytest.raises(GraphInvariantError, match="duplicate edge"):
        graph.validate()
    graph = sound_graph()
    graph._incoming[OTHER_CLAIM_ID].clear()
    with pytest.raises(GraphInvariantError, match="incoming index is missing"):
        graph.validate()


def test_misfiled_edges_are_detected() -> None:
    graph = sound_graph()
    edge = graph._edges[CLAIM_ID].pop()
    graph._edges.setdefault(OTHER_CLAIM_ID, []).append(edge)
    with pytest.raises(GraphInvariantError, match="wrong source"):
        graph.validate()
