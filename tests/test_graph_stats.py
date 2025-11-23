"""Link enumeration, statistics and the content signature."""

from __future__ import annotations

from helpers import (
    CLAIM_ID,
    EVIDENCE_ID,
    OTHER_CLAIM_ID,
    OTHER_EVIDENCE_ID,
    sample_claim,
    sample_evidence,
)
from hypoarena.graph import GraphStats, HypothesisGraph
from hypoarena.schema import ClaimRelation, PredictedRelation


def populated() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_claim(sample_claim(claim_id="clm_222222222222"))
    graph.add_evidence(sample_evidence())
    graph.add_evidence(sample_evidence(evidence_id=OTHER_EVIDENCE_ID))
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    graph.link_evidence(CLAIM_ID, OTHER_EVIDENCE_ID)
    graph.link_evidence(OTHER_CLAIM_ID, EVIDENCE_ID)
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.CONTRADICTS)
    return graph


def test_link_pairs_are_sorted_and_complete() -> None:
    assert populated().link_pairs() == (
        (CLAIM_ID, EVIDENCE_ID),
        (CLAIM_ID, OTHER_EVIDENCE_ID),
        (OTHER_CLAIM_ID, EVIDENCE_ID),
    )


def test_stats_summarize_shape() -> None:
    stats = populated().stats()
    assert isinstance(stats, GraphStats)
    assert (stats.claims, stats.evidence, stats.links, stats.edges) == (3, 2, 3, 2)
    assert stats.edges_by_relation == (("contradicts", 1), ("entails", 1))
    assert stats.isolated_claims == 1
    # both the entailment source and the unconnected claim have no incoming edge
    assert stats.roots == 2
    assert stats.leaves == 2


def test_stats_of_an_empty_graph_are_all_zero() -> None:
    stats = HypothesisGraph().stats()
    assert stats.as_dict() == {
        "claims": 0,
        "evidence": 0,
        "links": 0,
        "edges": 0,
        "edges_by_relation": {},
        "isolated_claims": 0,
        "roots": 0,
        "leaves": 0,
    }


def test_signature_is_stable_across_insertion_order() -> None:
    first = populated()
    second = HypothesisGraph()
    second.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    second.add_claim(sample_claim())
    second.add_claim(sample_claim(claim_id="clm_222222222222"))
    second.add_evidence(sample_evidence(evidence_id=OTHER_EVIDENCE_ID))
    second.add_evidence(sample_evidence())
    second.link_evidence(OTHER_CLAIM_ID, EVIDENCE_ID)
    second.link_evidence(CLAIM_ID, OTHER_EVIDENCE_ID)
    second.link_evidence(CLAIM_ID, EVIDENCE_ID)
    second.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.CONTRADICTS)
    second.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)
    assert first.signature() == second.signature()


def test_signature_tracks_content_changes() -> None:
    graph = populated()
    baseline = graph.signature()
    graph.replace_claim(sample_claim(relation=PredictedRelation.DECREASES))
    assert graph.signature() != baseline
    graph.remove_claim("clm_222222222222")
    assert graph.signature() != baseline


def test_signature_of_an_empty_graph_is_a_valid_digest() -> None:
    assert len(HypothesisGraph().signature()) == 16
