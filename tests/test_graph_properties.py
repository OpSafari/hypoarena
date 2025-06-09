"""Seeded structural sweeps over randomly shaped graphs.

The generator uses an explicit ``random.Random(seed)`` so every case is
reproducible. The properties checked are the ones the rest of the toolkit relies
on: every mutation path leaves the graph valid, serialization roundtrips are
byte-identical, and derived views stay sorted and duplicate-free.
"""

from __future__ import annotations

import random

import pytest

from helpers import sample_claim, sample_evidence
from hypoarena.graph import HypothesisGraph
from hypoarena.ids import make_id
from hypoarena.schema import ClaimRelation, EvidencePolarity, PredictedRelation
from hypoarena.serialize import graph_from_text, graph_to_text

RELATIONS = list(ClaimRelation)
DIRECTION_COUNT = 4


def random_graph(seed: int, claims: int = 12) -> HypothesisGraph:
    """Build a randomly shaped but valid graph from ``seed``."""
    rng = random.Random(seed)
    graph = HypothesisGraph()
    claim_ids = [make_id("clm", seed, index) for index in range(claims)]
    evidence_ids = [make_id("evd", seed, index) for index in range(claims // 2 + 1)]
    for index, claim_id in enumerate(claim_ids):
        graph.add_claim(
            sample_claim(
                claim_id=claim_id,
                statement=f"variable {index} affects variable {index + 1}",
                subject=f"variable {index}",
                object=f"variable {index + 1}",
                relation=rng.choice(list(PredictedRelation)),
            )
        )
    for index, evidence_id in enumerate(evidence_ids):
        graph.add_evidence(
            sample_evidence(
                evidence_id=evidence_id,
                statement=f"observation {index} reports an effect",
                polarity=EvidencePolarity.SUPPORT
                if index % 3
                else EvidencePolarity.REFUTE,
                strength=0.0 if index % 3 == 0 else round(rng.random(), 3),
            )
        )
    for _ in range(claims):
        claim_id = rng.choice(claim_ids)
        evidence_id = rng.choice(evidence_ids)
        if evidence_id not in [
            item.evidence_id for item in graph.evidence_for(claim_id)
        ]:
            graph.link_evidence(claim_id, evidence_id)
    for _ in range(claims):
        source = rng.choice(claim_ids)
        target = rng.choice(claim_ids)
        relation = rng.choice(RELATIONS)
        if source == target or graph.has_edge(source, target, relation):
            continue
        if relation is ClaimRelation.CONTRADICTS and graph.has_edge(
            target, source, relation
        ):
            continue
        graph.add_edge(source, target, relation)
    graph.validate()
    return graph


@pytest.mark.parametrize("seed", [1, 7, 270106])
def test_random_graphs_are_valid_and_sorted(seed: int) -> None:
    graph = random_graph(seed)
    assert graph.validate() is None
    assert list(graph.claim_ids) == sorted(graph.claim_ids)
    assert len(set(graph.claim_ids)) == len(graph)
    keys = [edge.key() for edge in graph.edges]
    assert keys == sorted(keys) and len(keys) == len(set(keys))


@pytest.mark.parametrize("seed", [1, 7, 270106])
def test_serialization_roundtrip_is_byte_identical(seed: int) -> None:
    graph = random_graph(seed)
    text = graph_to_text(graph)
    assert graph_to_text(graph_from_text(text)) == text
    assert graph_from_text(text).signature() == graph.signature()


@pytest.mark.parametrize("seed", [1, 7])
def test_removals_and_subgraphs_stay_valid(seed: int) -> None:
    graph = random_graph(seed)
    claim_ids = list(graph.claim_ids)
    for claim_id in claim_ids[:3]:
        graph.remove_claim(claim_id)
        graph.validate()
    remaining = graph.claim_ids
    sliced = graph.subgraph(remaining[: max(1, len(remaining) // 2)])
    sliced.validate()
    assert len(sliced) <= len(graph)
    assert graph.merge(sliced) == 0


@pytest.mark.parametrize("seed", [1, 7])
def test_merging_disjoint_graphs_adds_everything(seed: int) -> None:
    first = random_graph(seed, claims=6)
    second = random_graph(seed + 1000, claims=6)
    total = len(first) + len(second)
    edges = first.edge_count + second.edge_count
    combined = first.merged(second)
    combined.validate()
    assert len(combined) == total
    assert combined.edge_count == edges
    assert combined.signature() == second.merged(first).signature()
