"""Bundle graph wiring stays correct when several chains share vocabulary."""

from __future__ import annotations

import pytest

from hypoarena.schema import ClaimRelation, EvidencePolarity
from hypoarena.synthetic import SyntheticConfig, build_bundle


@pytest.mark.parametrize("chains", [1, 2, 4, 6])
def test_graph_is_valid_for_every_chain_count(chains: int) -> None:
    bundle = build_bundle(SyntheticConfig(seed=101, chains=chains, chain_length=3))
    graph = bundle.graph()
    assert graph.validate() is None
    assert len(graph) == len(bundle.claims)
    assert len(graph.evidence_items) == len(bundle.evidence)


@pytest.mark.parametrize("chains", [1, 2, 4, 6])
def test_every_evidence_item_is_linked_to_its_claim(chains: int) -> None:
    bundle = build_bundle(SyntheticConfig(seed=101, chains=chains, chain_length=3))
    graph = bundle.graph()
    assert graph.link_count == len(bundle.evidence)
    for item in bundle.evidence:
        assert len(graph.claims_for_evidence(item.evidence_id)) == 1


def test_rival_edges_match_the_planted_rival_count() -> None:
    bundle = build_bundle(SyntheticConfig(seed=101, chains=5, chain_length=3))
    graph = bundle.graph()
    edges = graph.edges
    assert len(edges) == len(bundle.truth.competing)
    assert all(edge.relation is ClaimRelation.CONTRADICTS for edge in edges)
    assert all(edge.note == "planted rival" for edge in edges)


def test_refuting_evidence_lands_on_planted_claims() -> None:
    bundle = build_bundle(SyntheticConfig(seed=101, chains=3, chain_length=3))
    graph = bundle.graph()
    refuting = [
        item for item in bundle.evidence if item.polarity is EvidencePolarity.REFUTE
    ]
    assert refuting
    for item in refuting:
        owner = graph.claims_for_evidence(item.evidence_id)[0]
        assert graph.claim(owner).provenance.notes == "planted"


def test_supporting_evidence_lands_on_the_claim_it_states() -> None:
    bundle = build_bundle(SyntheticConfig(seed=101, chains=2, chain_length=3))
    graph = bundle.graph()
    for item, finding in zip(bundle.evidence, bundle.findings, strict=True):
        if finding.is_negated:
            continue
        owner = graph.claims_for_evidence(item.evidence_id)[0]
        assert owner == finding.claim_id
        assert graph.claim(owner).statement == finding.link.statement
