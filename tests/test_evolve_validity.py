"""Every operator must leave the graph valid — checked over seeded sweeps.

The engine builds children indirectly (new identifiers, new provenance, new
edges), so validity cannot be assumed from the individual operators alone. These
sweeps run whole generations over graphs derived from synthetic bundles and
assert the structural invariants after every insertion.
"""

from __future__ import annotations

import pytest

from hypoarena.evolve import EvolutionEngine
from hypoarena.ids import is_valid_id
from hypoarena.schema import ClaimRelation
from hypoarena.synthetic import SyntheticConfig, build_bundle

SEEDS = (1, 17, 270106)
CONFIGS = (
    {"chains": 2, "chain_length": 3},
    {"chains": 3, "chain_length": 2, "paraphrases_per_link": 2},
    {"chains": 1, "chain_length": 4, "competitors_per_chain": 2},
)


def graph_for(seed: int, **overrides: object) -> object:
    return build_bundle(
        SyntheticConfig(seed=seed, **overrides)  # type: ignore[arg-type]
    ).graph()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("overrides", CONFIGS)
def test_generations_keep_the_graph_valid(seed: int, overrides: dict) -> None:
    graph = graph_for(seed, **overrides)
    engine = EvolutionEngine(seed=seed)
    before = len(graph)
    for generation in (1, 2):
        step = engine.step(graph, generation)
        graph.validate()
        assert len(graph) == before + step.accepted_count
        before = len(graph)


@pytest.mark.parametrize("seed", SEEDS)
def test_children_are_well_formed_and_traceable(seed: int) -> None:
    graph = graph_for(seed, chains=2, chain_length=3)
    engine = EvolutionEngine(seed=seed)
    for generation in (1, 2, 3):
        for record in engine.step(graph, generation).accepted:
            child = record.child
            assert is_valid_id(child.claim_id)
            assert child.provenance.origin == "evolved"
            assert child.provenance.generation >= 1
            assert child.provenance.parents == record.parents
            for parent_id in record.parents:
                assert graph.has_claim(parent_id)
                assert graph.claim(parent_id).provenance.generation < (
                    child.provenance.generation
                )


@pytest.mark.parametrize("seed", SEEDS)
def test_edges_only_ever_connect_existing_claims(seed: int) -> None:
    graph = graph_for(seed, chains=2, chain_length=3)
    EvolutionEngine(seed=seed).run(graph, generations=3)
    for edge in graph.edges:
        assert graph.has_claim(edge.source)
        assert graph.has_claim(edge.target)
        assert edge.relation in set(ClaimRelation)
        assert edge.note in {
            "narrow_scope",
            "flip_relation",
            "decompose",
            "planted rival",
        }


@pytest.mark.parametrize("seed", SEEDS)
def test_identifiers_never_collide(seed: int) -> None:
    graph = graph_for(seed, chains=2, chain_length=3)
    EvolutionEngine(seed=seed).run(graph, generations=4)
    identifiers = [claim.claim_id for claim in graph.claims]
    assert len(identifiers) == len(set(identifiers))
    evidence_ids = [item.evidence_id for item in graph.evidence_items]
    assert len(evidence_ids) == len(set(evidence_ids))


@pytest.mark.parametrize("seed", SEEDS)
def test_evidence_links_survive_evolution(seed: int) -> None:
    bundle = build_bundle(SyntheticConfig(seed=seed, chains=2, chain_length=3))
    graph = bundle.graph()
    links_before = graph.link_count
    EvolutionEngine(seed=seed).run(graph, generations=3)
    assert graph.link_count == links_before
    for item in bundle.evidence:
        assert len(graph.claims_for_evidence(item.evidence_id)) == 1
