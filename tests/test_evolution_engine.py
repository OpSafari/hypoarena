"""The evolution engine: candidate choice, insertion semantics and bookkeeping."""

from __future__ import annotations

import pytest

from helpers import sample_claim
from hypoarena.errors import ValidationError
from hypoarena.evolve import (
    CONDITION_POOL,
    OPERATOR_EDGES,
    VARIABLE_POOL,
    EvolutionEngine,
    EvolutionStep,
    Rejection,
)
from hypoarena.graph import HypothesisGraph
from hypoarena.schema import ClaimRelation, PredictedRelation
from hypoarena.synthetic import SyntheticConfig, build_bundle


def graph_from_bundle(seed: int = 5) -> HypothesisGraph:
    return build_bundle(SyntheticConfig(seed=seed, chains=2, chain_length=3)).graph()


def test_operator_edge_semantics_are_golden() -> None:
    assert OPERATOR_EDGES == {
        "narrow_scope": ClaimRelation.REFINES,
        "substitute_variable": None,
        "flip_relation": ClaimRelation.CONTRADICTS,
        "crossover": None,
        "decompose": ClaimRelation.ENTAILS,
    }
    assert CONDITION_POOL and VARIABLE_POOL


def test_a_step_inserts_children_and_keeps_the_graph_valid() -> None:
    graph = graph_from_bundle()
    before = len(graph)
    step = EvolutionEngine(seed=3).step(graph, generation=1)
    assert isinstance(step, EvolutionStep)
    assert step.generation == 1
    assert len(graph) == before + step.accepted_count
    assert graph.validate() is None
    assert set(step.child_ids) <= set(graph.claim_ids)


def test_accepted_children_carry_evolved_provenance() -> None:
    graph = graph_from_bundle()
    step = EvolutionEngine(seed=3).step(graph, generation=1)
    for record in step.accepted:
        child = graph.claim(record.child.claim_id)
        assert child.provenance.origin == "evolved"
        assert child.provenance.notes == record.operator
        assert child.provenance.parents == record.parents
        assert all(graph.has_claim(parent) for parent in record.parents)


def test_narrowing_adds_a_refines_edge() -> None:
    graph = HypothesisGraph()
    parent = sample_claim(statement="Protein A increases gene B expression")
    graph.add_claim(parent)
    step = EvolutionEngine(seed=1, operators=("narrow_scope",)).step(graph, 1)
    assert step.accepted_count == 1
    child_id = step.child_ids[0]
    assert graph.has_edge(parent.claim_id, child_id, ClaimRelation.REFINES)
    assert graph.edge_count == 1


def test_flipping_adds_a_contradiction_edge() -> None:
    graph = HypothesisGraph()
    parent = sample_claim(statement="Protein A increases gene B expression")
    graph.add_claim(parent)
    step = EvolutionEngine(seed=1, operators=("flip_relation",)).step(graph, 1)
    assert step.accepted_count == 1
    child_id = step.child_ids[0]
    assert graph.has_edge(parent.claim_id, child_id, ClaimRelation.CONTRADICTS)
    assert graph.claim(child_id).relation is PredictedRelation.DECREASES


def test_a_flipped_child_is_not_flipped_again_into_a_duplicate() -> None:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim(statement="Protein A increases gene B expression"))
    engine = EvolutionEngine(seed=1, operators=("flip_relation",))
    engine.step(graph, 1)
    second = engine.step(graph, 2)
    # whichever claim was chosen, the graph stays valid and duplicate-free
    assert graph.validate() is None
    assert len({claim.claim_id for claim in graph.claims}) == len(graph)
    assert second.generation == 2


def test_refusals_are_recorded_with_a_reason() -> None:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim(relation=PredictedRelation.CAUSES))
    step = EvolutionEngine(seed=2, operators=("flip_relation",)).step(graph, 1)
    assert step.accepted_count == 0
    assert len(step.rejected) == 1
    assert isinstance(step.rejected[0], Rejection)
    assert step.rejected[0].reason == "operator produced no child"


def test_runs_number_generations_from_the_graph_state() -> None:
    graph = graph_from_bundle()
    engine = EvolutionEngine(seed=4)
    steps = engine.run(graph, generations=3)
    # bundle claims are synthetic (generation 0), so the run starts at 1
    assert [step.generation for step in steps] == [1, 2, 3]
    assert graph.validate() is None
    with pytest.raises(ValidationError, match="generations"):
        engine.run(graph, generations=0)


def test_engine_settings_are_validated() -> None:
    with pytest.raises(ValidationError, match="unknown evolution operator"):
        EvolutionEngine(operators=("teleport",))
    with pytest.raises(ValidationError, match="at least one operator"):
        EvolutionEngine(operators=())
    with pytest.raises(ValidationError, match="per_operator"):
        EvolutionEngine(per_operator=0)


def test_a_negative_generation_is_rejected() -> None:
    graph = graph_from_bundle()
    with pytest.raises(ValidationError, match="generation"):
        EvolutionEngine().step(graph, -1)


def test_step_summaries_are_serializable_and_stable() -> None:
    first = EvolutionEngine(seed=7).step(graph_from_bundle(), 1)
    second = EvolutionEngine(seed=7).step(graph_from_bundle(), 1)
    assert first.as_dict() == second.as_dict()
    assert first.signature() == second.signature()
    assert sorted(first.as_dict()) == ["accepted", "generation", "rejected"]
