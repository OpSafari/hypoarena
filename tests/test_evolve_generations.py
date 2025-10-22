"""Multi-generation behaviour: bounded growth, determinism, saturation.

Growth is bounded by construction — at most one child per operator slot per
generation — and the novelty gate is what stops a long run from filling the
graph with restatements. Both properties are measured here rather than assumed.
"""

from __future__ import annotations

import pytest

from hypoarena.evolve import OPERATORS, EvolutionEngine
from hypoarena.synthetic import SyntheticConfig, build_bundle


def graph_for(seed: int = 21) -> object:
    return build_bundle(SyntheticConfig(seed=seed, chains=2, chain_length=3)).graph()


def test_growth_is_bounded_by_the_operator_slots() -> None:
    graph = graph_for()
    before = len(graph)
    engine = EvolutionEngine(seed=5, per_operator=2)
    steps = engine.run(graph, generations=3)
    assert len(graph) - before == sum(step.accepted_count for step in steps)
    assert len(graph) - before <= 3 * len(OPERATORS) * 2


def test_single_operator_runs_grow_at_most_one_claim_per_generation() -> None:
    graph = graph_for()
    engine = EvolutionEngine(seed=5, operators=("narrow_scope",))
    before = len(graph)
    engine.run(graph, generations=4)
    assert len(graph) - before <= 4


def test_runs_are_reproducible_for_a_seed() -> None:
    first = graph_for()
    second = graph_for()
    steps_a = EvolutionEngine(seed=9).run(first, generations=3)
    steps_b = EvolutionEngine(seed=9).run(second, generations=3)
    assert [step.signature() for step in steps_a] == [
        step.signature() for step in steps_b
    ]
    assert first.signature() == second.signature()


def test_different_seeds_produce_different_lineages() -> None:
    first = graph_for()
    second = graph_for()
    EvolutionEngine(seed=1).run(first, generations=3)
    EvolutionEngine(seed=2).run(second, generations=3)
    assert first.signature() != second.signature()


def test_novelty_saturation_shows_up_as_rejections() -> None:
    graph = graph_for()
    engine = EvolutionEngine(seed=11, operators=("narrow_scope",), per_operator=3)
    steps = engine.run(graph, generations=6)
    rejected = sum(len(step.rejected) for step in steps)
    accepted = sum(step.accepted_count for step in steps)
    assert rejected > 0
    assert accepted > 0
    reasons = {rejection.reason for step in steps for rejection in step.rejected}
    assert reasons <= {
        "not novel",
        "operator produced no child",
        "child already present",
    }


def test_generations_count_up_from_the_deepest_parent() -> None:
    graph = graph_for()
    steps = EvolutionEngine(seed=13).run(graph, generations=4)
    generations = [
        record.child.provenance.generation for step in steps for record in step.accepted
    ]
    assert min(generations) >= 1
    assert max(generations) >= 2


def test_a_wider_operator_set_produces_more_lineages() -> None:
    narrow = graph_for()
    wide = graph_for()
    EvolutionEngine(seed=3, operators=("narrow_scope",)).run(narrow, generations=3)
    EvolutionEngine(seed=3).run(wide, generations=3)
    assert len(wide) >= len(narrow)
    notes = {
        claim.provenance.notes
        for claim in wide.claims
        if claim.provenance.origin == "evolved"
    }
    assert len(notes) >= len(
        {
            claim.provenance.notes
            for claim in narrow.claims
            if claim.provenance.origin == "evolved"
        }
    )


@pytest.mark.parametrize("per_operator", [1, 2, 3])
def test_per_operator_scales_the_number_of_attempts(per_operator: int) -> None:
    graph = graph_for()
    steps = EvolutionEngine(seed=15, per_operator=per_operator).run(
        graph, generations=2
    )
    attempts = sum(len(step.accepted) + len(step.rejected) for step in steps)
    assert attempts <= 2 * len(OPERATORS) * per_operator
    assert attempts >= 1
