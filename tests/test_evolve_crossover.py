"""Crossover: what is merged, and when merging is refused."""

from __future__ import annotations

from helpers import sample_citation, sample_claim
from hypoarena.evolve import crossover, merge_citations, shared_variables
from hypoarena.schema import PredictedRelation, Scope


def parents() -> tuple[object, object]:
    first = sample_claim(
        claim_id="clm_0123456789ab",
        subject="protein A",
        object="gene B expression",
        relation=PredictedRelation.INCREASES,
        scope=Scope("hek293 cells", ("hypoxia",)),
        mechanism=None,
        citations=(sample_citation(),),
    )
    second = sample_claim(
        claim_id="clm_ffffffffffff",
        subject="gene B expression",
        object="cell growth",
        relation=PredictedRelation.ENABLES,
        scope=Scope("hek293 cells", ("serum starved",)),
        mechanism="promoter remodelling",
        citations=(sample_citation(start=40, end=60, quote="promoter remodelling"),),
    )
    return first, second


def test_the_child_combines_variables_mechanism_and_scope() -> None:
    first, second = parents()
    child = crossover(first, second)
    assert child is not None
    assert child.subject == "protein A"
    assert child.object == "cell growth"
    assert child.relation is PredictedRelation.INCREASES
    assert child.mechanism == "promoter remodelling"
    assert child.scope.conditions == ("hypoxia", "serum starved")
    assert "through promoter remodelling" in child.statement


def test_citations_are_unioned_without_duplicates() -> None:
    first, second = parents()
    child = crossover(first, second)
    assert child is not None
    assert len(child.citations) == 2
    assert merge_citations((first, first)) == first.citations


def test_provenance_names_both_parents() -> None:
    first, second = parents()
    child = crossover(first, second)
    assert child is not None
    assert child.provenance.parents == (first.claim_id, second.claim_id)
    assert child.provenance.notes == "crossover"
    assert child.provenance.generation == first.provenance.generation + 1


def test_unrelated_parents_are_refused() -> None:
    first, _ = parents()
    unrelated = sample_claim(
        claim_id="clm_222222222222",
        subject="kinase K1",
        object="splice fidelity",
        scope=Scope("yeast lysate"),
    )
    assert crossover(first, unrelated) is None
    assert shared_variables(first, unrelated) == frozenset()


def test_a_different_population_blocks_crossover() -> None:
    first, second = parents()
    elsewhere = sample_claim(
        claim_id=second.claim_id,
        subject=second.subject,
        object=second.object,
        scope=Scope("mouse liver"),
    )
    assert crossover(first, elsewhere) is None


def test_self_crossover_is_refused() -> None:
    first, _ = parents()
    assert crossover(first, first) is None


def test_crossover_is_deterministic() -> None:
    first, second = parents()
    assert crossover(first, second) == crossover(first, second)


def test_a_degenerate_combination_is_refused() -> None:
    # the child would relate "protein A" to itself, so crossover refuses
    first = sample_claim(
        claim_id="clm_0123456789ab",
        subject="protein A",
        object="gene B expression",
        scope=Scope("hek293 cells"),
    )
    second = sample_claim(
        claim_id="clm_ffffffffffff",
        subject="gene B expression",
        object="protein A",
        scope=Scope("hek293 cells"),
    )
    assert shared_variables(first, second)
    assert crossover(first, second) is None
