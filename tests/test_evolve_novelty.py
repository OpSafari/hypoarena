"""Novelty gating: acceptance, refusal and the rationale text."""

from __future__ import annotations

import pytest

from helpers import sample_claim
from hypoarena.dedup import DedupConfig
from hypoarena.errors import ValidationError
from hypoarena.evolve import (
    EvolutionRecord,
    NoveltyGate,
    Rejection,
    crossover,
    decompose,
    flip_relation,
    narrow_scope,
    rationale_for,
    substitute_variable,
)
from hypoarena.schema import PredictedRelation


def gate(threshold: float = 0.9) -> NoveltyGate:
    return NoveltyGate(
        DedupConfig(
            method="jaccard",
            threshold=threshold,
            ngram_size=1,
            shingle_unit="word",
            content_only=True,
        )
    )


def test_rationales_describe_the_actual_change() -> None:
    parent = sample_claim(statement="Protein A increases gene B expression")
    child = narrow_scope(parent, "hypoxia")
    assert rationale_for("narrow_scope", [parent], child) == "scope narrowed by hypoxia"
    substituted = substitute_variable(parent, "subject", "kinase K1")
    assert "subject changed from" in rationale_for(
        "substitute_variable", [parent], substituted
    )
    flipped = flip_relation(parent)
    assert flipped is not None
    assert rationale_for("flip_relation", [parent], flipped) == (
        "relation flipped from increases to decreases"
    )


def test_unknown_operators_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown evolution operator"):
        rationale_for("teleport", [sample_claim()], sample_claim())


def test_a_novel_child_is_accepted_with_its_rationale() -> None:
    produced = gate()
    parent = sample_claim(statement="Protein A increases gene B expression")
    child = narrow_scope(parent, "hypoxia")
    record = produced.consider("narrow_scope", [parent], child)
    assert record.child.claim_id == child.claim_id
    assert record.rationale == "scope narrowed by hypoxia"
    assert produced.guard.size == 1


def test_a_restatement_is_refused_with_evidence() -> None:
    produced = gate()
    parent = sample_claim(statement="Protein A increases gene B expression")
    produced.consider("narrow_scope", [parent], narrow_scope(parent, "hypoxia"))
    twin = narrow_scope(parent, "hypoxia")
    refusal = produced.consider("narrow_scope", [parent], twin)
    assert isinstance(refusal, Rejection)
    assert refusal.reason == "not novel"
    assert refusal.similarity == 1.0
    assert produced.guard.size == 1


def test_a_lower_threshold_refuses_more() -> None:
    parent = sample_claim(statement="Protein A increases gene B expression")
    child = narrow_scope(parent, "hypoxia")
    strict = gate(threshold=1.0)
    strict.preload([parent])
    loose = gate(threshold=0.3)
    loose.preload([parent])
    accepted = strict.consider("narrow_scope", [parent], child)
    assert isinstance(accepted, EvolutionRecord)
    assert accepted.rationale.startswith("scope narrowed")
    refusal = loose.consider("narrow_scope", [parent], child)
    assert isinstance(refusal, Rejection)
    assert refusal.nearest == parent.claim_id


def test_preloading_seeds_the_guard_from_a_graph() -> None:
    produced = gate()
    claims = [
        sample_claim(claim_id="clm_0123456789ab", statement="first statement here"),
        sample_claim(claim_id="clm_ffffffffffff", statement="second statement here"),
    ]
    assert produced.preload(claims) == 2
    assert produced.preload(claims) == 0
    assert produced.guard.size == 2


def test_crossover_children_pass_through_the_gate() -> None:
    produced = gate()
    first = sample_claim(
        claim_id="clm_0123456789ab",
        subject="protein A",
        object="gene B expression",
        mechanism=None,
    )
    second = sample_claim(
        claim_id="clm_ffffffffffff",
        subject="gene B expression",
        object="cell growth",
        mechanism="promoter remodelling",
    )
    child = crossover(first, second)
    assert child is not None
    record = produced.consider("crossover", [first, second], child)
    assert "combined" in record.rationale


def test_decomposed_children_are_gated_individually() -> None:
    produced = gate()
    parent = sample_claim(
        statement="Protein A binds the promoter and gene B is expressed"
    )
    children = decompose(parent)
    outcomes = [produced.consider("decompose", [parent], child) for child in children]
    assert all(isinstance(outcome, Rejection) is False for outcome in outcomes)
    assert produced.guard.size == 2


def test_relation_flips_are_not_confused_with_their_parents() -> None:
    produced = gate()
    parent = sample_claim(
        statement="Protein A increases gene B expression",
        relation=PredictedRelation.INCREASES,
    )
    flipped = flip_relation(parent)
    assert flipped is not None
    accepted = produced.consider("flip_relation", [parent], flipped)
    assert accepted.rationale == "relation flipped from increases to decreases"
