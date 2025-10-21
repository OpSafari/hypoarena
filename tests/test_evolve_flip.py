"""Relation flipping: opposite directions, statement rewriting and refusals."""

from __future__ import annotations

from helpers import sample_claim
from hypoarena.evolve import flip_relation
from hypoarena.schema import PredictedRelation


def test_flipping_increases_yields_decreases() -> None:
    parent = sample_claim(
        statement="Protein A increases gene B expression",
        relation=PredictedRelation.INCREASES,
    )
    child = flip_relation(parent)
    assert child is not None
    assert child.relation is PredictedRelation.DECREASES
    assert child.statement == "Protein A decreases gene B expression"


def test_flipping_preserves_capitalisation() -> None:
    parent = sample_claim(
        statement="kinase K1 inhibits gene G1", relation=PredictedRelation.INHIBITS
    )
    child = flip_relation(parent)
    assert child is not None
    assert child.statement == "kinase K1 enables gene G1"


def test_flipping_twice_returns_to_the_original_relation() -> None:
    parent = sample_claim(relation=PredictedRelation.ENABLES)
    child = flip_relation(parent)
    assert child is not None
    restored = flip_relation(child)
    assert restored is not None
    assert restored.relation is parent.relation
    assert restored.claim_id != parent.claim_id


def test_relations_without_an_opposite_are_refused() -> None:
    assert flip_relation(sample_claim(relation=PredictedRelation.CAUSES)) is None
    assert flip_relation(sample_claim(relation=PredictedRelation.ASSOCIATES)) is None


def test_a_statement_without_the_verb_is_rebuilt() -> None:
    parent = sample_claim(
        statement="A effect on B was measured repeatedly",
        subject="protein A",
        object="cell growth",
        relation=PredictedRelation.INCREASES,
    )
    child = flip_relation(parent)
    assert child is not None
    assert child.statement == "protein A decreases cell growth"


def test_provenance_marks_the_operator_and_the_parent() -> None:
    parent = sample_claim()
    child = flip_relation(parent)
    assert child is not None
    assert child.provenance.notes == "flip_relation"
    assert child.provenance.parents == (parent.claim_id,)
    assert child.citations == parent.citations
