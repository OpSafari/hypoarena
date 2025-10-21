"""Canonical relation verbs: the single source of truth for statement wording."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.schema import (
    CANONICAL_RELATION_VERBS,
    PredictedRelation,
    canonical_statement,
    canonical_verb,
)
from hypoarena.synthetic import RELATION_VERBS, relation_verbs


def test_every_relation_has_a_canonical_verb() -> None:
    assert set(CANONICAL_RELATION_VERBS) == set(PredictedRelation)
    assert canonical_verb(PredictedRelation.INCREASES) == "increases"
    assert canonical_verb(PredictedRelation.ASSOCIATES) == "is associated with"


def test_the_synthetic_vocabulary_stays_consistent() -> None:
    for relation in PredictedRelation:
        assert relation_verbs(relation)[0] == canonical_verb(relation)
        assert relation_verbs(relation) == RELATION_VERBS[relation]


def test_canonical_statements_are_deterministic() -> None:
    statement = canonical_statement(
        "protein A", PredictedRelation.INCREASES, "cell growth"
    )
    assert statement == "protein A increases cell growth"
    assert canonical_statement(
        "protein A", PredictedRelation.DECREASES, "cell growth"
    ) != (statement)


def test_blank_variables_are_rejected() -> None:
    with pytest.raises(ValidationError, match="non-blank"):
        canonical_statement("  ", PredictedRelation.CAUSES, "cell growth")
    with pytest.raises(ValidationError, match="non-blank"):
        canonical_statement("protein A", PredictedRelation.CAUSES, " ")
