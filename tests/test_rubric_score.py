"""Rubric scores: bounds, accessors and canonical ordering."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.tournament import RUBRIC_DIMENSIONS, RubricScore


def score(**overrides: float) -> RubricScore:
    payload: dict[str, float] = {
        "novelty": 0.5,
        "testability": 0.6,
        "grounding": 0.7,
        "consistency": 0.8,
    }
    payload.update(overrides)
    return RubricScore(**payload)


def test_dimensions_are_golden() -> None:
    assert RUBRIC_DIMENSIONS == (
        "novelty",
        "testability",
        "grounding",
        "consistency",
    )


def test_scores_expose_every_dimension() -> None:
    produced = score()
    assert produced.as_dict() == {
        "novelty": 0.5,
        "testability": 0.6,
        "grounding": 0.7,
        "consistency": 0.8,
    }
    assert produced.as_tuple() == (0.5, 0.6, 0.7, 0.8)
    assert produced.dimension("grounding") == 0.7
    assert produced.mean() == pytest.approx(0.65)


def test_bounds_are_inclusive_and_enforced() -> None:
    assert score(novelty=0.0).novelty == 0.0
    assert score(novelty=1.0).novelty == 1.0
    with pytest.raises(ValidationError, match=r"within \[0, 1\]"):
        score(novelty=1.5)
    with pytest.raises(ValidationError, match=r"within \[0, 1\]"):
        score(consistency=-0.1)


def test_non_numeric_scores_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must be a number"):
        RubricScore(0.5, 0.5, True, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="must be a number"):
        RubricScore(0.5, 0.5, "high", 0.5)  # type: ignore[arg-type]


def test_unknown_dimensions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown rubric dimension"):
        score().dimension("elegance")


def test_scores_compare_structurally() -> None:
    assert score() == score()
    assert score() != score(novelty=0.4)
