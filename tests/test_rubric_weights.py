"""Rubric weighting: normalization, totals and validation."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.tournament import RubricScore, RubricWeights, weighted_total


def test_defaults_sum_to_one() -> None:
    weights = RubricWeights()
    assert weights.total() == pytest.approx(1.0)
    assert weights.as_dict() == {
        "novelty": 0.25,
        "testability": 0.30,
        "grounding": 0.30,
        "consistency": 0.15,
    }


def test_normalization_makes_scale_irrelevant() -> None:
    small = RubricWeights(0.1, 0.1, 0.1, 0.1)
    large = RubricWeights(10.0, 10.0, 10.0, 10.0)
    assert small.normalized().as_dict() == large.normalized().as_dict()
    assert small.fingerprint() == large.fingerprint()


def test_negative_and_all_zero_weights_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must be >= 0"):
        RubricWeights(novelty=-0.5)
    with pytest.raises(ValidationError, match="more than zero"):
        RubricWeights(0.0, 0.0, 0.0, 0.0)


def test_unknown_dimension_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown rubric dimension"):
        RubricWeights().weight("elegance")


def test_weighted_total_stays_within_bounds() -> None:
    weights = RubricWeights()
    low = RubricScore(0.0, 0.0, 0.0, 0.0)
    high = RubricScore(1.0, 1.0, 1.0, 1.0)
    assert weighted_total(low, weights) == 0.0
    assert weighted_total(high, weights) == pytest.approx(1.0)


def test_weighted_total_follows_the_weights() -> None:
    score = RubricScore(1.0, 0.0, 0.0, 0.0)
    assert weighted_total(score, RubricWeights(1.0, 0.0, 0.0, 0.0)) == pytest.approx(
        1.0
    )
    assert weighted_total(score, RubricWeights(0.0, 1.0, 1.0, 1.0)) == 0.0
    assert weighted_total(score, RubricWeights()) == pytest.approx(0.25)


def test_uniform_weights_match_the_plain_mean() -> None:
    score = RubricScore(0.2, 0.4, 0.6, 0.8)
    uniform = RubricWeights(1.0, 1.0, 1.0, 1.0)
    assert weighted_total(score, uniform) == pytest.approx(score.mean())
