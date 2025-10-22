"""Probability/odds conversions: round trips, clamping and validation."""

from __future__ import annotations

import math

import pytest

from hypoarena.belief import (
    DEFAULT_PRIOR,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    clamp_probability,
    from_odds,
    to_odds,
)
from hypoarena.errors import ValidationError


def test_conversions_round_trip() -> None:
    for probability in (0.01, 0.25, DEFAULT_PRIOR, 0.75, 0.99):
        assert from_odds(to_odds(probability)) == pytest.approx(probability)


def test_even_priors_have_unit_odds() -> None:
    assert to_odds(0.5) == pytest.approx(1.0)
    assert from_odds(1.0) == pytest.approx(0.5)


def test_known_conversions_are_exact() -> None:
    assert to_odds(0.75) == pytest.approx(3.0)
    assert to_odds(0.9) == pytest.approx(9.0)
    assert from_odds(3.0) == pytest.approx(0.75)


def test_extremes_are_clamped_so_beliefs_stay_revisable() -> None:
    assert to_odds(0.0) == pytest.approx(to_odds(MIN_PROBABILITY))
    assert to_odds(1.0) == pytest.approx(to_odds(MAX_PROBABILITY))
    assert math.isfinite(to_odds(1.0))
    assert clamp_probability(0.0) == MIN_PROBABILITY
    assert clamp_probability(1.0) == MAX_PROBABILITY
    assert clamp_probability(0.4) == 0.4


def test_invalid_probabilities_and_odds_are_rejected() -> None:
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        to_odds(1.5)
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        to_odds(-0.1)
    with pytest.raises(ValidationError, match="must be a number"):
        to_odds("0.5")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="must be a number"):
        to_odds(True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="odds"):
        from_odds(-1.0)


def test_odds_are_monotone_in_the_probability() -> None:
    values = [to_odds(value) for value in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert values == sorted(values)
