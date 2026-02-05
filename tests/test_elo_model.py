"""Elo/Bradley–Terry maths: expectations, updates and outcome classification."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.tournament import (
    DEFAULT_INITIAL_RATING,
    DEFAULT_K_FACTOR,
    DEFAULT_RATING_SCALE,
    EloModel,
)


def test_defaults_are_golden() -> None:
    model = EloModel()
    assert model.initial == DEFAULT_INITIAL_RATING == 1500.0
    assert model.k_factor == DEFAULT_K_FACTOR == 32.0
    assert model.scale == DEFAULT_RATING_SCALE == 400.0
    assert model.draw_margin == 0.05


def test_equal_ratings_expect_a_draw() -> None:
    assert EloModel().expectation(1500.0, 1500.0) == 0.5


def test_expectations_are_complementary() -> None:
    model = EloModel()
    for left, right in ((1500.0, 1600.0), (1200.0, 1800.0), (2000.0, 1000.0)):
        assert model.expectation(left, right) + model.expectation(right, left) == (
            pytest.approx(1.0)
        )


def test_expectation_is_monotone_in_the_rating_gap() -> None:
    model = EloModel()
    gaps = [model.expectation(1500.0 + gap, 1500.0) for gap in (0, 100, 200, 400)]
    assert gaps == sorted(gaps)
    assert 0.0 < gaps[0] < gaps[-1] < 1.0


def test_the_winner_gains_and_the_loser_loses() -> None:
    model = EloModel()
    left, right = model.update(1500.0, 1500.0, 1.0)
    assert left > 1500.0
    assert right < 1500.0
    assert left - 1500.0 == pytest.approx(1500.0 - right)


def test_updates_are_zero_sum_for_equal_experience() -> None:
    model = EloModel()
    for outcome in (0.0, 0.5, 1.0):
        left, right = model.update(
            1400.0, 1600.0, outcome, left_played=3, right_played=3
        )
        assert left + right == pytest.approx(3000.0)


def test_a_draw_moves_ratings_toward_each_other() -> None:
    model = EloModel()
    left, right = model.update(1700.0, 1300.0, 0.5)
    assert left < 1700.0
    assert right > 1300.0


def test_an_upset_win_moves_ratings_further_than_an_expected_win() -> None:
    model = EloModel()
    upset_left, _ = model.update(1200.0, 1800.0, 1.0)
    expected_left, _ = model.update(1800.0, 1200.0, 1.0)
    assert upset_left - 1200.0 > expected_left - 1800.0


def test_outcome_classification_uses_the_draw_margin() -> None:
    model = EloModel(draw_margin=0.05)
    assert model.outcome(0.80, 0.60) == 1.0
    assert model.outcome(0.60, 0.80) == 0.0
    assert model.outcome(0.80, 0.78) == 0.5
    assert model.outcome(0.80, 0.74) == 1.0
    assert EloModel(draw_margin=0.0).outcome(0.80, 0.80) == 0.5


def test_invalid_outcomes_and_settings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="outcome"):
        EloModel().update(1500.0, 1500.0, 0.75)
    with pytest.raises(ValidationError, match="k_factor"):
        EloModel(k_factor=0.0)
    with pytest.raises(ValidationError, match="k_decay"):
        EloModel(k_decay=1.5)
    with pytest.raises(ValidationError, match="k_floor"):
        EloModel(k_floor=64.0)
    with pytest.raises(ValidationError, match="scale"):
        EloModel(scale=0.0)
    with pytest.raises(ValidationError, match="draw_margin"):
        EloModel(draw_margin=1.0)


def test_fingerprint_tracks_the_settings() -> None:
    assert EloModel().fingerprint() == EloModel().fingerprint()
    assert EloModel(k_factor=16.0).fingerprint() != EloModel().fingerprint()
