"""K-factor scheduling and the bounds it puts on rating movement.

A decaying K is what makes a tournament converge: early matches move ratings a
lot, later matches only refine them. These tests pin that behaviour so a change
to the schedule shows up as a change in convergence, not as a silent drift.
"""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.tournament import EloModel


def test_the_first_matches_use_the_full_k_factor() -> None:
    model = EloModel(k_factor=32.0, k_decay=0.97, k_floor=6.0)
    assert model.k_for(0) == 32.0
    assert model.k_for(1) == pytest.approx(32.0 * 0.97)


def test_k_decays_monotonically_to_the_floor() -> None:
    model = EloModel(k_factor=32.0, k_decay=0.5, k_floor=6.0)
    values = [model.k_for(played) for played in range(12)]
    assert values == sorted(values, reverse=True)
    assert values[-1] == 6.0
    assert all(value >= model.k_floor for value in values)


def test_a_decay_of_one_keeps_k_constant() -> None:
    model = EloModel(k_decay=1.0)
    assert model.k_for(0) == model.k_for(100)


def test_negative_experience_is_rejected() -> None:
    with pytest.raises(ValidationError, match="played"):
        EloModel().k_for(-1)


def test_rating_steps_are_bounded_by_the_current_k() -> None:
    model = EloModel(k_factor=32.0)
    for played in (0, 5, 50):
        for outcome in (0.0, 0.5, 1.0):
            left, right = model.update(
                1500.0, 1500.0, outcome, left_played=played, right_played=played
            )
            assert abs(left - 1500.0) <= model.k_for(played) + 1e-9
            assert abs(right - 1500.0) <= model.k_for(played) + 1e-9


def test_a_win_streak_produces_shrinking_steps() -> None:
    model = EloModel(k_factor=32.0, k_decay=0.9)
    rating = model.initial
    steps = []
    for played in range(8):
        updated, _ = model.update(rating, 1500.0, 1.0, left_played=played)
        steps.append(updated - rating)
        rating = updated
    assert all(step > 0 for step in steps)
    assert steps == sorted(steps, reverse=True)


def test_ratings_converge_toward_a_stable_gap() -> None:
    model = EloModel(k_factor=32.0, k_decay=0.8, k_floor=1.0)
    strong, weak = model.initial, model.initial
    gaps = []
    for played in range(30):
        strong, weak = model.update(
            strong, weak, 1.0, left_played=played, right_played=played
        )
        gaps.append(strong - weak)
    assert gaps[-1] > gaps[0]
    assert gaps[-1] - gaps[-2] < gaps[1] - gaps[0]
    assert gaps[-1] < 1000.0


def test_asymmetric_experience_breaks_zero_sum_by_design() -> None:
    model = EloModel(k_factor=32.0, k_decay=0.5)
    before = 1500.0 + 1500.0
    left, right = model.update(1500.0, 1500.0, 1.0, left_played=0, right_played=10)
    assert left + right != pytest.approx(before)
    assert left - 1500.0 > 1500.0 - right
