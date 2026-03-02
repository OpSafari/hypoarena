"""Single-evidence updates and the belief state record."""

from __future__ import annotations

import pytest

from hypoarena.belief import BeliefState, update_belief
from hypoarena.errors import ValidationError


def state(**overrides: object) -> BeliefState:
    payload: dict[str, object] = {
        "claim_id": "clm_0123456789ab",
        "prior": 0.5,
        "posterior": 0.75,
        "likelihood_ratio": 3.0,
        "supporting": 1,
    }
    payload.update(overrides)
    return BeliefState(**payload)  # type: ignore[arg-type]


def test_a_full_strength_support_moves_a_half_prior_to_three_quarters() -> None:
    assert update_belief(0.5, 3.0) == pytest.approx(0.75)


def test_updates_compose_multiplicatively() -> None:
    once = update_belief(0.5, 3.0)
    twice = update_belief(once, 3.0)
    assert twice == pytest.approx(0.9)
    assert update_belief(0.5, 9.0) == pytest.approx(twice)


def test_refuting_ratios_move_the_belief_down() -> None:
    assert update_belief(0.5, 1 / 3) == pytest.approx(0.25)
    assert update_belief(0.75, 1 / 3) == pytest.approx(0.5)


def test_a_neutral_ratio_changes_nothing() -> None:
    assert update_belief(0.42, 1.0) == pytest.approx(0.42)


def test_extreme_ratios_stay_inside_the_open_interval() -> None:
    assert 0.0 < update_belief(0.5, 1e12) < 1.0
    assert 0.0 < update_belief(0.5, 1e-12) < 1.0


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValidationError, match="ratio"):
        update_belief(0.5, 0.0)
    with pytest.raises(ValidationError, match="ratio"):
        update_belief(0.5, -2.0)
    with pytest.raises(ValidationError, match="prior"):
        update_belief(1.2, 2.0)


def test_state_counts_and_derived_values() -> None:
    produced = state(refuting=2, neutral=1)
    assert produced.updates == 4
    assert produced.odds == pytest.approx(3.0)
    assert produced.movement == pytest.approx(0.25)


def test_state_dict_is_rounded_and_complete() -> None:
    payload = state().as_dict()
    assert sorted(payload) == [
        "claim_id",
        "likelihood_ratio",
        "movement",
        "neutral",
        "posterior",
        "prior",
        "refuting",
        "supporting",
    ]
    assert payload["posterior"] == 0.75


def test_state_validation_covers_every_field() -> None:
    with pytest.raises(ValidationError, match="claim id"):
        state(claim_id="  ")
    with pytest.raises(ValidationError, match="posterior"):
        state(posterior=1.5)
    with pytest.raises(ValidationError, match="likelihood_ratio"):
        state(likelihood_ratio=0.0)
    with pytest.raises(ValidationError, match="supporting"):
        state(supporting=-1)
