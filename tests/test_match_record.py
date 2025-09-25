"""Match audit records: validation, perspective helpers and serialization."""

from __future__ import annotations

import pytest

from hypoarena.errors import UnknownReferenceError, ValidationError
from hypoarena.tournament import MatchResult, RubricScore

LEFT = "clm_0123456789ab"
RIGHT = "clm_ffffffffffff"


def match(**overrides: object) -> MatchResult:
    payload: dict[str, object] = {
        "left": LEFT,
        "right": RIGHT,
        "left_score": RubricScore(0.8, 0.7, 0.9, 0.6),
        "right_score": RubricScore(0.4, 0.5, 0.3, 0.4),
        "left_total": 0.77,
        "right_total": 0.41,
        "outcome": 1.0,
        "judge": "features",
    }
    payload.update(overrides)
    return MatchResult(**payload)  # type: ignore[arg-type]


def test_a_win_reports_the_winner_and_margin() -> None:
    produced = match()
    assert produced.winner == LEFT
    assert produced.is_draw is False
    assert produced.margin == pytest.approx(0.36)
    assert produced.pair() == tuple(sorted((LEFT, RIGHT)))


def test_outcomes_are_reported_from_either_perspective() -> None:
    produced = match()
    assert produced.outcome_for(LEFT) == 1.0
    assert produced.outcome_for(RIGHT) == 0.0
    assert produced.total_for(RIGHT) == 0.41
    drawn = match(outcome=0.5, left_total=0.5, right_total=0.5)
    assert drawn.outcome_for(LEFT) == 0.5
    assert drawn.winner is None
    assert drawn.is_draw is True


def test_unknown_subjects_are_rejected() -> None:
    with pytest.raises(UnknownReferenceError):
        match().total_for("clm_999999999999")
    with pytest.raises(UnknownReferenceError):
        match().outcome_for("clm_999999999999")


def test_self_matches_and_blank_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="two different subjects"):
        match(right=LEFT)
    with pytest.raises(ValidationError, match="blank"):
        match(judge="  ")


def test_invalid_outcomes_totals_and_indices_are_rejected() -> None:
    with pytest.raises(ValidationError, match="outcome"):
        match(outcome=0.3)
    with pytest.raises(ValidationError, match="left_total"):
        match(left_total=1.5)
    with pytest.raises(ValidationError, match="right_total"):
        match(right_total=-0.1)
    with pytest.raises(ValidationError, match="round_index"):
        match(round_index=-1)
    with pytest.raises(ValidationError, match="match_index"):
        match(match_index=-2)


def test_dict_view_carries_the_whole_audit_trail() -> None:
    produced = match(round_index=2, match_index=7, seed=270106)
    payload = produced.as_dict()
    assert sorted(payload) == [
        "judge",
        "left",
        "left_score",
        "left_total",
        "match_index",
        "outcome",
        "right",
        "right_score",
        "right_total",
        "round_index",
        "seed",
        "winner",
    ]
    assert payload["left_score"] == {
        "novelty": 0.8,
        "testability": 0.7,
        "grounding": 0.9,
        "consistency": 0.6,
    }
    assert payload["winner"] == LEFT
    assert payload["seed"] == 270106
