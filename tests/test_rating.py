"""Rating records: tally invariants, derived rates and advancement."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.tournament import DEFAULT_INITIAL_RATING, Rating


def test_a_fresh_rating_starts_at_the_model_initial_value() -> None:
    rating = Rating("clm_0123456789ab")
    assert rating.elo == DEFAULT_INITIAL_RATING
    assert (rating.played, rating.wins, rating.losses, rating.draws) == (0, 0, 0, 0)
    assert rating.win_rate == 0.0
    assert rating.score_points == 0.0


def test_blank_subjects_and_negative_counts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="blank"):
        Rating("  ")
    with pytest.raises(ValidationError, match="wins"):
        Rating("a", played=1, wins=-1, losses=2)


def test_tally_mismatches_are_rejected() -> None:
    with pytest.raises(ValidationError, match="tally disagrees"):
        Rating("a", played=3, wins=1, losses=1, draws=0)
    assert Rating("a", played=3, wins=1, losses=1, draws=1).played == 3


def test_advancement_updates_the_matching_tally() -> None:
    rating = Rating("a")
    won = rating.advanced(1.0, 1516.0)
    assert (won.wins, won.losses, won.draws, won.played) == (1, 0, 0, 1)
    assert won.elo == 1516.0
    drawn = won.advanced(0.5, 1516.0)
    assert (drawn.wins, drawn.draws, drawn.played) == (1, 1, 2)
    lost = drawn.advanced(0.0, 1500.0)
    assert (lost.wins, lost.losses, lost.draws, lost.played) == (1, 1, 1, 3)


def test_advancement_rejects_unknown_outcomes() -> None:
    with pytest.raises(ValidationError, match="outcome"):
        Rating("a").advanced(0.25, 1500.0)


def test_win_rate_and_points_follow_the_tally() -> None:
    rating = Rating("a", played=4, wins=3, losses=1, draws=0)
    assert rating.score_points == 3.0
    assert rating.win_rate == 0.75
    mixed = Rating("b", played=4, wins=1, losses=1, draws=2)
    assert mixed.score_points == 2.0
    assert mixed.win_rate == 0.5


def test_dict_view_is_rounded_and_complete() -> None:
    rating = Rating("a", elo=1500.123456, played=2, wins=1, losses=0, draws=1)
    assert rating.as_dict() == {
        "subject": "a",
        "elo": 1500.1235,
        "played": 2,
        "wins": 1,
        "losses": 0,
        "draws": 1,
        "win_rate": 0.75,
        "score_points": 1.5,
    }


def test_ratings_are_immutable_value_objects() -> None:
    rating = Rating("a", played=1, wins=1)
    advanced = rating.advanced(1.0, 1600.0)
    assert rating.elo == DEFAULT_INITIAL_RATING
    assert advanced is not rating
