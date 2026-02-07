"""Tournament execution: schedule, ratings, standings and validation."""

from __future__ import annotations

import pytest

from helpers import sample_claim
from hypoarena.errors import DuplicateIdError, UnknownReferenceError, ValidationError
from hypoarena.tournament import (
    PlantedJudge,
    Rating,
    Tournament,
    TournamentConfig,
    pair_count,
)

SUBJECTS = (
    "clm_0123456789ab",
    "clm_111111111111",
    "clm_222222222222",
    "clm_ffffffffffff",
)


def claims() -> list[object]:
    return [sample_claim(claim_id=subject) for subject in SUBJECTS]


def tournament(qualities: dict[str, float], **config: object) -> Tournament:
    return Tournament(
        PlantedJudge(qualities=qualities),
        TournamentConfig(**config),  # type: ignore[arg-type]
    )


def test_a_tournament_plays_every_pair_once_per_repeat() -> None:
    result = tournament({}).run(claims())
    assert len(result.matches) == pair_count(SUBJECTS)
    assert result.subjects == tuple(sorted(SUBJECTS))
    assert all(rating.played == 3 for rating in result.ratings)


def test_repeats_multiply_the_match_count() -> None:
    result = tournament({}, repeats=3).run(claims())
    assert len(result.matches) == 3 * pair_count(SUBJECTS)
    assert all(rating.played == 9 for rating in result.ratings)


def test_standings_are_ordered_by_rating_then_identifier() -> None:
    qualities = {
        SUBJECTS[0]: 0.9,
        SUBJECTS[1]: 0.6,
        SUBJECTS[2]: 0.3,
        SUBJECTS[3]: 0.0,
    }
    result = tournament(qualities).run(claims())
    ratings = [rating.elo for rating in result.ratings]
    assert ratings == sorted(ratings, reverse=True)
    assert result.ranking()[0] == SUBJECTS[0]
    assert [row["position"] for row in result.standings()] == [1, 2, 3, 4]


def test_wins_losses_and_draws_are_tallied() -> None:
    result = tournament({SUBJECTS[0]: 1.0, SUBJECTS[1]: 0.0}).run(
        [sample_claim(claim_id=SUBJECTS[0]), sample_claim(claim_id=SUBJECTS[1])]
    )
    best = result.rating(SUBJECTS[0])
    worst = result.rating(SUBJECTS[1])
    assert (best.wins, best.losses, best.played) == (1, 0, 1)
    assert (worst.wins, worst.losses, worst.played) == (0, 1, 1)
    assert best.score_points + worst.score_points == 1.0


def test_equal_qualities_produce_draws_and_unchanged_ratings() -> None:
    config = TournamentConfig()
    result = Tournament(PlantedJudge({}), config).run(claims())
    assert all(match.is_draw for match in result.matches)
    assert all(rating.elo == config.model.initial for rating in result.ratings)


def test_rating_lookup_reports_unknown_subjects() -> None:
    result = tournament({}).run(claims())
    assert isinstance(result.rating(SUBJECTS[0]), Rating)
    with pytest.raises(UnknownReferenceError):
        result.rating("clm_999999999999")


def test_degenerate_inputs_are_rejected() -> None:
    with pytest.raises(ValidationError, match="at least two distinct claims"):
        tournament({}).run([sample_claim()])
    with pytest.raises(ValidationError, match="at least two distinct claims"):
        tournament({}).run([])
    with pytest.raises(DuplicateIdError):
        tournament({}).run([sample_claim(), sample_claim()])
    with pytest.raises(ValidationError, match="repeats"):
        TournamentConfig(repeats=0)
    with pytest.raises(ValidationError, match="Judge protocol"):
        Tournament("not a judge")  # type: ignore[arg-type]


def test_matches_carry_their_round_and_index() -> None:
    result = tournament({}, repeats=2).run(claims())
    assert [match.match_index for match in result.matches] == list(
        range(len(result.matches))
    )
    assert {match.round_index for match in result.matches} == {0, 1}
    assert all(match.seed == result.config.seed for match in result.matches)
    assert all(match.judge == "planted" for match in result.matches)


def test_result_summary_is_serializable_and_stable() -> None:
    first = tournament({SUBJECTS[0]: 0.9}, seed=3).run(claims())
    second = tournament({SUBJECTS[0]: 0.9}, seed=3).run(claims())
    assert first.as_dict() == second.as_dict()
    assert first.signature() == second.signature()
    assert first.as_dict()["config_fingerprint"] == first.config.fingerprint()
