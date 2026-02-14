"""Seeded sweeps over tournament configurations.

The invariants that must hold for every seed and schedule: the audit trail is
complete and self-consistent, replaying it reproduces the standings, tallies add
up, ratings stay finite, and a rerun of the same configuration is identical.
"""

from __future__ import annotations

import math

import pytest

from helpers import sample_claim
from hypoarena.tournament import (
    EloModel,
    PlantedJudge,
    Tournament,
    TournamentConfig,
    pair_count,
    replay_ratings,
    transitivity_rate,
)

SUBJECTS = tuple(f"clm_{index:012x}" for index in range(1, 7))
QUALITIES = {
    subject: round(0.15 + 0.14 * index, 2) for index, subject in enumerate(SUBJECTS)
}
SEEDS = (0, 7, 270106)
REPEATS = (1, 2, 4)


def claims() -> list[object]:
    return [sample_claim(claim_id=subject) for subject in SUBJECTS]


def result(seed: int, repeats: int, noise: float = 0.1) -> object:
    return Tournament(
        PlantedJudge(QUALITIES, noise=noise, seed=seed),
        TournamentConfig(seed=seed, repeats=repeats),
    ).run(claims())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("repeats", REPEATS)
def test_the_schedule_is_complete(seed: int, repeats: int) -> None:
    produced = result(seed, repeats)
    assert len(produced.matches) == repeats * pair_count(SUBJECTS)
    pairs = [match.pair() for match in produced.matches]
    assert len(set(pairs)) == pair_count(SUBJECTS)
    assert all(pairs.count(pair) == repeats for pair in set(pairs))


@pytest.mark.parametrize("seed", SEEDS)
def test_replays_reproduce_the_standings(seed: int) -> None:
    produced = result(seed, 3)
    assert replay_ratings(produced.matches, produced.config.model) == produced.ratings


@pytest.mark.parametrize("seed", SEEDS)
def test_tallies_and_ratings_stay_consistent(seed: int) -> None:
    produced = result(seed, 2)
    for rating in produced.ratings:
        assert rating.played == rating.wins + rating.losses + rating.draws
        assert math.isfinite(rating.elo)
        assert rating.played == 2 * (len(SUBJECTS) - 1)
    total_points = sum(rating.score_points for rating in produced.ratings)
    decisive = sum(1 for match in produced.matches if not match.is_draw)
    draws = len(produced.matches) - decisive
    assert total_points == pytest.approx(decisive + 0.5 * draws)


@pytest.mark.parametrize("seed", SEEDS)
def test_reruns_are_identical(seed: int) -> None:
    assert result(seed, 2).signature() == result(seed, 2).signature()
    assert result(seed, 2).as_dict() == result(seed, 2).as_dict()


def test_different_seeds_can_reorder_matches_but_not_coverage() -> None:
    first = result(1, 2)
    second = result(2, 2)
    assert [match.pair() for match in first.matches] != [
        match.pair() for match in second.matches
    ]
    assert sorted(match.pair() for match in first.matches) == sorted(
        match.pair() for match in second.matches
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_transitivity_is_high_for_a_seeded_quality_ladder(seed: int) -> None:
    produced = result(seed, 4, noise=0.05)
    assert transitivity_rate(produced.matches) >= 0.8


def test_a_stronger_k_widens_the_spread() -> None:
    claims_list = claims()
    judge = PlantedJudge(QUALITIES)
    calm = Tournament(judge, TournamentConfig(model=EloModel(k_factor=8.0))).run(
        claims_list
    )
    aggressive = Tournament(judge, TournamentConfig(model=EloModel(k_factor=64.0))).run(
        claims_list
    )
    assert aggressive.ratings[0].elo - aggressive.ratings[-1].elo > (
        calm.ratings[0].elo - calm.ratings[-1].elo
    )
