"""Elo convergence: more judged rounds stabilise the recovered order.

Qualities are planted by the test and the judge adds bounded deterministic
noise. These assert the rating mechanism converges toward the planted order and
never explodes - a property of the update rule on synthetic input, not a claim
about any real model's ability to rank hypotheses.
"""

from __future__ import annotations

from helpers import sample_claim
from hypoarena.tournament import (
    PlantedJudge,
    Tournament,
    TournamentConfig,
    order_recovery,
    rating_spread,
    transitivity_rate,
)

SUBJECTS = tuple(f"clm_{index:012x}" for index in range(6))
QUALITIES = dict(zip(SUBJECTS, (0.95, 0.85, 0.70, 0.50, 0.35, 0.20), strict=True))
PLANTED_ORDER = tuple(sorted(QUALITIES, key=lambda subject: -QUALITIES[subject]))


def claims() -> list[object]:
    return [sample_claim(claim_id=subject) for subject in SUBJECTS]


def run(repeats: int, seed: int = 0, noise: float = 0.1) -> object:
    judge = PlantedJudge(QUALITIES, noise=noise, seed=seed)
    return Tournament(judge, TournamentConfig(seed=seed, repeats=repeats)).run(claims())


def test_the_top_rated_claim_is_the_planted_best_across_seeds() -> None:
    best = PLANTED_ORDER[0]
    for seed in range(6):
        assert run(repeats=4, seed=seed).ranking()[0] == best


def test_recovery_never_collapses_as_rounds_grow() -> None:
    for repeats in (1, 2, 4, 8):
        recovery = order_recovery(PLANTED_ORDER, run(repeats=repeats).ranking())
        assert recovery >= 0.6


def test_more_rounds_reach_high_recovery() -> None:
    result = run(repeats=8)
    assert order_recovery(PLANTED_ORDER, result.ranking()) >= 0.9


def test_rating_spread_stays_bounded() -> None:
    spread = rating_spread(run(repeats=6))
    assert 0.0 <= spread < 1000.0


def test_transitivity_is_perfect_without_noise() -> None:
    assert transitivity_rate(run(repeats=2, noise=0.0).matches) == 1.0
