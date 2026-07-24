"""Tournament statistics stay strong even at the maximum judge noise.

At the loudest deterministic noise the judge allows (0.4) the rating still
recovers most of the planted order and stays transitive. The thresholds sit below
the measured values, so a real regression in the update rule trips the test
rather than passing trivially.
"""

from __future__ import annotations

from helpers import sample_claim
from hypoarena.tournament import (
    PlantedJudge,
    Tournament,
    TournamentConfig,
    kendall_tau,
    order_recovery,
    transitivity_rate,
)

MAX_NOISE = 0.4
SUBJECTS = tuple(f"clm_{index:012x}" for index in range(5))
QUALITIES = dict(zip(SUBJECTS, (0.9, 0.7, 0.5, 0.3, 0.1), strict=True))
PLANTED = tuple(sorted(QUALITIES, key=lambda subject: -QUALITIES[subject]))


def claims() -> list[object]:
    return [sample_claim(claim_id=subject) for subject in SUBJECTS]


def run(seed: int, repeats: int = 3) -> object:
    judge = PlantedJudge(QUALITIES, noise=MAX_NOISE, seed=seed)
    config = TournamentConfig(seed=seed, repeats=repeats)
    return Tournament(judge, config).run(claims())


def test_transitivity_stays_high_at_max_noise() -> None:
    for seed in range(6):
        assert transitivity_rate(run(seed).matches) >= 0.9


def test_recovery_stays_high_at_max_noise() -> None:
    recoveries = [order_recovery(PLANTED, run(seed).ranking()) for seed in range(6)]
    assert all(value >= 0.8 for value in recoveries)
    assert sum(recoveries) / len(recoveries) > 0.9


def test_kendall_tau_stays_positive_at_max_noise() -> None:
    for seed in range(6):
        assert kendall_tau(PLANTED, run(seed).ranking()) >= 0.6
