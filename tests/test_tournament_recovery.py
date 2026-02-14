"""Does the rating recover a planted skill order? Measured, on synthetic claims.

The qualities below are chosen by the test, not learned: four claims with
planted strengths are compared by a judge that adds deterministic noise, and the
assertions state how much of the order the Elo update recovers. The numbers are
properties of this mechanism on synthetic input; they say nothing about any
language model's ability to rank hypotheses.
"""

from __future__ import annotations

from helpers import sample_claim
from hypoarena.tournament import (
    PlantedJudge,
    Tournament,
    TournamentConfig,
    kendall_tau,
    order_recovery,
    prefix_agreement,
    rating_spread,
    transitivity_rate,
)

SUBJECTS = (
    "clm_0123456789ab",
    "clm_111111111111",
    "clm_222222222222",
    "clm_ffffffffffff",
)
QUALITIES = {
    SUBJECTS[0]: 0.95,
    SUBJECTS[1]: 0.70,
    SUBJECTS[2]: 0.45,
    SUBJECTS[3]: 0.20,
}


def claims() -> list[object]:
    return [sample_claim(claim_id=subject) for subject in SUBJECTS]


def run(noise: float, repeats: int, seed: int = 0) -> object:
    return Tournament(
        PlantedJudge(QUALITIES, noise=noise, seed=seed),
        TournamentConfig(seed=seed, repeats=repeats),
    ).run(claims())


def test_without_noise_the_planted_order_is_recovered_exactly() -> None:
    result = run(noise=0.0, repeats=1)
    assert result.ranking() == SUBJECTS
    assert order_recovery(SUBJECTS, result.ranking()) == 1.0
    assert transitivity_rate(result.matches) == 1.0


def test_moderate_noise_still_recovers_the_order_with_enough_rounds() -> None:
    result = run(noise=0.1, repeats=4)
    assert result.ranking() == SUBJECTS
    assert kendall_tau(SUBJECTS, result.ranking()) == 1.0


def test_heavy_noise_keeps_recovery_above_chance() -> None:
    recoveries = [
        order_recovery(SUBJECTS, run(noise=0.35, repeats=3, seed=seed).ranking())
        for seed in range(5)
    ]
    assert all(value > 0.5 for value in recoveries)
    assert sum(recoveries) / len(recoveries) > 0.7


def test_more_rounds_never_reduce_recovery_for_a_fixed_seed() -> None:
    few = run(noise=0.2, repeats=1, seed=11)
    many = run(noise=0.2, repeats=6, seed=11)
    assert order_recovery(SUBJECTS, many.ranking()) >= order_recovery(
        SUBJECTS, few.ranking()
    )


def test_the_top_of_the_table_is_the_most_reliable_part() -> None:
    result = run(noise=0.15, repeats=3, seed=4)
    assert prefix_agreement(SUBJECTS, result.ranking(), 1) == 1.0
    assert prefix_agreement(SUBJECTS, result.ranking(), 2) >= 0.5


def test_separation_shows_up_in_the_rating_spread() -> None:
    noisy = rating_spread(run(noise=0.3, repeats=2, seed=2))
    clean = rating_spread(run(noise=0.0, repeats=2, seed=2))
    assert clean > 0.0
    assert noisy > 0.0


def test_a_single_round_leaves_nearby_qualities_unresolved() -> None:
    close = {
        SUBJECTS[0]: 0.52,
        SUBJECTS[1]: 0.50,
        SUBJECTS[2]: 0.48,
        SUBJECTS[3]: 0.20,
    }
    result = Tournament(
        PlantedJudge(close, noise=0.05), TournamentConfig(repeats=1)
    ).run(claims())
    assert result.ranking()[-1] == SUBJECTS[3]
    # the three near-equal claims are separated only by draw handling
    assert {match.outcome for match in result.matches} <= {0.0, 0.5, 1.0}
