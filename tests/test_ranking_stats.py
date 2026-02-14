"""Order statistics: tau, prefix agreement, transitivity and convergence."""

from __future__ import annotations

import pytest

from helpers import sample_claim
from hypoarena.errors import ValidationError
from hypoarena.tournament import (
    EloModel,
    MatchResult,
    PlantedJudge,
    RubricScore,
    Tournament,
    TournamentConfig,
    convergence_ratio,
    kendall_tau,
    order_recovery,
    prefix_agreement,
    rating_spread,
    replay_ratings,
    step_sizes,
    transitivity_rate,
)

A = "clm_0123456789ab"
B = "clm_111111111111"
C = "clm_222222222222"


def win(left: str, right: str) -> MatchResult:
    high = RubricScore(0.9, 0.9, 0.9, 0.9)
    low = RubricScore(0.1, 0.1, 0.1, 0.1)
    return MatchResult(left, right, high, low, 0.9, 0.1, 1.0, "planted")


def draw(left: str, right: str) -> MatchResult:
    level = RubricScore(0.5, 0.5, 0.5, 0.5)
    return MatchResult(left, right, level, level, 0.5, 0.5, 0.5, "planted")


def test_kendall_tau_spans_the_documented_range() -> None:
    order = (A, B, C)
    assert kendall_tau(order, order) == 1.0
    assert kendall_tau(order, tuple(reversed(order))) == -1.0
    assert order_recovery(order, order) == 1.0
    assert order_recovery(order, tuple(reversed(order))) == 0.0


def test_kendall_tau_counts_a_single_swap() -> None:
    assert kendall_tau((A, B, C), (A, C, B)) == pytest.approx(1 / 3)


def test_kendall_tau_requires_the_same_subject_set() -> None:
    with pytest.raises(ValidationError, match="same subjects"):
        kendall_tau((A, B), (A, C))


def test_prefix_agreement_measures_top_k_overlap() -> None:
    expected = (A, B, C)
    assert prefix_agreement(expected, expected, 2) == 1.0
    assert prefix_agreement(expected, (C, B, A), 2) == pytest.approx(1 / 2)
    # top-3 over three subjects is the whole set, so it always agrees
    assert prefix_agreement(expected, (C, B, A), 3) == 1.0
    fourth = "clm_333333333333"
    longer = (A, B, C, fourth)
    assert prefix_agreement(longer, (fourth, C, B, A), 3) == pytest.approx(2 / 3)
    with pytest.raises(ValidationError, match="top"):
        prefix_agreement(expected, expected, 0)


def test_transitivity_is_one_for_a_chain_and_zero_for_a_cycle() -> None:
    chain = [win(A, B), win(B, C), win(A, C)]
    assert transitivity_rate(chain) == 1.0
    cycle = [win(A, B), win(B, C), win(C, A)]
    assert transitivity_rate(cycle) == 0.0


def test_transitivity_ignores_draws_and_empty_trails() -> None:
    assert transitivity_rate([]) == 1.0
    assert transitivity_rate([draw(A, B), draw(B, C)]) == 1.0
    partial = [win(A, B), win(B, C), draw(A, C)]
    assert partial and transitivity_rate(partial) == 1.0


def test_replaying_the_audit_trail_reproduces_the_standings() -> None:
    claims = [sample_claim(claim_id=subject) for subject in (A, B, C)]
    result = Tournament(
        PlantedJudge({A: 0.9, B: 0.5, C: 0.1}), TournamentConfig(seed=5, repeats=2)
    ).run(claims)
    assert replay_ratings(result.matches, result.config.model) == result.ratings


def test_step_sizes_shrink_as_the_tournament_settles() -> None:
    claims = [sample_claim(claim_id=subject) for subject in (A, B, C)]
    result = Tournament(
        PlantedJudge({A: 0.9, B: 0.5, C: 0.1}), TournamentConfig(repeats=4)
    ).run(claims)
    sizes = step_sizes(result.matches, result.config.model)
    assert len(sizes) == len(result.matches)
    assert convergence_ratio(result.matches, result.config.model) < 1.0


def test_convergence_needs_enough_matches() -> None:
    with pytest.raises(ValidationError, match="at least four"):
        convergence_ratio([win(A, B), win(B, C)])


def test_rating_spread_measures_the_standings_gap() -> None:
    claims = [sample_claim(claim_id=subject) for subject in (A, B, C)]
    result = Tournament(PlantedJudge({A: 1.0, B: 0.5, C: 0.0})).run(claims)
    assert rating_spread(result) == pytest.approx(
        result.ratings[0].elo - result.ratings[-1].elo
    )
    assert rating_spread(result) > 0.0
    level = Tournament(PlantedJudge({})).run(claims)
    assert rating_spread(level) == 0.0


def test_a_decisive_judge_recovers_the_planted_order() -> None:
    claims = [sample_claim(claim_id=subject) for subject in (A, B, C)]
    result = Tournament(
        PlantedJudge({A: 0.9, B: 0.5, C: 0.1}), TournamentConfig(repeats=2)
    ).run(claims)
    assert result.ranking() == (A, B, C)
    assert order_recovery((A, B, C), result.ranking()) == 1.0
    assert EloModel().initial == 1500.0
