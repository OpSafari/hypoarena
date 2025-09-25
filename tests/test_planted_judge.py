"""The planted-quality judge: determinism, noise bounds and ordering."""

from __future__ import annotations

import pytest

from helpers import CLAIM_ID, OTHER_CLAIM_ID, sample_claim
from hypoarena.errors import ValidationError
from hypoarena.tournament import (
    DEFAULT_UNKNOWN_QUALITY,
    Judge,
    PlantedJudge,
    RubricWeights,
    weighted_total,
)


def judge(noise: float = 0.0, seed: int = 0) -> PlantedJudge:
    return PlantedJudge(
        qualities={CLAIM_ID: 0.9, OTHER_CLAIM_ID: 0.2}, noise=noise, seed=seed
    )


def test_the_judge_satisfies_the_protocol() -> None:
    assert isinstance(judge(), Judge)
    assert judge().name == "planted"


def test_scores_are_deterministic_for_a_seed_and_pairing() -> None:
    strong = sample_claim()
    weak = sample_claim(claim_id=OTHER_CLAIM_ID)
    first = judge(noise=0.2).score(strong, opponent=weak)
    second = judge(noise=0.2).score(strong, opponent=weak)
    assert first == second
    assert judge(seed=1, noise=0.2).score(strong, opponent=weak) != first


def test_zero_noise_reproduces_the_planted_quality_exactly() -> None:
    score = judge().score(sample_claim())
    assert score.as_tuple() == (0.9, 0.9, 0.9, 0.9)
    assert weighted_total(score, RubricWeights()) == pytest.approx(0.9)


def test_noise_stays_inside_its_bound() -> None:
    noise = 0.3
    claim = sample_claim()
    for seed in range(20):
        score = PlantedJudge({CLAIM_ID: 0.5}, noise=noise, seed=seed).score(claim)
        for value in score.as_tuple():
            assert 0.5 - noise - 1e-9 <= value <= 0.5 + noise + 1e-9


def test_stronger_claims_outscore_weaker_ones_on_average() -> None:
    strong = sample_claim()
    weak = sample_claim(claim_id=OTHER_CLAIM_ID)
    noisy = judge(noise=0.2)
    strong_total = (
        sum(noisy.score(strong, opponent=weak).mean() for _ in range(10)) / 10
    )
    weak_total = sum(noisy.score(weak, opponent=strong).mean() for _ in range(10)) / 10
    assert strong_total > weak_total


def test_unknown_claims_get_the_default_quality() -> None:
    score = judge().score(sample_claim(claim_id="clm_222222222222"))
    assert score.as_tuple() == (DEFAULT_UNKNOWN_QUALITY,) * 4


def test_out_of_range_settings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="noise"):
        PlantedJudge({}, noise=0.9)
    with pytest.raises(ValidationError, match="planted quality"):
        PlantedJudge({CLAIM_ID: 1.5})
