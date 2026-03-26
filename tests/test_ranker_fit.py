"""End-to-end fit: the ranker learns the planted signal (slow, needs torch)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="optional torch extra not installed")

from hypoarena.ranker import (  # noqa: E402
    RankerData,
    RankerDatasetConfig,
    RankerFeaturizer,
    synthetic_ranking_dataset,
)
from hypoarena.ranker_torch import fit_ranker  # noqa: E402

pytestmark = [pytest.mark.model, pytest.mark.slow]


def make_data(n: int = 128, seed: int = 270106) -> RankerData:
    examples = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=n, seed=seed))
    return RankerFeaturizer().fit_transform(examples)


def test_fit_recovers_the_planted_order() -> None:
    result = fit_ranker(make_data(), seed=0)
    assert result.spearman > 0.8
    assert result.train.loss_decreased()
    assert result.predictions.shape == (128,)


def test_fit_predictions_stay_in_range() -> None:
    result = fit_ranker(make_data(n=64), seed=0)
    assert float(result.predictions.min()) >= 0.0
    assert float(result.predictions.max()) <= 1.0


def test_fit_is_reproducible_for_a_seed() -> None:
    data = make_data()
    first = fit_ranker(data, seed=3)
    second = fit_ranker(data, seed=3)
    assert first.spearman == second.spearman
    assert first.train.losses == second.train.losses
