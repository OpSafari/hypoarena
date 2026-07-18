"""Ranker edge cases on the torch-free layer: degenerate datasets and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hypoarena.errors import ValidationError
from hypoarena.ranker import (
    RankerDatasetConfig,
    RankerFeaturizer,
    calibration_curve,
    planted_quality,
    spearman_correlation,
    synthetic_ranking_dataset,
)


def test_single_example_dataset_featurizes() -> None:
    data = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=1, seed=0))
    assert len(data) == 1
    ranker_data = RankerFeaturizer().fit_transform(data)
    assert ranker_data.features.shape[0] == 1
    assert ranker_data.scores.shape == (1,)


def test_planted_quality_bounds_and_unknown_tokens() -> None:
    config = RankerDatasetConfig()
    assert planted_quality((), config) == 0.0
    assert planted_quality(config.signal_tokens, config) == pytest.approx(1.0)
    assert planted_quality(("not-a-signal",), config) == 0.0


def test_featurizer_transform_before_fit_raises() -> None:
    featurizer = RankerFeaturizer()
    examples = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=2, seed=1))
    with pytest.raises(ValidationError):
        featurizer.transform(examples)


def test_calibration_single_bin_holds_everything() -> None:
    curve = calibration_curve([0.2, 0.5, 0.9], [0.1, 0.6, 1.0], bins=1)
    assert len(curve) == 1
    assert curve[0].count == 3


def test_spearman_is_bounded_for_random_inputs() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        rho = spearman_correlation(list(rng.random(10)), list(rng.random(10)))
        assert -1.0 <= rho <= 1.0
