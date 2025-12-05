"""Measured recovery: the ranker separates planted quality (honest numbers).

Every figure here is measured on synthetic data with planted ground truth. It is
not a benchmark of any real model and says nothing about scientific truth; it
only shows the train/predict/calibrate mechanism recovers a signal we planted.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="optional torch extra not installed")

from hypoarena.ranker import (  # noqa: E402
    RankerData,
    RankerDatasetConfig,
    RankerFeaturizer,
    calibration_curve,
    synthetic_ranking_dataset,
)
from hypoarena.ranker_torch import fit_ranker  # noqa: E402

pytestmark = [pytest.mark.model, pytest.mark.slow]


def make_data(n: int = 128, seed: int = 270106) -> RankerData:
    examples = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=n, seed=seed))
    return RankerFeaturizer().fit_transform(examples)


def test_top_predicted_quintile_outranks_the_bottom() -> None:
    data = make_data()
    result = fit_ranker(data, seed=0)
    order = np.argsort(result.predictions)
    k = len(order) // 5
    bottom = float(data.scores[order[:k]].mean())
    top = float(data.scores[order[-k:]].mean())
    assert top > bottom + 0.2  # a real, measured separation on planted quality


def test_rank_agreement_is_strong_and_bounded() -> None:
    result = fit_ranker(make_data(), seed=0)
    assert 0.8 < result.spearman <= 1.0


def test_calibration_curve_covers_every_example() -> None:
    data = make_data()
    result = fit_ranker(data, seed=0)
    curve = calibration_curve(list(result.predictions), list(data.scores), bins=5)
    assert sum(entry.count for entry in curve) == len(data)
