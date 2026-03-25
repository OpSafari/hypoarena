"""Inference: predict_scores shape, range and determinism (no training)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional torch extra not installed")

from hypoarena.ranker import (  # noqa: E402
    RankerData,
    RankerDatasetConfig,
    RankerFeaturizer,
    synthetic_ranking_dataset,
)
from hypoarena.ranker_torch import RubricRanker, predict_scores  # noqa: E402

pytestmark = pytest.mark.model


def make_data(n: int = 16, seed: int = 270106) -> RankerData:
    examples = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=n, seed=seed))
    return RankerFeaturizer().fit_transform(examples)


def test_predict_scores_shape_and_range() -> None:
    data = make_data(n=16)
    torch.manual_seed(0)
    model = RubricRanker(input_dim=data.features.shape[1])
    preds = predict_scores(model, data.features)
    assert preds.shape == (16,)
    assert bool(np.all(preds >= 0.0)) and bool(np.all(preds <= 1.0))


def test_predict_scores_is_deterministic() -> None:
    data = make_data(n=8)
    torch.manual_seed(0)
    model = RubricRanker(input_dim=data.features.shape[1])
    first = predict_scores(model, data.features)
    second = predict_scores(model, data.features)
    assert np.array_equal(first, second)


def test_predict_scores_accepts_a_single_row() -> None:
    data = make_data(n=4)
    torch.manual_seed(1)
    model = RubricRanker(input_dim=data.features.shape[1])
    assert predict_scores(model, data.features[:1]).shape == (1,)
