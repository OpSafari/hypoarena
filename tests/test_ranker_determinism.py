"""Determinism and seed sensitivity of the trainable ranker (slow, torch)."""

from __future__ import annotations

import numpy as np
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


def test_same_seed_gives_identical_predictions() -> None:
    data = make_data()
    first = fit_ranker(data, seed=7)
    second = fit_ranker(data, seed=7)
    assert np.array_equal(first.predictions, second.predictions)
    assert first.train.losses == second.train.losses


def test_different_seed_changes_init_but_still_fits() -> None:
    data = make_data()
    first = fit_ranker(data, seed=0)
    second = fit_ranker(data, seed=1)
    assert not np.allclose(first.predictions, second.predictions)
    assert second.spearman > 0.8


def test_a_different_dataset_seed_changes_the_labels() -> None:
    first = make_data(seed=1)
    second = make_data(seed=2)
    assert not np.array_equal(first.scores, second.scores)
