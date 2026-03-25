"""Training the ranker: validation, real loss decrease and determinism.

The hyperparameter checks are fast and run in the default suite; the two tests
that actually train are marked ``slow`` so per-commit runs stay quick.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="optional torch extra not installed")

from hypoarena.errors import ValidationError  # noqa: E402
from hypoarena.ranker import (  # noqa: E402
    RankerData,
    RankerDatasetConfig,
    RankerFeaturizer,
    synthetic_ranking_dataset,
)
from hypoarena.ranker_torch import train_ranker  # noqa: E402

pytestmark = pytest.mark.model


def make_data(n: int = 128, seed: int = 270106) -> RankerData:
    examples = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=n, seed=seed))
    return RankerFeaturizer().fit_transform(examples)


def test_training_rejects_bad_hyperparameters() -> None:
    data = make_data(n=8)
    with pytest.raises(ValidationError):
        train_ranker(data, epochs=0)
    with pytest.raises(ValidationError):
        train_ranker(data, lr=0.0)


def test_training_rejects_an_empty_dataset() -> None:
    empty = RankerData(
        features=np.zeros((0, 3)), scores=np.zeros(0), vocabulary=("a", "b", "c")
    )
    with pytest.raises(ValidationError):
        train_ranker(empty)


@pytest.mark.slow
def test_loss_really_decreases_on_synthetic_data() -> None:
    result = train_ranker(make_data(), seed=0)
    assert len(result.losses) == 300
    assert result.loss_decreased()
    assert result.final_loss < result.first_loss * 0.5


@pytest.mark.slow
def test_training_is_deterministic_for_a_seed() -> None:
    data = make_data()
    first = train_ranker(data, seed=0)
    second = train_ranker(data, seed=0)
    assert first.losses == second.losses
