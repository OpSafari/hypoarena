"""The training mechanism reduces loss across hyperparameter choices.

Marked ``model`` (needs torch) and ``slow`` (it actually trains). The point is
honest: for several widths and epoch budgets on the planted synthetic dataset the
loss really moves down and the history length matches the requested epochs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="optional torch extra not installed")

from hypoarena.ranker import (  # noqa: E402
    RankerData,
    RankerDatasetConfig,
    RankerFeaturizer,
    synthetic_ranking_dataset,
)
from hypoarena.ranker_torch import train_ranker  # noqa: E402

pytestmark = [pytest.mark.model, pytest.mark.slow]


def make_data(n: int = 96, seed: int = 5) -> RankerData:
    examples = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=n, seed=seed))
    return RankerFeaturizer().fit_transform(examples)


@pytest.mark.parametrize("hidden,epochs", [(8, 150), (16, 200), (32, 250)])
def test_loss_decreases_across_hyperparameters(hidden: int, epochs: int) -> None:
    result = train_ranker(make_data(), hidden_dim=hidden, epochs=epochs, seed=0)
    assert result.loss_decreased()
    assert len(result.losses) == epochs


def test_longer_training_reaches_a_low_loss() -> None:
    result = train_ranker(make_data(), epochs=400, seed=0)
    assert result.final_loss < 0.05
