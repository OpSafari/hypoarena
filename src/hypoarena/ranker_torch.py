"""The trainable half of the ranker: a small MLP over TF-IDF features.

Importing this module requires the optional ``torch`` extra; the NumPy-only
layers live in :mod:`hypoarena.ranker`. Training is fully seeded and
deterministic on CPU and uses no pretrained weights or downloads.

Honest scope: this model learns to predict *planted synthetic-quality* scores
from lexical features. Demonstrating that the train/eval/calibrate mechanism
works is the point; it is not a claim about scientific truth or about any real
model's ability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from hypoarena.errors import ValidationError
from hypoarena.ranker import RankerData, spearman_correlation


class RubricRanker(nn.Module):
    """A small feed-forward model mapping TF-IDF features to a score in [0, 1].

    Deliberately tiny (one hidden layer, sigmoid output): the goal is to show the
    ranking mechanism works, not to fit a large model.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValidationError("input_dim must be >= 1", input_dim=input_dim)
        if hidden_dim < 1:
            raise ValidationError("hidden_dim must be >= 1", hidden_dim=hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a column of predicted scores, each within ``[0, 1]``."""
        return self.net(features)


@dataclass(frozen=True)
class TrainResult:
    """The trained model plus its loss history and hyperparameters.

    The full per-epoch loss history is kept so a test can assert the loss really
    moved down over training instead of trusting a single final number.
    """

    model: RubricRanker
    losses: tuple[float, ...]
    epochs: int
    seed: int

    @property
    def first_loss(self) -> float:
        """The loss at epoch 0, before any meaningful update."""
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        """The loss after the last update."""
        return self.losses[-1]

    def loss_decreased(self) -> bool:
        """True when training moved the loss down, as it must."""
        return self.final_loss < self.first_loss


def train_ranker(
    data: RankerData,
    *,
    hidden_dim: int = 16,
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 0,
) -> TrainResult:
    """Train the ranker with full-batch Adam; the seed makes it reproducible.

    Full-batch gradient descent on a small synthetic problem is deterministic on
    CPU once the seed fixes the initial weights, so two calls with the same seed
    produce identical loss histories.
    """
    if epochs < 1:
        raise ValidationError("epochs must be >= 1", epochs=epochs)
    if lr <= 0:
        raise ValidationError("learning rate must be > 0", lr=lr)
    if len(data) == 0:
        raise ValidationError("cannot train on an empty dataset")
    torch.manual_seed(seed)
    input_dim = int(data.features.shape[1])
    model = RubricRanker(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    features = torch.tensor(data.features, dtype=torch.float32)
    targets = torch.tensor(data.scores, dtype=torch.float32).unsqueeze(1)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(features), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return TrainResult(model=model, losses=tuple(losses), epochs=epochs, seed=seed)


def predict_scores(model: RubricRanker, features: np.ndarray) -> np.ndarray:
    """Return predicted scores in ``[0, 1]`` for a NumPy feature matrix.

    Runs in eval mode under ``no_grad`` so inference is cheap and free of any
    autograd bookkeeping; the result is a flat 1-D array aligned with the rows.
    """
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(np.asarray(features, dtype=np.float32))
        predicted = model(tensor).detach().cpu().numpy().ravel()
    return predicted


@dataclass(frozen=True)
class FitResult:
    """End-to-end outcome: the training result, predictions and rank agreement."""

    train: TrainResult
    predictions: np.ndarray
    spearman: float


def fit_ranker(
    data: RankerData,
    *,
    hidden_dim: int = 16,
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 0,
) -> FitResult:
    """Train, predict on the same data and report rank agreement.

    In-sample on purpose: this demonstrates the mechanism fits planted signal,
    not that it generalises to held-out science. The returned Spearman rho is a
    measured number on synthetic data, never a benchmark claim.
    """
    result = train_ranker(data, hidden_dim=hidden_dim, epochs=epochs, lr=lr, seed=seed)
    predictions = predict_scores(result.model, data.features)
    rho = spearman_correlation(list(predictions), list(data.scores))
    return FitResult(train=result, predictions=predictions, spearman=rho)
