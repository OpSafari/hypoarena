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

import torch
from torch import nn

from hypoarena.errors import ValidationError


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
