"""RubricRanker construction and forward pass (requires the torch extra).

These are fast - no training - so they run in the default suite when torch is
installed and skip cleanly when it is not.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="optional torch extra not installed")

from hypoarena.errors import ValidationError  # noqa: E402
from hypoarena.ranker_torch import RubricRanker  # noqa: E402

pytestmark = pytest.mark.model


def test_forward_returns_one_score_per_row() -> None:
    model = RubricRanker(input_dim=5, hidden_dim=8)
    out = model(torch.zeros(3, 5))
    assert out.shape == (3, 1)


def test_forward_output_stays_within_the_unit_interval() -> None:
    model = RubricRanker(input_dim=4)
    out = model(torch.randn(64, 4) * 10)
    assert bool(torch.all(out >= 0.0)) and bool(torch.all(out <= 1.0))


def test_seeded_construction_is_deterministic() -> None:
    torch.manual_seed(0)
    first = RubricRanker(input_dim=4)
    torch.manual_seed(0)
    second = RubricRanker(input_dim=4)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)


def test_construction_rejects_bad_dimensions() -> None:
    with pytest.raises(ValidationError):
        RubricRanker(input_dim=0)
    with pytest.raises(ValidationError):
        RubricRanker(input_dim=4, hidden_dim=0)
