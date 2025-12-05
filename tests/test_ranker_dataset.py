"""Synthetic ranking dataset: determinism, planted labels, validation, guard.

These tests run without torch on purpose - the dataset and its planted quality
function are pure NumPy/stdlib, so they are exercised in the fast suite.
"""

from __future__ import annotations

from typing import Any

import pytest

import hypoarena.ranker as ranker
from hypoarena.errors import ValidationError
from hypoarena.ranker import (
    LabeledExample,
    RankerDatasetConfig,
    planted_quality,
    synthetic_ranking_dataset,
)


def test_dataset_is_deterministic_for_a_seed() -> None:
    config = RankerDatasetConfig(n_examples=16, seed=7)
    assert synthetic_ranking_dataset(config) == synthetic_ranking_dataset(config)


def test_different_seeds_give_different_datasets() -> None:
    a = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=16, seed=1))
    b = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=16, seed=2))
    assert a != b


def test_dataset_size_and_score_bounds() -> None:
    data = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=32, seed=3))
    assert len(data) == 32
    assert all(0.0 <= example.score <= 1.0 for example in data)
    assert all(example.example_id for example in data)


def test_text_only_uses_known_tokens() -> None:
    config = RankerDatasetConfig(n_examples=8, seed=4)
    allowed = set(config.signal_tokens) | set(config.noise_tokens)
    for example in synthetic_ranking_dataset(config):
        assert set(example.text.split()) <= allowed


def test_planted_quality_is_a_weighted_fraction() -> None:
    config = RankerDatasetConfig()
    assert planted_quality((), config) == 0.0
    assert planted_quality(config.signal_tokens, config) == pytest.approx(1.0)
    assert planted_quality(("alpha",), config) > planted_quality(("delta",), config)


def test_planted_quality_ignores_noise_tokens() -> None:
    config = RankerDatasetConfig()
    assert planted_quality(("system", "cell"), config) == 0.0


def test_example_score_matches_its_planted_quality() -> None:
    config = RankerDatasetConfig(n_examples=16, seed=9)
    for example in synthetic_ranking_dataset(config):
        assert example.score == pytest.approx(planted_quality(example.signals, config))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_examples": 0},
        {"signal_tokens": ()},
        {"signal_weights": ()},
        {"signal_weights": (-0.1, 0.2, 0.3, 0.4)},
        {"signal_weights": (0.0, 0.0, 0.0, 0.0)},
        {"noise_tokens": ()},
        {"filler_min": 5, "filler_max": 2},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RankerDatasetConfig(**kwargs)


def test_labeled_example_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        LabeledExample(example_id="x", text="a", score=1.5, signals=())


def test_require_torch_reports_a_clear_error_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ranker, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="torch extra"):
        ranker.require_torch()


def test_torch_availability_flag_is_a_bool() -> None:
    assert isinstance(ranker.TORCH_AVAILABLE, bool)
