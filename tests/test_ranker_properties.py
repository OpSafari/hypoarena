"""Metric properties on the torch-free ranker layer.

Spearman and the calibration curve are the honest yardsticks the ranker reports,
so their mathematical properties must hold for arbitrary bounded inputs, not just
the happy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from hypoarena.ranker import calibration_curve, spearman_correlation


def test_spearman_of_a_monotonic_transform_is_one() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [value * 3 + 1 for value in x]
    assert spearman_correlation(x, y) == pytest.approx(1.0)


def test_spearman_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    a = list(rng.random(12))
    b = list(rng.random(12))
    assert spearman_correlation(a, b) == pytest.approx(spearman_correlation(b, a))


def test_calibration_counts_always_sum_to_n() -> None:
    rng = np.random.default_rng(2)
    for _ in range(10):
        preds = list(rng.random(20))
        actual = list(rng.random(20))
        curve = calibration_curve(preds, actual, bins=5)
        assert sum(entry.count for entry in curve) == 20


def test_calibration_of_perfect_predictions_is_diagonal() -> None:
    scores = list(np.linspace(0.02, 0.98, 25))
    curve = calibration_curve(scores, scores, bins=5)
    for entry in curve:
        if entry.count:
            assert entry.mean_predicted == pytest.approx(entry.mean_actual, abs=1e-9)


def test_calibration_rejects_out_of_range_predictions() -> None:
    from hypoarena.errors import ValidationError

    with pytest.raises(ValidationError):
        calibration_curve([1.5, 0.2], [0.1, 0.2], bins=3)
