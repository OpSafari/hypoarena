"""Rank-correlation and reliability-curve metrics, both torch-free."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.ranker import CalibrationBin, calibration_curve, spearman_correlation


def test_spearman_of_identical_order_is_one() -> None:
    assert spearman_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_spearman_of_reversed_order_is_minus_one() -> None:
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_ignores_monotonic_rescaling() -> None:
    assert spearman_correlation([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)


def test_spearman_of_a_constant_input_is_zero_not_nan() -> None:
    assert spearman_correlation([1, 1, 1], [1, 2, 3]) == 0.0


def test_spearman_handles_ties_with_average_ranks() -> None:
    rho = spearman_correlation([1, 1, 2], [10, 20, 30])
    assert 0.8 < rho <= 1.0


def test_spearman_rejects_mismatched_or_short_inputs() -> None:
    with pytest.raises(ValidationError):
        spearman_correlation([1, 2], [1, 2, 3])
    with pytest.raises(ValidationError):
        spearman_correlation([1], [1])


def test_calibration_curve_has_exactly_bins_entries() -> None:
    curve = calibration_curve([0.1, 0.9], [0.0, 1.0], bins=5)
    assert len(curve) == 5
    assert sum(entry.count for entry in curve) == 2


def test_calibration_known_two_bin_case() -> None:
    curve = calibration_curve([0.1, 0.9], [0.0, 1.0], bins=2)
    assert curve[0].count == 1
    assert curve[0].mean_predicted == pytest.approx(0.1)
    assert curve[0].mean_actual == pytest.approx(0.0)
    assert curve[1].count == 1
    assert curve[1].mean_predicted == pytest.approx(0.9)
    assert curve[1].mean_actual == pytest.approx(1.0)


def test_perfectly_calibrated_predictions_match_actuals_per_bin() -> None:
    scores = [0.05, 0.15, 0.55, 0.95]
    curve = calibration_curve(scores, scores, bins=2)
    for entry in curve:
        if entry.count:
            assert entry.mean_predicted == pytest.approx(entry.mean_actual)


def test_empty_bins_report_zero_count_and_means() -> None:
    curve = calibration_curve([0.9, 0.95], [1.0, 1.0], bins=4)
    assert curve[0].count == 0
    assert curve[0].mean_predicted == 0.0
    assert curve[0].mean_actual == 0.0


def test_prediction_of_exactly_one_lands_in_the_last_bin() -> None:
    curve = calibration_curve([1.0], [1.0], bins=2)
    assert curve[-1].count == 1
    assert curve[0].count == 0


def test_calibration_validation() -> None:
    with pytest.raises(ValidationError):
        calibration_curve([0.1], [0.1, 0.2])
    with pytest.raises(ValidationError):
        calibration_curve([0.1], [0.1], bins=0)
    with pytest.raises(ValidationError):
        calibration_curve([1.5], [0.5])
    with pytest.raises(ValidationError):
        calibration_curve([0.5], [-0.1])


def test_calibration_bin_rejects_bad_edges() -> None:
    with pytest.raises(ValidationError):
        CalibrationBin(
            index=0, low=0.8, high=0.2, count=0, mean_predicted=0, mean_actual=0
        )
