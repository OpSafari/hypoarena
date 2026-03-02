"""Likelihood ratios for graded evidence."""

from __future__ import annotations

import pytest

from helpers import sample_evidence
from hypoarena.belief import (
    DEFAULT_REFUTE_RATIO,
    DEFAULT_SUPPORT_RATIO,
    LikelihoodModel,
)
from hypoarena.errors import ValidationError
from hypoarena.schema import EvidencePolarity


def test_defaults_are_golden() -> None:
    model = LikelihoodModel()
    assert model.support_ratio == DEFAULT_SUPPORT_RATIO == 3.0
    assert model.refute_ratio == DEFAULT_REFUTE_RATIO == 3.0
    assert model.neutral_ratio == 1.0


def test_full_strength_support_multiplies_by_the_ratio() -> None:
    model = LikelihoodModel()
    evidence = sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=1.0)
    assert model.ratio_for(evidence) == pytest.approx(3.0)


def test_strength_interpolates_exponentially() -> None:
    model = LikelihoodModel()
    half = sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=0.5)
    assert model.ratio_for(half) == pytest.approx(3.0**0.5)
    quarter = sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=0.25)
    assert model.ratio_for(quarter) == pytest.approx(3.0**0.25)


def test_refuting_evidence_divides_the_odds() -> None:
    model = LikelihoodModel()
    full = sample_evidence(polarity=EvidencePolarity.REFUTE, strength=1.0)
    assert model.ratio_for(full) == pytest.approx(1 / 3)
    assert full.weighted_polarity == pytest.approx(-1.0)
    weak = sample_evidence(polarity=EvidencePolarity.REFUTE, strength=0.5)
    assert model.ratio_for(weak) == pytest.approx(3.0**-0.5)


def test_neutral_evidence_has_no_effect() -> None:
    model = LikelihoodModel()
    evidence = sample_evidence(polarity=EvidencePolarity.NEUTRAL, strength=0.0)
    assert model.ratio_for(evidence) == 1.0
    custom = LikelihoodModel(neutral_ratio=1.5)
    assert custom.ratio_for(evidence) == 1.5


def test_asymmetric_models_are_supported() -> None:
    model = LikelihoodModel(support_ratio=2.0, refute_ratio=8.0)
    support = sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=1.0)
    refute = sample_evidence(polarity=EvidencePolarity.REFUTE, strength=1.0)
    assert model.ratio_for(support) == pytest.approx(2.0)
    assert model.ratio_for(refute) == pytest.approx(1 / 8)


def test_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ValidationError, match="support_ratio"):
        LikelihoodModel(support_ratio=0.5)
    with pytest.raises(ValidationError, match="refute_ratio"):
        LikelihoodModel(refute_ratio=0.1)
    with pytest.raises(ValidationError, match="neutral_ratio"):
        LikelihoodModel(neutral_ratio=0.0)


def test_the_fingerprint_tracks_the_settings() -> None:
    assert LikelihoodModel().fingerprint() == LikelihoodModel().fingerprint()
    assert LikelihoodModel(support_ratio=4.0).fingerprint() != (
        LikelihoodModel().fingerprint()
    )
