"""Verifier configuration: golden defaults, bounds and fingerprints."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.grounding import (
    DEFAULT_MIN_ENTITY_OVERLAP,
    DEFAULT_MIN_QUOTE_LENGTH,
    DEFAULT_NUMERIC_TOLERANCE,
    VerifierConfig,
)


def test_defaults_are_golden() -> None:
    config = VerifierConfig()
    assert config.min_entity_overlap == DEFAULT_MIN_ENTITY_OVERLAP == 0.34
    assert config.min_quote_length == DEFAULT_MIN_QUOTE_LENGTH == 12
    assert config.numeric_tolerance == DEFAULT_NUMERIC_TOLERANCE == 1e-9
    assert config.polarity_checks is True
    assert config.numeric_checks is True


def test_threshold_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError, match="min_entity_overlap"):
        VerifierConfig(min_entity_overlap=1.5)
    with pytest.raises(ValidationError, match="min_entity_overlap"):
        VerifierConfig(min_entity_overlap=-0.1)
    with pytest.raises(ValidationError, match="min_quote_length"):
        VerifierConfig(min_quote_length=0)
    with pytest.raises(ValidationError, match="numeric_tolerance"):
        VerifierConfig(numeric_tolerance=-1.0)


def test_extreme_but_valid_thresholds_are_accepted() -> None:
    assert VerifierConfig(min_entity_overlap=0.0).min_entity_overlap == 0.0
    assert VerifierConfig(min_entity_overlap=1.0).min_entity_overlap == 1.0
    assert VerifierConfig(min_quote_length=1).min_quote_length == 1


def test_fingerprint_is_stable_and_field_sensitive() -> None:
    assert VerifierConfig().fingerprint() == VerifierConfig().fingerprint()
    assert VerifierConfig(min_quote_length=20).fingerprint() != (
        VerifierConfig().fingerprint()
    )
    assert VerifierConfig(polarity_checks=False).fingerprint() != (
        VerifierConfig().fingerprint()
    )
    assert len(VerifierConfig().fingerprint()) == 16
