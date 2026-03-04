"""Contradiction policies and belief configuration."""

from __future__ import annotations

import pytest

from helpers import sample_evidence
from hypoarena.belief import (
    DEFAULT_DOWNWEIGHT_FACTOR,
    BeliefConfig,
    ContradictionPolicy,
    LikelihoodModel,
)
from hypoarena.errors import ValidationError
from hypoarena.schema import EvidencePolarity


def refute(strength: float = 1.0) -> object:
    return sample_evidence(polarity=EvidencePolarity.REFUTE, strength=strength)


def support(strength: float = 1.0) -> object:
    return sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=strength)


def test_policy_members_are_golden() -> None:
    assert [policy.value for policy in ContradictionPolicy] == [
        "ignore",
        "downweight",
        "discount",
        "reject",
    ]


def test_defaults_are_golden() -> None:
    config = BeliefConfig()
    assert config.prior == 0.5
    assert config.likelihood == LikelihoodModel()
    assert config.contradiction_policy is ContradictionPolicy.DISCOUNT
    assert config.downweight_factor == DEFAULT_DOWNWEIGHT_FACTOR == 0.5
    assert config.contradiction_threshold == 1


def test_ignore_neutralizes_refuting_evidence() -> None:
    config = BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE)
    assert config.ratio_for(refute(), refuting=1) == 1.0
    assert config.ratio_for(support(), refuting=1) == pytest.approx(3.0)


def test_downweight_softens_refuting_evidence() -> None:
    config = BeliefConfig(contradiction_policy=ContradictionPolicy.DOWNWEIGHT)
    assert config.ratio_for(refute(), refuting=1) == pytest.approx((1 / 3) ** 0.5)
    full = BeliefConfig(
        contradiction_policy=ContradictionPolicy.DOWNWEIGHT, downweight_factor=1.0
    )
    assert full.ratio_for(refute(), refuting=1) == pytest.approx(1 / 3)


def test_discount_and_reject_leave_individual_ratios_alone() -> None:
    for policy in (ContradictionPolicy.DISCOUNT, ContradictionPolicy.REJECT):
        config = BeliefConfig(contradiction_policy=policy)
        assert config.ratio_for(refute(), refuting=1) == pytest.approx(1 / 3)


def test_contested_uses_the_threshold() -> None:
    config = BeliefConfig(contradiction_threshold=2)
    assert config.contested(1) is False
    assert config.contested(2) is True
    assert BeliefConfig().contested(1) is True


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="prior"):
        BeliefConfig(prior=1.5)
    with pytest.raises(ValidationError, match="downweight_factor"):
        BeliefConfig(downweight_factor=0.0)
    with pytest.raises(ValidationError, match="downweight_factor"):
        BeliefConfig(downweight_factor=1.5)
    with pytest.raises(ValidationError, match="contradiction_threshold"):
        BeliefConfig(contradiction_threshold=0)


def test_the_fingerprint_covers_every_setting() -> None:
    baseline = BeliefConfig().fingerprint()
    assert baseline == BeliefConfig().fingerprint()
    assert BeliefConfig(prior=0.3).fingerprint() != baseline
    assert (
        BeliefConfig(contradiction_policy=ContradictionPolicy.REJECT).fingerprint()
        != baseline
    )
    assert BeliefConfig(
        likelihood=LikelihoodModel(support_ratio=4.0)
    ).fingerprint() != (baseline)
