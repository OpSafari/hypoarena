"""Batch accumulation: composition, policies and hand-checkable numbers."""

from __future__ import annotations

import pytest

from helpers import sample_evidence
from hypoarena.belief import (
    MIN_PROBABILITY,
    BeliefConfig,
    ContradictionPolicy,
    LikelihoodModel,
    accumulate,
)
from hypoarena.schema import EvidencePolarity

CLAIM = "clm_0123456789ab"


def support(strength: float = 1.0) -> object:
    return sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=strength)


def refute(strength: float = 1.0) -> object:
    return sample_evidence(polarity=EvidencePolarity.REFUTE, strength=strength)


def neutral() -> object:
    return sample_evidence(polarity=EvidencePolarity.NEUTRAL, strength=0.0)


def test_no_evidence_leaves_the_prior_alone() -> None:
    state = accumulate(CLAIM, [])
    assert state.posterior == 0.5
    assert state.likelihood_ratio == 1.0
    assert state.updates == 0


def test_supporting_evidence_composes_multiplicatively() -> None:
    state = accumulate(CLAIM, [support(), support()])
    assert state.likelihood_ratio == pytest.approx(9.0)
    assert state.posterior == pytest.approx(0.9)
    assert state.supporting == 2


def test_partial_strength_evidence_composes_by_exponent() -> None:
    state = accumulate(CLAIM, [support(0.5), support(0.5)])
    assert state.likelihood_ratio == pytest.approx(3.0)
    assert state.posterior == pytest.approx(0.75)


def test_ignoring_contradictions_leaves_support_untouched() -> None:
    state = accumulate(
        CLAIM,
        [support(), refute()],
        BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE),
    )
    assert state.likelihood_ratio == pytest.approx(3.0)
    assert state.posterior == pytest.approx(0.75)


def test_mixed_evidence_multiplies_both_ways() -> None:
    state = accumulate(
        CLAIM,
        [support(), refute()],
        BeliefConfig(
            contradiction_policy=ContradictionPolicy.REJECT,
            contradiction_threshold=2,
        ),
    )
    assert state.likelihood_ratio == pytest.approx(1.0)
    assert state.posterior == pytest.approx(0.5)


def test_neutral_evidence_is_counted_but_has_no_effect() -> None:
    state = accumulate(CLAIM, [support(), neutral(), neutral()])
    assert state.neutral == 2
    assert state.likelihood_ratio == pytest.approx(3.0)


def test_the_discount_policy_pulls_a_contested_claim_toward_the_prior() -> None:
    contested = accumulate(CLAIM, [support(), support(), refute()])
    undiscounted = accumulate(
        CLAIM,
        [support(), support(), refute()],
        BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE),
    )
    assert contested.posterior < undiscounted.posterior
    assert contested.posterior > 0.5


def test_the_reject_policy_floors_a_contested_claim() -> None:
    state = accumulate(
        CLAIM,
        [support(), refute()],
        BeliefConfig(contradiction_policy=ContradictionPolicy.REJECT),
    )
    assert state.posterior == MIN_PROBABILITY
    assert state.refuting == 1


def test_the_reject_policy_respects_its_threshold() -> None:
    config = BeliefConfig(
        contradiction_policy=ContradictionPolicy.REJECT, contradiction_threshold=2
    )
    single = accumulate(CLAIM, [support(), refute()], config)
    assert single.refuting == 1
    assert single.posterior > MIN_PROBABILITY
    double = accumulate(CLAIM, [support(), refute(), refute()], config)
    assert double.refuting == 2
    assert double.posterior == MIN_PROBABILITY


def test_the_prior_is_carried_into_the_state() -> None:
    state = accumulate(CLAIM, [support()], BeliefConfig(prior=0.2))
    assert state.prior == 0.2
    assert state.posterior == pytest.approx(0.428571, abs=1e-5)
    assert state.movement > 0


def test_a_stronger_likelihood_model_moves_beliefs_further() -> None:
    mild = accumulate(CLAIM, [support()], BeliefConfig(likelihood=LikelihoodModel(2.0)))
    strong = accumulate(
        CLAIM, [support()], BeliefConfig(likelihood=LikelihoodModel(9.0))
    )
    assert strong.posterior > mild.posterior
