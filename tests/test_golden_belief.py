"""Hand-computable belief goldens.

Every number below can be derived with a calculator from the documented model:
odds multiply by ``3 ** strength`` for support and by ``3 ** -strength`` for
refutation, and the posterior is ``odds / (1 + odds)``. Pinning them makes any
change to the update rule visible immediately.
"""

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
IGNORE = BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE)
# a high threshold keeps REJECT from flooring, so refutations apply in full
COUNTING = BeliefConfig(
    contradiction_policy=ContradictionPolicy.REJECT, contradiction_threshold=99
)


def support(strength: float) -> object:
    return sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=strength)


def refute(strength: float) -> object:
    return sample_evidence(polarity=EvidencePolarity.REFUTE, strength=strength)


def test_one_full_support_gives_three_quarters() -> None:
    state = accumulate(CLAIM, [support(1.0)], IGNORE)
    assert state.likelihood_ratio == pytest.approx(3.0)
    assert state.posterior == pytest.approx(0.75)


def test_two_full_supports_give_nine_tenths() -> None:
    state = accumulate(CLAIM, [support(1.0), support(1.0)], IGNORE)
    assert state.likelihood_ratio == pytest.approx(9.0)
    assert state.posterior == pytest.approx(0.9)


def test_support_then_refutation_returns_to_the_prior() -> None:
    state = accumulate(CLAIM, [support(1.0), refute(1.0)], COUNTING)
    assert state.likelihood_ratio == pytest.approx(1.0)
    assert state.posterior == pytest.approx(0.5)


def test_a_single_refutation_gives_one_quarter() -> None:
    state = accumulate(CLAIM, [refute(1.0)], COUNTING)
    assert state.posterior == pytest.approx(0.25)


def test_half_strength_evidence_uses_the_square_root() -> None:
    state = accumulate(CLAIM, [support(0.5)], IGNORE)
    assert state.likelihood_ratio == pytest.approx(3.0**0.5)
    assert state.posterior == pytest.approx(3.0**0.5 / (1 + 3.0**0.5))


def test_a_quarter_prior_with_one_support() -> None:
    config = BeliefConfig(prior=0.25, contradiction_policy=ContradictionPolicy.IGNORE)
    state = accumulate(CLAIM, [support(1.0)], config)
    # odds 1/3 * 3 = 1 -> back to an even posterior
    assert state.posterior == pytest.approx(0.5)


def test_discount_halves_the_exponent_of_a_contested_claim() -> None:
    undiscounted = accumulate(CLAIM, [support(1.0), support(1.0)], IGNORE)
    contested = accumulate(
        CLAIM,
        [support(1.0), support(1.0), refute(1.0)],
        BeliefConfig(contradiction_policy=ContradictionPolicy.DISCOUNT),
    )
    assert contested.likelihood_ratio == pytest.approx((9.0 * (1 / 3.0)) ** 0.5)
    assert contested.posterior < undiscounted.posterior


def test_reject_floors_the_posterior() -> None:
    state = accumulate(
        CLAIM,
        [support(1.0), refute(1.0)],
        BeliefConfig(contradiction_policy=ContradictionPolicy.REJECT),
    )
    assert state.posterior == MIN_PROBABILITY


def test_a_custom_likelihood_model_is_applied_exactly() -> None:
    config = BeliefConfig(
        likelihood=LikelihoodModel(support_ratio=10.0, refute_ratio=10.0),
        contradiction_policy=ContradictionPolicy.IGNORE,
    )
    state = accumulate(CLAIM, [support(0.5)], config)
    assert state.likelihood_ratio == pytest.approx(10.0**0.5)
    assert state.posterior == pytest.approx(10.0**0.5 / (1 + 10.0**0.5))


def test_state_counts_are_exact() -> None:
    state = accumulate(
        CLAIM,
        [
            support(1.0),
            support(0.5),
            refute(1.0),
            sample_evidence(polarity=EvidencePolarity.NEUTRAL, strength=0.0),
        ],
        COUNTING,
    )
    assert (state.supporting, state.refuting, state.neutral) == (2, 1, 1)
    assert state.updates == 4
