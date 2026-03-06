"""Property sweeps over belief accumulation.

These are the guarantees the runner relies on when it accumulates evidence in
batches: the result depends on the evidence set, not on its order or on how it
was split across calls; adding support never lowers a belief; adding refutation
never raises it; and posteriors stay inside the open unit interval.
"""

from __future__ import annotations

from random import Random

import pytest

from helpers import sample_evidence
from hypoarena.belief import (
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    BeliefConfig,
    ContradictionPolicy,
    accumulate,
    accumulate_graph,
    prior_sensitivity,
)
from hypoarena.schema import EvidencePolarity
from hypoarena.synthetic import SyntheticConfig, build_bundle

CLAIM = "clm_0123456789ab"
POLICIES = tuple(ContradictionPolicy)


def support(strength: float) -> object:
    return sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=strength)


def refute(strength: float) -> object:
    return sample_evidence(polarity=EvidencePolarity.REFUTE, strength=strength)


def item_key(evidence: object) -> str:
    return evidence.evidence_id  # type: ignore[attr-defined]


def random_evidence(seed: int, count: int) -> list[object]:
    rng = Random(seed)
    items = []
    for index in range(count):
        polarity = rng.choice(
            [
                EvidencePolarity.SUPPORT,
                EvidencePolarity.REFUTE,
                EvidencePolarity.NEUTRAL,
            ]
        )
        strength = (
            0.0 if polarity is EvidencePolarity.NEUTRAL else round(rng.random(), 3)
        )
        items.append(
            sample_evidence(
                evidence_id=f"evd_{index:012x}", polarity=polarity, strength=strength
            )
        )
    return items


@pytest.mark.parametrize("policy", POLICIES)
def test_order_does_not_matter(policy: ContradictionPolicy) -> None:
    config = BeliefConfig(contradiction_policy=policy)
    evidence = random_evidence(3, 8)
    forward = accumulate(CLAIM, evidence, config)
    backward = accumulate(CLAIM, list(reversed(evidence)), config)
    shuffled = accumulate(CLAIM, Random(9).sample(evidence, len(evidence)), config)
    assert forward.posterior == pytest.approx(backward.posterior)
    assert forward.posterior == pytest.approx(shuffled.posterior)


def test_splitting_a_batch_gives_the_same_answer() -> None:
    evidence = random_evidence(5, 6)
    whole = accumulate(
        CLAIM, evidence, BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE)
    )
    config = BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE)
    first = accumulate(CLAIM, evidence[:3], config)
    second = accumulate(CLAIM, evidence[3:], config)
    combined = first.likelihood_ratio * second.likelihood_ratio
    assert whole.likelihood_ratio == pytest.approx(combined)


def test_more_support_never_lowers_the_belief() -> None:
    config = BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE)
    posteriors = [
        accumulate(CLAIM, [support(1.0)] * count, config).posterior
        for count in range(5)
    ]
    assert posteriors == sorted(posteriors)


def test_more_refutation_never_raises_the_belief() -> None:
    config = BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE)
    posteriors = [
        accumulate(CLAIM, [refute(1.0)] * count, config).posterior for count in range(5)
    ]
    assert posteriors == sorted(posteriors, reverse=True)


@pytest.mark.parametrize("seed", [1, 7, 270106])
def test_posteriors_stay_inside_the_open_interval(seed: int) -> None:
    evidence = random_evidence(seed, 20)
    for policy in POLICIES:
        state = accumulate(CLAIM, evidence, BeliefConfig(contradiction_policy=policy))
        assert MIN_PROBABILITY <= state.posterior <= MAX_PROBABILITY


@pytest.mark.parametrize("seed", [1, 7])
def test_graph_accumulation_matches_manual_accumulation(seed: int) -> None:
    bundle = build_bundle(SyntheticConfig(seed=seed, chains=2, chain_length=3))
    graph = bundle.graph()
    states = {state.claim_id: state for state in accumulate_graph(graph)}
    for claim in graph.claims:
        evidence = sorted(graph.evidence_for(claim.claim_id), key=item_key)
        manual = accumulate(claim.claim_id, evidence)
        assert states[claim.claim_id].posterior == pytest.approx(manual.posterior)


def test_sensitivity_is_monotone_in_the_prior_for_any_evidence() -> None:
    for seed in (1, 2, 3):
        pairs = prior_sensitivity(random_evidence(seed, 5))
        posteriors = [posterior for _, posterior in pairs]
        assert posteriors == sorted(posteriors)
