"""Measured belief accumulation on synthetic corpora with planted ground truth.

Corpus: ``SyntheticConfig(seed=11, chains=2, chain_length=3,
paraphrases_per_link=2)`` — four planted claims (two with three supporting items
and one refuting item, two with two supporting items and one refuting item) and
two rival claims with a single supporting item each.

The point of this file is the *policy* comparison: whether contested-but-well-
supported hypotheses outrank untested rivals depends entirely on how
contradictions are handled, and the numbers below pin that behaviour.
"""

from __future__ import annotations

import pytest

from hypoarena.belief import (
    BeliefConfig,
    ContradictionPolicy,
    accumulate_graph,
    prior_sensitivity,
    rank_beliefs,
    sensitivity_spread,
    state_for,
)
from hypoarena.synthetic import SyntheticConfig, build_bundle

PLANTED_MEANS = {
    ContradictionPolicy.IGNORE: 0.8523,
    ContradictionPolicy.DOWNWEIGHT: 0.8274,
    ContradictionPolicy.DISCOUNT: 0.7436,
    ContradictionPolicy.REJECT: 0.4330,
}
RIVAL_MEAN = 0.7061


def bundle() -> object:
    return build_bundle(SyntheticConfig(seed=11, chains=2, chain_length=3))


def roles(produced: object) -> tuple[list[str], list[str]]:
    planted = [
        claim.claim_id
        for claim in produced.claims
        if claim.provenance.notes == "planted"
    ]
    rivals = [
        claim.claim_id
        for claim in produced.claims
        if claim.provenance.notes == "competing"
    ]
    return planted, rivals


def mean(states: tuple[object, ...], identifiers: list[str]) -> float:
    return round(
        sum(state_for(states, item).posterior for item in identifiers)
        / len(identifiers),
        4,
    )


def test_evidence_is_distributed_as_planted() -> None:
    produced = bundle()
    graph = produced.graph()
    planted, rivals = roles(produced)
    assert len(planted) == 4
    assert len(rivals) == 2
    assert sorted(len(graph.evidence_for(item)) for item in planted) == [2, 2, 3, 3]
    assert [len(graph.evidence_for(item)) for item in rivals] == [1, 1]


@pytest.mark.parametrize("policy", list(ContradictionPolicy))
def test_planted_means_match_the_measured_table(policy: ContradictionPolicy) -> None:
    produced = bundle()
    states = accumulate_graph(
        produced.graph(), BeliefConfig(contradiction_policy=policy)
    )
    planted, rivals = roles(produced)
    assert mean(states, planted) == pytest.approx(PLANTED_MEANS[policy], abs=1e-4)
    assert mean(states, rivals) == pytest.approx(RIVAL_MEAN, abs=1e-4)


def test_lenient_policies_keep_planted_claims_above_rivals() -> None:
    produced = bundle()
    planted, rivals = roles(produced)
    for policy in (
        ContradictionPolicy.IGNORE,
        ContradictionPolicy.DOWNWEIGHT,
        ContradictionPolicy.DISCOUNT,
    ):
        states = accumulate_graph(
            produced.graph(), BeliefConfig(contradiction_policy=policy)
        )
        assert mean(states, planted) > mean(states, rivals), policy
        assert rank_beliefs(states)[0] in planted


def test_the_reject_policy_floors_contested_claims() -> None:
    produced = bundle()
    states = accumulate_graph(
        produced.graph(),
        BeliefConfig(contradiction_policy=ContradictionPolicy.REJECT),
    )
    planted, rivals = roles(produced)
    assert mean(states, planted) < mean(states, rivals)
    # the uncontested planted claim is still the top-ranked one
    assert rank_beliefs(states)[0] in planted


def test_results_do_not_depend_on_the_corpus_seed() -> None:
    first = bundle()
    second = build_bundle(SyntheticConfig(seed=99, chains=2, chain_length=3))
    config = BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE)
    assert mean(accumulate_graph(first.graph(), config), roles(first)[0]) == (
        pytest.approx(
            mean(accumulate_graph(second.graph(), config), roles(second)[0]),
            abs=1e-4,
        )
    )


def test_prior_sensitivity_of_a_planted_claim_is_measured() -> None:
    produced = bundle()
    graph = produced.graph()
    planted, _ = roles(produced)
    evidence = sorted(graph.evidence_for(planted[0]), key=lambda item: item.evidence_id)
    pairs = prior_sensitivity(evidence)
    assert [prior for prior, _ in pairs] == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert sensitivity_spread(evidence) == pytest.approx(0.7843, abs=1e-4)
