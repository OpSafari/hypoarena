"""Graph accumulation, ranking and report tables."""

from __future__ import annotations

import pytest

from helpers import CLAIM_ID, OTHER_CLAIM_ID, sample_claim, sample_evidence
from hypoarena.belief import (
    BeliefConfig,
    accumulate_graph,
    belief_table,
    rank_beliefs,
    state_for,
)
from hypoarena.errors import UnknownReferenceError
from hypoarena.graph import HypothesisGraph
from hypoarena.schema import EvidencePolarity


def graph() -> HypothesisGraph:
    produced = HypothesisGraph()
    produced.add_claim(sample_claim())
    produced.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    produced.add_claim(sample_claim(claim_id="clm_222222222222"))
    produced.add_evidence(sample_evidence())
    produced.add_evidence(
        sample_evidence(
            evidence_id="evd_111111111111",
            polarity=EvidencePolarity.SUPPORT,
            strength=1.0,
        )
    )
    produced.link_evidence(CLAIM_ID, "evd_0123456789ab")
    produced.link_evidence(CLAIM_ID, "evd_111111111111")
    produced.link_evidence(OTHER_CLAIM_ID, "evd_0123456789ab")
    return produced


def test_every_claim_gets_a_state() -> None:
    states = accumulate_graph(graph())
    assert len(states) == 3
    assert [state.claim_id for state in states] == sorted(
        [CLAIM_ID, OTHER_CLAIM_ID, "clm_222222222222"]
    )


def test_uncited_claims_keep_the_prior() -> None:
    states = accumulate_graph(graph())
    uncited = state_for(states, "clm_222222222222")
    assert uncited.updates == 0
    assert uncited.posterior == 0.5


def test_more_supporting_evidence_means_a_higher_posterior() -> None:
    states = accumulate_graph(graph())
    assert (
        state_for(states, CLAIM_ID).posterior
        > state_for(states, OTHER_CLAIM_ID).posterior
    )


def test_ranking_is_total_and_reproducible() -> None:
    states = accumulate_graph(graph())
    ranking = rank_beliefs(states)
    assert ranking[0] == CLAIM_ID
    assert ranking == rank_beliefs(list(reversed(states)))
    assert len(set(ranking)) == 3


def test_ties_are_broken_by_identifier() -> None:
    tied = accumulate_graph(HypothesisGraph())
    produced = HypothesisGraph()
    produced.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    produced.add_claim(sample_claim(claim_id=CLAIM_ID))
    assert rank_beliefs(accumulate_graph(produced)) == (CLAIM_ID, OTHER_CLAIM_ID)
    assert tied == ()


def test_the_table_carries_positions() -> None:
    rows = belief_table(accumulate_graph(graph()))
    assert [row["position"] for row in rows] == [1, 2, 3]
    assert rows[0]["claim_id"] == CLAIM_ID
    posterior = [float(row["posterior"]) for row in rows]
    assert posterior == sorted(posterior, reverse=True)


def test_configuration_changes_the_table() -> None:
    produced = graph()
    default = accumulate_graph(produced)
    rejected = accumulate_graph(
        produced,
        BeliefConfig(prior=0.1),
    )
    assert (
        state_for(default, CLAIM_ID).posterior
        != state_for(rejected, CLAIM_ID).posterior
    )


def test_unknown_lookups_are_reported() -> None:
    with pytest.raises(UnknownReferenceError):
        state_for(accumulate_graph(graph()), "clm_999999999999")
