"""Debate outcomes become revised claims with honest provenance."""

from __future__ import annotations

import pytest

from hypoarena.agents import ScriptedAgent
from hypoarena.debate import DebateConfig, DebateLoop, apply_debate, revised_claim
from hypoarena.errors import UnknownReferenceError, ValidationError
from hypoarena.graph import HypothesisGraph
from hypoarena.schema import ClaimRelation
from hypoarena.synthetic import SyntheticConfig, build_bundle

CONTEXT = ("kinase K1 phosphorylates protein A",)


def debate(quality: float = 0.9) -> object:
    proposer = ScriptedAgent("proposer", quality=quality)
    loop = DebateLoop(
        proposer,
        [ScriptedAgent("critic", quality=quality)],
        config=DebateConfig(rounds=2, critics=1),
    )
    return loop.run(CONTEXT)


def test_a_revised_claim_keeps_identity_and_citations() -> None:
    bundle = build_bundle(SyntheticConfig(seed=3, chains=1, chain_length=2))
    original = bundle.claims[0]
    revised = revised_claim(original, debate())
    assert revised.claim_id == original.claim_id
    assert revised.citations == original.citations
    assert revised.statement != original.statement
    assert revised.subject == original.subject


def test_provenance_records_the_debate() -> None:
    bundle = build_bundle(SyntheticConfig(seed=3, chains=1, chain_length=2))
    original = bundle.claims[0]
    revised = revised_claim(original, debate())
    assert revised.provenance.origin == "agent"
    assert revised.provenance.agent_id == "proposer"
    assert revised.provenance.generation == original.provenance.generation + 1
    assert revised.provenance.parents == (original.claim_id,)
    assert revised.provenance.notes == "debate:2"
    assert revised.provenance.corpus_hash == original.provenance.corpus_hash


def test_an_explicit_agent_id_overrides_the_proposer_name() -> None:
    bundle = build_bundle(SyntheticConfig(seed=3, chains=1, chain_length=2))
    revised = revised_claim(bundle.claims[0], debate(), agent_id="agt_0123456789ab")
    assert revised.provenance.agent_id == "agt_0123456789ab"


def test_blank_outcomes_are_rejected() -> None:
    bundle = build_bundle(SyntheticConfig(seed=3, chains=1, chain_length=2))
    result = debate()
    empty = type(result)(
        proposal=result.proposal,
        final_statement="   ",
        turns=result.turns,
        converged=result.converged,
        rounds_run=result.rounds_run,
        agents=result.agents,
        usage=result.usage,
        context=result.context,
        config_fingerprint=result.config_fingerprint,
    )
    with pytest.raises(ValidationError, match="no statement"):
        revised_claim(bundle.claims[0], empty)


def test_apply_debate_preserves_graph_structure() -> None:
    bundle = build_bundle(SyntheticConfig(seed=3, chains=2, chain_length=2))
    graph = bundle.graph()
    claim = bundle.claims[0]
    before = (graph.link_count, graph.edge_count, len(graph))
    apply_debate(graph, claim.claim_id, debate())
    assert (graph.link_count, graph.edge_count, len(graph)) == before
    assert graph.claim(claim.claim_id).provenance.generation == (
        claim.provenance.generation + 1
    )
    assert graph.validate() is None


def test_apply_debate_reports_unknown_claims() -> None:
    graph = HypothesisGraph()
    with pytest.raises(UnknownReferenceError):
        apply_debate(graph, "clm_999999999999", debate())


def test_revisions_can_be_chained_with_increasing_generations() -> None:
    bundle = build_bundle(SyntheticConfig(seed=3, chains=1, chain_length=2))
    graph = HypothesisGraph()
    claim = bundle.claims[0]
    graph.add_claim(claim)
    graph.add_claim(bundle.claims[1])
    graph.add_edge(claim.claim_id, bundle.claims[1].claim_id, ClaimRelation.ENTAILS)
    for expected in (1, 2, 3):
        claim = apply_debate(graph, claim.claim_id, debate())
        assert claim.provenance.generation == expected
    assert graph.edge_count == 1
