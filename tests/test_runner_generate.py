"""Proposal extraction: relation keywords, claim building and refusals."""

from __future__ import annotations

from hypoarena.runner import (
    PROPOSAL_SCOPE,
    RELATION_KEYWORDS,
    claim_from_proposal,
    relation_from_statement,
)
from hypoarena.schema import PredictedRelation


def test_keyword_mapping_covers_the_documented_verbs() -> None:
    assert relation_from_statement("protein A increases cell growth") is (
        PredictedRelation.INCREASES
    )
    assert relation_from_statement("kinase K1 blocks gene G1") is (
        PredictedRelation.INHIBITS
    )
    assert relation_from_statement("metabolite M1 is required for growth") is (
        PredictedRelation.ENABLES
    )


def test_unknown_wording_stays_associative() -> None:
    assert relation_from_statement("something happens somewhere") is (
        PredictedRelation.ASSOCIATES
    )
    assert relation_from_statement("") is PredictedRelation.ASSOCIATES


def test_the_first_matching_keyword_wins() -> None:
    statement = "protein A increases and then decreases cell growth"
    assert relation_from_statement(statement) is PredictedRelation.INCREASES
    assert RELATION_KEYWORDS[0][0] == "increases"


def test_a_proposal_becomes_an_uncited_agent_claim() -> None:
    claim = claim_from_proposal(
        "kinase K1 increases protein A activity", agent="scripted-0.9", seed=7
    )
    assert claim is not None
    assert claim.subject == "kinase"
    assert claim.object == "activity"
    assert claim.relation is PredictedRelation.INCREASES
    assert claim.scope.population == PROPOSAL_SCOPE
    assert claim.citations == ()
    assert claim.is_cited is False
    assert claim.provenance.origin == "agent"
    assert claim.provenance.notes == "proposed"
    assert claim.provenance.seed == 7


def test_proposals_are_deterministic_and_agent_scoped() -> None:
    statement = "kinase K1 increases protein A activity"
    first = claim_from_proposal(statement, agent="a", seed=7)
    assert first == claim_from_proposal(statement, agent="a", seed=7)
    other = claim_from_proposal(statement, agent="b", seed=7)
    assert other is not None and first is not None
    assert first.claim_id != other.claim_id


def test_unusable_proposals_are_refused() -> None:
    assert claim_from_proposal("binding", agent="a", seed=1) is None
    assert claim_from_proposal("the of and", agent="a", seed=1) is None
    assert claim_from_proposal("   ", agent="a", seed=1) is None


def test_statements_are_whitespace_normalized() -> None:
    claim = claim_from_proposal(
        "  kinase   K1\tincreases protein A  ", agent="a", seed=1
    )
    assert claim is not None
    assert claim.statement == "kinase K1 increases protein A"


def test_content_words_come_from_the_stopword_filtered_tokens() -> None:
    claim = claim_from_proposal(
        "we observed that gene G3 is required for metabolite M1", agent="a", seed=1
    )
    assert claim is not None
    assert claim.subject == "gene"
    assert claim.object == "m1"


def test_a_pipeline_runs_the_corpus_and_generate_stages(tmp_path) -> None:
    from hypoarena.artifacts import ArtifactStore
    from hypoarena.config import RunConfig
    from hypoarena.runner import Pipeline
    from hypoarena.synthetic import SyntheticConfig

    config = RunConfig(
        seed=11,
        run_id="smoke",
        stages=("corpus", "generate"),
        corpus=SyntheticConfig(seed=11, chains=1, chain_length=2),
    )
    store = ArtifactStore(tmp_path, "smoke")
    pipeline = Pipeline(config, store)
    summary = pipeline.run()
    assert summary.completed is True
    assert summary.executed == ("corpus", "generate")
    assert store.exists("corpus.jsonl")
    assert store.exists("truth.jsonl")
    assert store.exists("graph.jsonl")
    assert store.exists("candidates.jsonl")
    assert store.completed_stages() == ("corpus", "generate")
    assert len(pipeline.state.candidates) == len(pipeline.agents)
    assert all(
        claim.provenance.notes == "proposed" for claim in pipeline.state.candidates
    )
    assert pipeline.ledger.totals()["calls"] == len(pipeline.agents)


def test_generated_candidates_are_uncited_and_verified_later(tmp_path) -> None:
    from hypoarena.artifacts import ArtifactStore
    from hypoarena.config import RunConfig
    from hypoarena.runner import Pipeline
    from hypoarena.synthetic import SyntheticConfig

    config = RunConfig(
        seed=11,
        run_id="smoke2",
        stages=("corpus", "generate"),
        corpus=SyntheticConfig(seed=11, chains=1, chain_length=2),
    )
    pipeline = Pipeline(config, ArtifactStore(tmp_path, "smoke2"))
    pipeline.run()
    for claim in pipeline.state.candidates:
        assert claim.is_cited is False
        assert pipeline.state.graph.has_claim(claim.claim_id)
    gold = [
        item
        for item in pipeline.state.graph.claims
        if item.provenance.notes in ("planted", "competing")
    ]
    assert gold and all(item.is_cited for item in gold)
