"""Debate and rank stages: transcripts, revisions and standings."""

from __future__ import annotations

from pathlib import Path

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import RunConfig
from hypoarena.debate import DebateConfig
from hypoarena.runner import DEBATE_CLAIM_LIMIT, Pipeline, debate_from_line, debate_line
from hypoarena.synthetic import SyntheticConfig
from hypoarena.tournament import TournamentConfig

STAGES = ("corpus", "generate", "verify", "debate", "rank")


def config(run_id: str) -> RunConfig:
    return RunConfig(
        seed=13,
        run_id=run_id,
        stages=STAGES,
        corpus=SyntheticConfig(seed=13, chains=2, chain_length=2),
        debate=DebateConfig(rounds=2, critics=1),
        tournament=TournamentConfig(seed=13, repeats=1),
    )


def run(tmp_path: Path, run_id: str) -> Pipeline:
    pipeline = Pipeline(config(run_id), ArtifactStore(tmp_path, run_id))
    pipeline.run()
    return pipeline


def test_debates_are_recorded_for_proposed_claims(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "debate")
    proposed = [
        claim
        for claim in pipeline.state.graph.claims
        if claim.provenance.origin == "agent"
    ]
    assert len(pipeline.state.debates) == min(len(proposed), DEBATE_CLAIM_LIMIT)
    assert all(result.rounds_run >= 1 for result in pipeline.state.debates)


def test_debated_claims_carry_revised_provenance(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "revised")
    revised = [
        claim
        for claim in pipeline.state.graph.claims
        if claim.provenance.notes.startswith("debate:")
    ]
    assert revised
    assert all(claim.provenance.generation >= 1 for claim in revised)
    assert all(claim.provenance.origin == "agent" for claim in revised)
    assert all(
        claim.provenance.agent_id == pipeline.agents[-1].name for claim in revised
    )


def test_corpus_claims_are_not_debated(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "untouched")
    for claim in pipeline.state.graph.claims:
        if claim.provenance.notes in ("planted", "competing"):
            assert claim.provenance.generation == 0
            assert claim.is_cited is True


def test_the_tournament_ranks_every_claim(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "ranked")
    tournament = pipeline.state.tournament
    assert tournament is not None
    assert len(tournament.ratings) == len(pipeline.state.graph)
    assert len(tournament.matches) >= 1
    assert tournament.config.seed == 13


def test_grounded_claims_outrank_uncited_proposals(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "ordering")
    tournament = pipeline.state.tournament
    assert tournament is not None
    ranking = tournament.ranking()
    top = pipeline.state.graph.claim(ranking[0])
    bottom = pipeline.state.graph.claim(ranking[-1])
    assert top.is_cited is True
    assert bottom.provenance.notes.startswith("debate:") or not bottom.is_cited


def test_debate_lines_roundtrip(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "lines")
    for result in pipeline.state.debates:
        assert debate_from_line(debate_line(result)) == result


def test_artifacts_include_debates_and_tournament(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "artifacts")
    listing = pipeline.store.listing()
    assert "debates.jsonl" in listing
    assert "tournament.jsonl" in listing


def test_debate_usage_is_booked_once(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "cost")
    stages = {entry.stage for entry in pipeline.ledger.entries}
    assert stages == {"generate", "debate"}
    totals = pipeline.ledger.totals()
    assert totals["calls"] == sum(agent.usage.calls for agent in pipeline.agents)


def test_resuming_restores_debates_and_standings(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "resume")
    run_config = config("resume")
    first = Pipeline(run_config, store)
    first.run()
    second = Pipeline(run_config, store)
    summary = second.run(resume=True)
    assert summary.skipped == STAGES
    assert len(second.state.debates) == len(first.state.debates)
    assert second.state.tournament is not None
    assert second.state.tournament.ratings == first.state.tournament.ratings
