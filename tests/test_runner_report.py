"""Full pipeline: evolution, accumulation and the run report."""

from __future__ import annotations

from pathlib import Path

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.evolve import EvolutionConfig
from hypoarena.runner import (
    REPORT_LIMITATIONS,
    Pipeline,
    recovered_links,
    run_report,
)
from hypoarena.synthetic import SyntheticConfig


def config(run_id: str, **overrides: object) -> RunConfig:
    payload: dict[str, object] = {
        "seed": 17,
        "run_id": run_id,
        "stages": STAGES,
        "corpus": SyntheticConfig(seed=17, chains=2, chain_length=2),
        "evolution": EvolutionConfig(seed=17, generations=2),
    }
    payload.update(overrides)
    return RunConfig(**payload)  # type: ignore[arg-type]


def run(tmp_path: Path, run_id: str) -> Pipeline:
    pipeline = Pipeline(config(run_id), ArtifactStore(tmp_path, run_id))
    pipeline.run()
    return pipeline


def test_the_whole_pipeline_produces_every_artifact(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "full")
    listing = pipeline.store.listing()
    for name in (
        "corpus.jsonl",
        "truth.jsonl",
        "graph.jsonl",
        "candidates.jsonl",
        "grounding.jsonl",
        "dedup.jsonl",
        "debates.jsonl",
        "tournament.jsonl",
        "evolution.jsonl",
        "beliefs.jsonl",
        "report.json",
        "run.json",
        "summary.json",
    ):
        assert name in listing, name


def test_evolution_grows_the_graph_and_records_generations(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "evolved")
    steps = pipeline.state.evolution
    assert len(steps) == 2
    accepted = sum(step.accepted_count for step in steps)
    assert accepted >= 1
    evolved = [
        claim
        for claim in pipeline.state.graph.claims
        if claim.provenance.origin == "evolved"
    ]
    assert len(evolved) == accepted
    pipeline.state.graph.validate()


def test_beliefs_cover_every_claim(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "beliefs")
    assert len(pipeline.state.beliefs) == len(pipeline.state.graph)
    assert all(0.0 < state.posterior < 1.0 for state in pipeline.state.beliefs)


def test_the_report_payload_is_complete(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "report")
    payload = run_report(pipeline)
    assert sorted(payload) == [
        "beliefs",
        "config_fingerprint",
        "corpus_hash",
        "cost",
        "counts",
        "dedup",
        "evolution",
        "grounding",
        "limitations",
        "ranking",
        "recovered",
        "run_id",
    ]
    assert payload["run_id"] == "report"
    assert payload["limitations"] == list(REPORT_LIMITATIONS)
    assert len(payload["limitations"]) == 5


def test_planted_links_are_recovered_from_the_corpus(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "recovery")
    truth = pipeline.state.truth
    assert truth is not None
    recovered = recovered_links(truth, pipeline.state.graph)
    assert recovered
    assert all(found for _, found in recovered)
    payload = run_report(pipeline)
    assert payload["recovered"]["rate"] == 1.0  # type: ignore[index]


def test_the_report_states_its_own_limits(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "limits")
    stored = pipeline.store.read_json("report.json")
    assert "synthetic" in " ".join(stored["limitations"]).lower()
    assert "not" in " ".join(stored["limitations"]).lower()


def test_counts_describe_the_final_graph(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "counts")
    payload = run_report(pipeline)
    stats = pipeline.state.graph.stats()
    counts = payload["counts"]
    assert isinstance(counts, dict)
    assert counts["claims"] == stats.claims
    assert counts["evidence"] == stats.evidence
    assert counts["documents"] == len(pipeline.state.corpus)


def test_cost_totals_match_the_agents(tmp_path: Path) -> None:
    pipeline = run(tmp_path, "cost")
    totals = run_report(pipeline)["cost"]
    assert isinstance(totals, dict)
    assert totals["calls"] == sum(agent.usage.calls for agent in pipeline.agents)
    assert totals["total_tokens"] > 0
