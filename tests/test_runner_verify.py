"""Verify and dedup stages: artifacts, state and resume behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import RunConfig
from hypoarena.dedup import DedupConfig
from hypoarena.errors import ConfigError
from hypoarena.grounding import GroundingFlag
from hypoarena.runner import Pipeline
from hypoarena.synthetic import SyntheticConfig


def config(run_id: str, **overrides: object) -> RunConfig:
    payload: dict[str, object] = {
        "seed": 11,
        "run_id": run_id,
        "stages": ("corpus", "generate", "verify", "dedup"),
        "corpus": SyntheticConfig(seed=11, chains=2, chain_length=2),
        "dedup": DedupConfig(
            method="jaccard",
            threshold=0.9,
            ngram_size=1,
            shingle_unit="word",
            content_only=True,
        ),
    }
    payload.update(overrides)
    return RunConfig(**payload)  # type: ignore[arg-type]


def test_verify_grades_every_claim(tmp_path: Path) -> None:
    pipeline = Pipeline(config("verify-run"), ArtifactStore(tmp_path, "verify-run"))
    summary = pipeline.run()
    assert summary.executed == ("corpus", "generate", "verify", "dedup")
    reports = pipeline.state.reports
    assert len(reports) == len(pipeline.state.graph)
    flags = {report.flag for report in reports}
    assert GroundingFlag.GROUNDED in flags
    # agent proposals carry no citations, so they must be graded ungrounded
    assert GroundingFlag.UNGROUNDED in flags


def test_proposed_claims_are_ungrounded_and_gold_claims_are_grounded(
    tmp_path: Path,
) -> None:
    pipeline = Pipeline(config("split"), ArtifactStore(tmp_path, "split"))
    pipeline.run()
    by_id = {report.claim_id: report for report in pipeline.state.reports}
    for claim in pipeline.state.graph.claims:
        report = by_id[claim.claim_id]
        if claim.provenance.notes == "proposed":
            assert report.flag is GroundingFlag.UNGROUNDED
        else:
            assert report.flag is GroundingFlag.GROUNDED


def test_dedup_reports_clusters_over_claims(tmp_path: Path) -> None:
    pipeline = Pipeline(config("dedup-run"), ArtifactStore(tmp_path, "dedup-run"))
    pipeline.run()
    report = pipeline.state.dedup
    assert report is not None
    assert report.total == len(pipeline.state.graph)
    assert report.config.threshold == 0.9


def test_artifacts_are_written_and_listed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "artifacts")
    Pipeline(config("artifacts"), store).run()
    listing = store.listing()
    for name in (
        "corpus.jsonl",
        "truth.jsonl",
        "graph.jsonl",
        "candidates.jsonl",
        "grounding.jsonl",
        "dedup.jsonl",
        "run.json",
        "summary.json",
    ):
        assert name in listing


def test_stages_refuse_to_run_without_their_inputs(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "orphan")
    pipeline = Pipeline(config("orphan", stages=("verify",)), store)
    with pytest.raises(ConfigError, match="corpus stage"):
        pipeline.run()


def test_resuming_skips_completed_stages(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "resume")
    run_config = config("resume")
    Pipeline(run_config, store).run()
    resumed = Pipeline(run_config, store)
    summary = resumed.run(resume=True)
    assert summary.skipped == ("corpus", "generate", "verify", "dedup")
    assert summary.executed == ()
    assert len(resumed.state.reports) == len(resumed.state.graph)
    assert resumed.state.dedup is not None
