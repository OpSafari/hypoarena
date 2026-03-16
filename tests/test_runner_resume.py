"""Checkpointing: a resumed run must not change a single byte.

This is the property that makes long discovery runs safe to interrupt: artifacts
are canonical JSONL, nothing reads the clock, and every iteration order is
sorted, so splitting a run across processes reproduces the same files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.errors import ArtifactError, ConfigError
from hypoarena.runner import Pipeline
from hypoarena.synthetic import SyntheticConfig

# summary.json is deliberately absent: it records *how* the run was executed
# (which stages were skipped), so a resumed run is expected to differ there.
ARTIFACTS = (
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
    "cost.json",
    "report.json",
)


def config(run_id: str, stages: tuple[str, ...] = STAGES) -> RunConfig:
    return RunConfig(
        seed=23,
        run_id=run_id,
        stages=stages,
        corpus=SyntheticConfig(seed=23, chains=2, chain_length=2),
    )


def read_all(root: Path, run_id: str) -> dict[str, bytes]:
    store = ArtifactStore(root, run_id)
    return {name: store.path(name).read_bytes() for name in ARTIFACTS}


def test_a_resumed_run_reproduces_every_artifact(tmp_path: Path) -> None:
    straight = tmp_path / "straight"
    split = tmp_path / "split"
    # one run id in two roots: only the execution split may differ
    Pipeline(config("same"), ArtifactStore(straight, "same")).run()
    partial = config("same", stages=STAGES[:4])
    Pipeline(partial, ArtifactStore(split, "same")).run()
    resumed = Pipeline(config("same"), ArtifactStore(split, "same"))
    summary = resumed.run(resume=True)
    assert summary.skipped == STAGES[:4]
    assert summary.executed == STAGES[4:]
    assert read_all(straight, "same") == read_all(split, "same")


def test_cost_accounting_survives_a_resume(tmp_path: Path) -> None:
    straight = tmp_path / "cost-straight"
    split = tmp_path / "cost-split"
    first = Pipeline(config("costed"), ArtifactStore(straight, "costed"))
    first.run()
    partial = config("costed", stages=STAGES[:4])
    Pipeline(partial, ArtifactStore(split, "costed")).run()
    resumed = Pipeline(config("costed"), ArtifactStore(split, "costed"))
    resumed.run(resume=True)
    assert resumed.ledger.totals() == first.ledger.totals()
    assert resumed.store.read_json("cost.json") == first.store.read_json("cost.json")
    assert len(resumed.ledger.entries) == len(first.ledger.entries)


def test_the_summary_records_how_the_run_was_executed(tmp_path: Path) -> None:
    straight = tmp_path / "sum-straight"
    split = tmp_path / "sum-split"
    Pipeline(config("straight"), ArtifactStore(straight, "straight")).run()
    Pipeline(config("split", stages=STAGES[:4]), ArtifactStore(split, "split")).run()
    resumed = Pipeline(config("split"), ArtifactStore(split, "split"))
    resumed.run(resume=True)
    straight_summary = ArtifactStore(straight, "straight").read_json("summary.json")
    split_summary = ArtifactStore(split, "split").read_json("summary.json")
    assert straight_summary["skipped"] == []
    assert split_summary["skipped"] == list(STAGES[:4])
    assert straight_summary["stages"] != split_summary["stages"]


def test_resuming_twice_changes_nothing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "idempotent")
    run_config = config("idempotent")
    Pipeline(run_config, store).run()
    first = read_all(tmp_path, "idempotent")
    Pipeline(run_config, store).run(resume=True)
    Pipeline(run_config, store).run(resume=True)
    assert read_all(tmp_path, "idempotent") == first


def test_a_failed_stage_leaves_earlier_artifacts_usable(tmp_path: Path) -> None:
    class Failing(Pipeline):
        """Pipeline whose ranking stage always fails."""

        def stage_rank(self):  # noqa: ANN201 - mirrors the parent signature
            raise ConfigError("ranking unavailable")

    store = ArtifactStore(tmp_path, "failed")
    run_config = config("failed")
    with pytest.raises(ConfigError, match="ranking unavailable"):
        Failing(run_config, store).run()
    assert store.stage_done("verify") is True
    assert store.stage_done("rank") is False
    assert store.exists("grounding.jsonl")
    with pytest.raises(ArtifactError):
        store.read_lines("tournament.jsonl")


def test_a_resumed_run_completes_after_a_failure(tmp_path: Path) -> None:
    class Failing(Pipeline):
        def stage_rank(self):  # noqa: ANN201
            raise ConfigError("ranking unavailable")

    run_config = config("recovered")
    store = ArtifactStore(tmp_path, "recovered")
    with pytest.raises(ConfigError):
        Failing(run_config, store).run()
    summary = Pipeline(run_config, store).run(resume=True)
    assert summary.skipped == STAGES[: STAGES.index("rank")]
    assert "rank" in summary.executed
    assert store.stage_done("report") is True
    assert store.exists("report.json")


def test_checkpoints_can_be_cleared_to_force_a_rerun(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "cleared")
    run_config = config("cleared")
    Pipeline(run_config, store).run()
    assert store.clear_checkpoints() == len(STAGES)
    summary = Pipeline(run_config, store).run(resume=True)
    assert summary.skipped == ()
    assert summary.executed == STAGES


def test_resume_without_checkpoints_runs_everything(tmp_path: Path) -> None:
    run_config = config("fresh")
    summary = Pipeline(run_config, ArtifactStore(tmp_path, "fresh")).run(resume=True)
    assert summary.executed == STAGES
    assert summary.skipped == ()
