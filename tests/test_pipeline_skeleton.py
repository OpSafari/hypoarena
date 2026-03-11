"""Pipeline plumbing: stage results, summaries and configuration checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.agents import ScriptedAgent
from hypoarena.artifacts import ArtifactStore
from hypoarena.config import RunConfig
from hypoarena.cost import CostLedger
from hypoarena.errors import ConfigError, ValidationError
from hypoarena.runner import (
    DEFAULT_AGENT_QUALITIES,
    Pipeline,
    RunState,
    RunSummary,
    StageResult,
    default_agents,
)


def test_stage_results_validate_and_serialize() -> None:
    result = StageResult("corpus", 12, ("corpus.jsonl",))
    assert result.as_dict() == {
        "stage": "corpus",
        "records": 12,
        "artifacts": ["corpus.jsonl"],
        "skipped": False,
    }
    with pytest.raises(ValidationError, match="stage name"):
        StageResult(" ", 0, ())
    with pytest.raises(ValidationError, match="records"):
        StageResult("corpus", -1, ())


def test_summaries_separate_executed_and_skipped_stages() -> None:
    summary = RunSummary(
        run_id="demo",
        config_fingerprint="0" * 16,
        stages=(
            StageResult("corpus", 3, ("corpus.jsonl",)),
            StageResult("verify", 0, (), skipped=True),
        ),
        completed=True,
    )
    assert summary.executed == ("corpus",)
    assert summary.skipped == ("verify",)
    assert summary.as_dict()["completed"] is True
    assert len(summary.signature()) == 16


def test_default_agents_cover_the_quality_tiers() -> None:
    agents = default_agents(7)
    assert DEFAULT_AGENT_QUALITIES == (0.9, 0.5, 0.2)
    assert [agent.quality for agent in agents] == list(DEFAULT_AGENT_QUALITIES)
    assert len({agent.name for agent in agents}) == 3
    assert all(isinstance(agent, ScriptedAgent) for agent in agents)


def test_a_pipeline_rejects_a_mismatched_store(tmp_path: Path) -> None:
    config = RunConfig(run_id="alpha")
    with pytest.raises(ConfigError, match="disagree"):
        Pipeline(config, ArtifactStore(tmp_path, "beta"))


def test_a_pipeline_needs_agents(tmp_path: Path) -> None:
    config = RunConfig(run_id="alpha")
    with pytest.raises(ConfigError, match="at least one agent"):
        Pipeline(config, ArtifactStore(tmp_path, "alpha"), agents=[])


def test_unknown_stages_are_reported(tmp_path: Path) -> None:
    config = RunConfig(run_id="alpha", stages=("corpus",))
    pipeline = Pipeline(config, ArtifactStore(tmp_path, "alpha"))
    with pytest.raises(ConfigError, match="no implementation"):
        pipeline.run()


def test_the_skeleton_exposes_shared_state_and_a_ledger(tmp_path: Path) -> None:
    config = RunConfig(run_id="alpha", stages=("corpus",))
    pipeline = Pipeline(config, ArtifactStore(tmp_path, "alpha"))
    assert isinstance(pipeline.state, RunState)
    assert isinstance(pipeline.ledger, CostLedger)
    assert pipeline.handlers() == {}
    with pytest.raises(NotImplementedError):
        pipeline.restore("corpus")
