"""Run configuration: defaults, stage rules and fingerprints."""

from __future__ import annotations

import pytest

from hypoarena.belief import BeliefConfig
from hypoarena.config import DEFAULT_RUN_ID, STAGES, RunConfig
from hypoarena.dedup import DedupConfig
from hypoarena.errors import ValidationError


def test_stages_are_golden() -> None:
    assert STAGES == (
        "corpus",
        "generate",
        "verify",
        "dedup",
        "debate",
        "rank",
        "evolve",
        "accumulate",
        "report",
    )


def test_defaults_cover_the_whole_pipeline() -> None:
    config = RunConfig()
    assert config.stages == STAGES
    assert config.seed == 270106
    assert config.run_id == DEFAULT_RUN_ID
    assert isinstance(config.belief, BeliefConfig)


def test_stage_helpers_follow_the_configured_order() -> None:
    config = RunConfig(stages=("corpus", "verify", "report"))
    assert config.stage_index("verify") == 1
    assert config.stages_from("verify") == ("verify", "report")
    with pytest.raises(ValidationError, match="not part of this run"):
        config.stage_index("rank")


def test_invalid_stage_lists_are_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one stage"):
        RunConfig(stages=())
    with pytest.raises(ValidationError, match="unknown pipeline stages"):
        RunConfig(stages=("corpus", "publish"))
    with pytest.raises(ValidationError, match="duplicates"):
        RunConfig(stages=("corpus", "corpus"))


def test_run_ids_must_be_a_single_path_segment() -> None:
    assert RunConfig(run_id="2026-09-01").run_id == "2026-09-01"
    for bad in ("", "  ", "a/b", "..", "a\\b"):
        with pytest.raises(ValidationError):
            RunConfig(run_id=bad)


def test_fingerprints_are_stable_and_setting_sensitive() -> None:
    assert RunConfig().fingerprint() == RunConfig().fingerprint()
    assert RunConfig(seed=1).fingerprint() != RunConfig().fingerprint()
    assert RunConfig(stages=("corpus",)).fingerprint() != RunConfig().fingerprint()
    assert RunConfig(dedup=DedupConfig(threshold=0.5)).fingerprint() != (
        RunConfig().fingerprint()
    )
    assert RunConfig(belief=BeliefConfig(prior=0.2)).fingerprint() != (
        RunConfig().fingerprint()
    )
    assert len(RunConfig().fingerprint()) == 16


def test_with_stages_returns_a_restricted_copy() -> None:
    config = RunConfig()
    restricted = config.with_stages(("corpus", "verify"))
    assert restricted.stages == ("corpus", "verify")
    assert config.stages == STAGES
    assert restricted.seed == config.seed
