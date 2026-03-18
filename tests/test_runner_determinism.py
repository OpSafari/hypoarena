"""Reproducibility sweeps over the whole pipeline.

Same configuration must mean same bytes; a different seed must mean different
content. Both directions matter: a run that silently varies would make every
measurement in the reports meaningless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.runner import Pipeline, run_report
from hypoarena.synthetic import SyntheticConfig

SEEDS = (5, 270106)


def config(seed: int, run_id: str) -> RunConfig:
    return RunConfig(
        seed=seed,
        run_id=run_id,
        stages=STAGES,
        corpus=SyntheticConfig(seed=seed, chains=2, chain_length=2),
    )


def artifact_bytes(root: Path, run_id: str) -> dict[str, bytes]:
    store = ArtifactStore(root, run_id)
    return {
        name: store.path(name).read_bytes()
        for name in ("corpus.jsonl", "graph.jsonl", "grounding.jsonl", "report.json")
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_two_runs_of_one_configuration_are_identical(tmp_path: Path, seed: int) -> None:
    first = tmp_path / f"first-{seed}"
    second = tmp_path / f"second-{seed}"
    Pipeline(config(seed, "run"), ArtifactStore(first, "run")).run()
    Pipeline(config(seed, "run"), ArtifactStore(second, "run")).run()
    assert artifact_bytes(first, "run") == artifact_bytes(second, "run")


def test_different_seeds_produce_different_artifacts(tmp_path: Path) -> None:
    first = tmp_path / "seed-a"
    second = tmp_path / "seed-b"
    Pipeline(config(5, "run"), ArtifactStore(first, "run")).run()
    Pipeline(config(99, "run"), ArtifactStore(second, "run")).run()
    assert artifact_bytes(first, "run") != artifact_bytes(second, "run")


@pytest.mark.parametrize("seed", SEEDS)
def test_reports_are_stable_for_a_seed(tmp_path: Path, seed: int) -> None:
    # Same run_id keeps config_fingerprint identical; only the on-disk location
    # differs, so the two reports must be byte-for-byte equal.
    first = Pipeline(
        config(seed, "stable"), ArtifactStore(tmp_path / f"a-{seed}", "stable")
    )
    second = Pipeline(
        config(seed, "stable"), ArtifactStore(tmp_path / f"b-{seed}", "stable")
    )
    first.run()
    second.run()
    assert run_report(first) == run_report(second)


def test_the_report_records_the_corpus_hash(tmp_path: Path) -> None:
    pipeline = Pipeline(
        config(7, "hashed"), ArtifactStore(tmp_path / "hashed", "hashed")
    )
    pipeline.run()
    payload = run_report(pipeline)
    assert payload["corpus_hash"] == pipeline.state.corpus_hash
    assert isinstance(payload["corpus_hash"], str)
    assert len(payload["corpus_hash"]) == 16


def test_recovery_is_complete_for_seeded_runs(tmp_path: Path) -> None:
    for seed in SEEDS:
        pipeline = Pipeline(
            config(seed, f"recovered-{seed}"),
            ArtifactStore(tmp_path / f"recovered-{seed}", f"recovered-{seed}"),
        )
        pipeline.run()
        payload = run_report(pipeline)
        recovered = payload["recovered"]
        assert isinstance(recovered, dict)
        # config() plants `chains` chains of `chain_length` variables, i.e.
        # chains * (chain_length - 1) causal links; every one must be recovered.
        assert recovered["planted"] == 2 * (2 - 1)
        assert recovered["recovered"] == recovered["planted"]
        assert recovered["rate"] == 1.0


def test_stage_order_in_the_summary_follows_the_configuration(
    tmp_path: Path,
) -> None:
    pipeline = Pipeline(
        config(11, "ordered"), ArtifactStore(tmp_path / "ordered", "ordered")
    )
    summary = pipeline.run()
    assert tuple(stage.stage for stage in summary.stages) == STAGES
    assert all(stage.records >= 0 for stage in summary.stages)
