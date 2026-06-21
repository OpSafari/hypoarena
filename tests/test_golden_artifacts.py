"""Golden over the on-disk artifact contract of a full run.

The exact set of files a run writes (and the per-stage checkpoints it leaves) is
a contract the resume logic, the CLI and the reports all rely on. Pinning it
means adding or renaming an artifact is a deliberate change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.runner import Pipeline
from hypoarena.synthetic import SyntheticConfig

EXPECTED_FILES = {
    "beliefs.jsonl",
    "candidates.jsonl",
    "corpus.jsonl",
    "cost.json",
    "debates.jsonl",
    "dedup.jsonl",
    "evolution.jsonl",
    "graph.jsonl",
    "grounding.jsonl",
    "report.html",
    "report.json",
    "report.md",
    "run.json",
    "summary.json",
    "tournament.jsonl",
    "truth.jsonl",
}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> ArtifactStore:
    root = tmp_path_factory.mktemp("artifacts")
    config = RunConfig(
        seed=270106,
        run_id="run",
        stages=STAGES,
        corpus=SyntheticConfig(seed=270106, chains=2, chain_length=2),
    )
    Pipeline(config, ArtifactStore(root, "run")).run()
    return ArtifactStore(root, "run")


def test_full_run_writes_exactly_the_expected_files(store: ArtifactStore) -> None:
    assert set(store.listing()) == EXPECTED_FILES


def test_every_stage_left_a_checkpoint(store: ArtifactStore) -> None:
    for stage in STAGES:
        assert store.stage_done(stage), stage


def test_the_checkpoint_count_matches_the_stage_count(store: ArtifactStore) -> None:
    markers = list(Path(store.checkpoint_dir).glob("*.done"))
    assert len(markers) == len(STAGES)
