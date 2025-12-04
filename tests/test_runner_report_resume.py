"""A resumed run reproduces the rendered reports byte-for-byte.

report.md and report.html are pure functions of report.json, and report.json is
already proven resume-stable; these tests close the loop on the rendered files so
a split run cannot silently ship a different document than a straight run.
"""

from __future__ import annotations

from pathlib import Path

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.reports import HTML_ARTIFACT, MARKDOWN_ARTIFACT
from hypoarena.runner import Pipeline
from hypoarena.synthetic import SyntheticConfig


def config(run_id: str, stages: tuple[str, ...] = STAGES) -> RunConfig:
    return RunConfig(
        seed=23,
        run_id=run_id,
        stages=stages,
        corpus=SyntheticConfig(seed=23, chains=2, chain_length=2),
    )


def read_reports(root: Path, run_id: str) -> dict[str, bytes]:
    store = ArtifactStore(root, run_id)
    return {
        name: store.path(name).read_bytes()
        for name in (MARKDOWN_ARTIFACT, HTML_ARTIFACT)
    }


def test_split_then_resumed_reports_match_a_straight_run(tmp_path: Path) -> None:
    straight = tmp_path / "straight"
    split = tmp_path / "split"
    Pipeline(config("same"), ArtifactStore(straight, "same")).run()
    Pipeline(config("same", stages=STAGES[:4]), ArtifactStore(split, "same")).run()
    resumed = Pipeline(config("same"), ArtifactStore(split, "same"))
    summary = resumed.run(resume=True)
    assert "report" in summary.executed
    assert read_reports(straight, "same") == read_reports(split, "same")


def test_resuming_twice_leaves_reports_unchanged(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "idem")
    run_config = config("idem")
    Pipeline(run_config, store).run()
    first = read_reports(tmp_path, "idem")
    Pipeline(run_config, store).run(resume=True)
    Pipeline(run_config, store).run(resume=True)
    assert read_reports(tmp_path, "idem") == first
