"""The report stage emits rendered Markdown and HTML alongside the JSON."""

from __future__ import annotations

from pathlib import Path

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.reports import (
    HTML_ARTIFACT,
    MARKDOWN_ARTIFACT,
    render_html,
    render_markdown,
)
from hypoarena.runner import REPORT_ARTIFACT, Pipeline
from hypoarena.stages import RunSummary
from hypoarena.synthetic import SyntheticConfig


def run_pipeline(tmp_path: Path, run_id: str) -> tuple[Pipeline, RunSummary]:
    config = RunConfig(
        seed=17,
        run_id=run_id,
        stages=STAGES,
        corpus=SyntheticConfig(seed=17, chains=2, chain_length=2),
    )
    pipeline = Pipeline(config, ArtifactStore(tmp_path / run_id, run_id))
    return pipeline, pipeline.run()


def test_report_stage_writes_all_three_artifacts(tmp_path: Path) -> None:
    pipeline, _ = run_pipeline(tmp_path, "reports")
    assert pipeline.store.exists(REPORT_ARTIFACT)
    assert pipeline.store.exists(MARKDOWN_ARTIFACT)
    assert pipeline.store.exists(HTML_ARTIFACT)


def test_rendered_files_match_the_stored_payload(tmp_path: Path) -> None:
    pipeline, _ = run_pipeline(tmp_path, "match")
    payload = pipeline.store.read_json(REPORT_ARTIFACT)
    assert pipeline.store.read_text(MARKDOWN_ARTIFACT) == render_markdown(payload)
    assert pipeline.store.read_text(HTML_ARTIFACT) == render_html(payload)


def test_report_stage_result_lists_all_three_artifacts(tmp_path: Path) -> None:
    _, summary = run_pipeline(tmp_path, "listing")
    report = next(stage for stage in summary.stages if stage.stage == "report")
    assert set(report.artifacts) == {REPORT_ARTIFACT, MARKDOWN_ARTIFACT, HTML_ARTIFACT}


def test_rendered_reports_carry_the_limitations(tmp_path: Path) -> None:
    pipeline, _ = run_pipeline(tmp_path, "honest")
    assert "## Limitations" in pipeline.store.read_text(MARKDOWN_ARTIFACT)
    assert "Limitations" in pipeline.store.read_text(HTML_ARTIFACT)
    assert "synthetic" in pipeline.store.read_text(MARKDOWN_ARTIFACT).lower()
