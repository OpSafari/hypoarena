"""write_reports emits both artifacts atomically and deterministically."""

from __future__ import annotations

from pathlib import Path

from hypoarena.artifacts import ArtifactStore
from hypoarena.reports import (
    HTML_ARTIFACT,
    MARKDOWN_ARTIFACT,
    render_html,
    render_markdown,
    write_reports,
)
from report_fixture import sample_report


def test_write_reports_returns_both_artifact_names(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run")
    assert write_reports(store, sample_report()) == (MARKDOWN_ARTIFACT, HTML_ARTIFACT)
    assert store.exists(MARKDOWN_ARTIFACT)
    assert store.exists(HTML_ARTIFACT)


def test_written_files_match_the_renderers(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run")
    report = sample_report()
    write_reports(store, report)
    assert store.read_text(MARKDOWN_ARTIFACT) == render_markdown(report)
    assert store.read_text(HTML_ARTIFACT) == render_html(report)


def test_write_reports_is_deterministic_across_stores(tmp_path: Path) -> None:
    report = sample_report()
    first = ArtifactStore(tmp_path / "a", "run")
    second = ArtifactStore(tmp_path / "b", "run")
    write_reports(first, report)
    write_reports(second, report)
    assert first.read_text(MARKDOWN_ARTIFACT) == second.read_text(MARKDOWN_ARTIFACT)
    assert first.read_text(HTML_ARTIFACT) == second.read_text(HTML_ARTIFACT)


def test_write_reports_leaves_no_partial_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run")
    write_reports(store, sample_report())
    assert list((tmp_path / "run").glob("*.partial")) == []
