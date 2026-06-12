"""CLI integration: report artifacts and demo output stay consistent.

These cross the CLI, the runner, the artifact store and the report renderers in
one offline pass, checking that what the commands print agrees with what they
write to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypoarena.cli import main

SMALL = ["--chains", "2", "--chain-length", "2", "--seed", "5"]


def test_report_writes_consistent_artifacts(tmp_path: Path) -> None:
    code = main(["report", "--out", str(tmp_path), "--run-id", "r", *SMALL])
    assert code == 0
    report = json.loads((tmp_path / "r" / "report.json").read_text().strip())
    assert report["recovered"]["rate"] == 1.0
    markdown = (tmp_path / "r" / "report.md").read_text()
    html = (tmp_path / "r" / "report.html").read_text()
    assert "## Limitations" in markdown
    assert "<script" not in html


def test_demo_output_matches_the_report_recovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["demo", "--out", str(tmp_path), "--run-id", "d", *SMALL])
    out = capsys.readouterr().out
    report = json.loads((tmp_path / "d" / "report.json").read_text().strip())
    recovered = report["recovered"]
    assert f"planted links: {recovered['planted']}" in out
    assert f"recovered: {recovered['recovered']}" in out
    assert "MISSING" not in out
