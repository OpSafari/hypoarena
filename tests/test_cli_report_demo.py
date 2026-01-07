"""CLI coverage for evolve, accumulate, report and the offline demo."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.cli import main

SMALL = ["--chains", "2", "--chain-length", "2"]


def test_evolve_reports_generations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["evolve", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert code == 0
    assert "evolve:" in out and "generations" in out
    assert (tmp_path / "run" / "evolution.jsonl").exists()


def test_accumulate_reports_beliefs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["accumulate", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert code == 0
    assert "beliefs updated" in out
    assert (tmp_path / "run" / "beliefs.jsonl").exists()


def test_report_writes_all_three_report_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["report", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert code == 0
    assert "report:" in out
    for name in ("report.json", "report.md", "report.html"):
        assert (tmp_path / "run" / name).exists()


def test_demo_prints_recovered_versus_planted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["demo", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert code == 0
    assert "planted links:" in out
    assert "synthetic demonstration only" in out


def test_demo_recovers_every_planted_link_on_the_small_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["demo", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert "rate: 1.0" in out
    assert "MISSING" not in out
