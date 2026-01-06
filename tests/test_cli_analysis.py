"""CLI coverage for the verify, dedup, debate and rank subcommands."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.cli import main

SMALL = ["--chains", "2", "--chain-length", "2"]


def test_verify_reports_grounding_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["verify", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert code == 0
    assert "verify:" in out
    assert "grounded_rate=" in out
    assert (tmp_path / "run" / "grounding.jsonl").exists()


def test_dedup_reports_clusters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["dedup", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert code == 0
    assert "dedup:" in out and "clusters" in out
    assert (tmp_path / "run" / "dedup.jsonl").exists()


def test_debate_reports_revised_claims(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["debate", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert code == 0
    assert "debate:" in out
    assert (tmp_path / "run" / "debates.jsonl").exists()


def test_rank_prints_standings_with_elo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["rank", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert code == 0
    assert "matches over" in out
    assert "elo=" in out
    assert (tmp_path / "run" / "tournament.jsonl").exists()
