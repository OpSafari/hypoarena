"""The --resume flag reuses checkpoints instead of re-running stages."""

from __future__ import annotations

from pathlib import Path

from hypoarena.cli import main

SMALL = ["--chains", "2", "--chain-length", "2", "--seed", "5"]


def test_resume_flag_is_accepted_by_a_stage_command(tmp_path: Path) -> None:
    code = main(["corpus", "--out", str(tmp_path), "--run-id", "c", "--resume", *SMALL])
    assert code == 0
    assert (tmp_path / "c" / "corpus.jsonl").exists()


def test_a_resumed_report_run_keeps_the_report_identical(tmp_path: Path) -> None:
    out = str(tmp_path)
    main(["report", "--out", out, "--run-id", "r", *SMALL])
    first = (tmp_path / "r" / "report.json").read_bytes()
    # every stage is checkpointed, so the resumed run skips them all
    main(["report", "--out", out, "--run-id", "r", "--resume", *SMALL])
    second = (tmp_path / "r" / "report.json").read_bytes()
    assert first == second


def test_resume_without_prior_checkpoints_still_runs(tmp_path: Path) -> None:
    code = main(
        ["verify", "--out", str(tmp_path), "--run-id", "fresh", "--resume", *SMALL]
    )
    assert code == 0
    assert (tmp_path / "fresh" / "grounding.jsonl").exists()
