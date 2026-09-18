"""A resumed CLI run completes a partial run from its checkpoints.

This is the CLI-level counterpart of the runner resume guarantees: stopping after
an early stage and later resuming a full run must finish the pipeline and produce
a report byte-identical to a straight full run.
"""

from __future__ import annotations

from pathlib import Path

from hypoarena.cli import main

SMALL = ["--chains", "2", "--chain-length", "2", "--seed", "5"]


def test_resume_completes_a_partial_run(tmp_path: Path) -> None:
    out = str(tmp_path)
    assert main(["generate", "--out", out, "--run-id", "p", *SMALL]) == 0
    assert (tmp_path / "p" / "candidates.jsonl").exists()
    assert not (tmp_path / "p" / "report.json").exists()
    assert main(["report", "--out", out, "--run-id", "p", "--resume", *SMALL]) == 0
    assert (tmp_path / "p" / "report.json").exists()
    assert (tmp_path / "p" / "report.md").exists()


def test_a_resumed_full_run_matches_a_straight_full_run(tmp_path: Path) -> None:
    straight = tmp_path / "straight"
    split = tmp_path / "split"
    main(["report", "--out", str(straight), "--run-id", "r", *SMALL])
    main(["dedup", "--out", str(split), "--run-id", "r", *SMALL])
    main(["report", "--out", str(split), "--run-id", "r", "--resume", *SMALL])
    straight_report = (straight / "r" / "report.json").read_bytes()
    split_report = (split / "r" / "report.json").read_bytes()
    assert straight_report == split_report
