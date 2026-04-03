"""CLI subcommands drive the offline pipeline and print short summaries.

Every command runs on a tiny synthetic corpus in a temporary artifact root, so
these tests are fully offline and fast. They assert on printed output and on the
artifacts each stage prefix is (and is not) allowed to produce.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.cli import main

SMALL = ["--chains", "2", "--chain-length", "2"]


def test_corpus_command_reports_document_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["corpus", "--out", str(tmp_path), *SMALL, "--seed", "5"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("corpus:")
    assert "documents" in out
    assert (tmp_path / "run" / "corpus.jsonl").exists()


def test_generate_command_reports_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["generate", "--out", str(tmp_path), *SMALL])
    out = capsys.readouterr().out
    assert code == 0
    assert "candidate claims proposed" in out
    assert (tmp_path / "run" / "candidates.jsonl").exists()


def test_a_stage_prefix_does_not_run_downstream_stages(tmp_path: Path) -> None:
    main(["corpus", "--out", str(tmp_path), *SMALL])
    assert (tmp_path / "run" / "corpus.jsonl").exists()
    assert not (tmp_path / "run" / "tournament.jsonl").exists()
    assert not (tmp_path / "run" / "report.json").exists()


def test_run_id_flag_controls_the_artifact_directory(tmp_path: Path) -> None:
    main(["corpus", "--out", str(tmp_path), "--run-id", "custom", *SMALL])
    assert (tmp_path / "custom" / "corpus.jsonl").exists()


def test_invalid_run_id_maps_to_the_validation_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["corpus", "--out", str(tmp_path), "--run-id", "a/b", *SMALL])
    assert code == 2
    assert "hypoarena:" in capsys.readouterr().err


def test_unknown_subcommand_is_an_argparse_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as info:
        main(["not-a-command", "--out", str(tmp_path)])
    assert info.value.code == 2
