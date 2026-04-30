"""CLI error handling: bad configuration maps to the right exit code.

Domain errors surface as :class:`hypoarena.errors.HypoArenaError` and ``main``
returns that error's ``exit_code`` after printing a message on stderr, so a
script can branch on the category without parsing prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.cli import main


def test_chain_length_below_the_minimum_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["corpus", "--out", str(tmp_path), "--chain-length", "1"])
    assert code == 2
    assert "hypoarena:" in capsys.readouterr().err


def test_chain_length_above_the_maximum_is_rejected(tmp_path: Path) -> None:
    assert main(["corpus", "--out", str(tmp_path), "--chain-length", "99"]) == 2


def test_zero_chains_is_rejected_and_names_the_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["corpus", "--out", str(tmp_path), "--chains", "0"])
    assert code == 2
    assert "chains" in capsys.readouterr().err.lower()


def test_a_bad_run_id_is_rejected_before_any_artifact_is_written(
    tmp_path: Path,
) -> None:
    code = main(["corpus", "--out", str(tmp_path), "--run-id", "../escape"])
    assert code == 2
    assert list(tmp_path.iterdir()) == []
