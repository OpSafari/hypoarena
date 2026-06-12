"""Every subcommand completes on a tiny corpus (heavy integration sweep).

Running all ten subcommands each drives a real pipeline prefix, so this sweep is
marked ``slow`` and runs in ``test-all`` rather than the per-commit fast suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.cli import COMMANDS, main

SMALL = ["--chains", "2", "--chain-length", "2", "--seed", "3"]


@pytest.mark.slow
@pytest.mark.parametrize("name", [spec.name for spec in COMMANDS])
def test_every_subcommand_exits_zero(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([name, "--out", str(tmp_path / name), "--run-id", "run", *SMALL])
    assert code == 0
    assert capsys.readouterr().out.strip()
