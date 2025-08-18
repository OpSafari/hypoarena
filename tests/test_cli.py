"""CLI surface: version reporting and help output."""

import pytest

from hypoarena._version import __version__
from hypoarena.cli import build_parser, main


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_arguments_prints_help_and_succeeds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    assert "hypoarena" in capsys.readouterr().out


def test_unknown_flag_is_rejected() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--definitely-not-a-flag"])
    assert exit_info.value.code == 2


def test_parser_is_named_hypoarena() -> None:
    assert build_parser().prog == "hypoarena"
