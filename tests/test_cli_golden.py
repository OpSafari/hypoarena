"""Golden checks over the public CLI surface.

The command registry, its order and the shared flags are the CLI's public
contract; pinning them makes any addition, removal or reorder a deliberate,
reviewed change rather than silent drift.
"""

from __future__ import annotations

import pytest

from hypoarena.cli import COMMANDS, build_parser

EXPECTED_COMMANDS = (
    "corpus",
    "generate",
    "verify",
    "dedup",
    "debate",
    "rank",
    "evolve",
    "accumulate",
    "report",
    "demo",
)

COMMON_FLAGS = ("--seed", "--out", "--run-id", "--chains", "--chain-length")


@pytest.fixture(autouse=True)
def _fixed_width(monkeypatch: pytest.MonkeyPatch) -> None:
    # argparse wraps help to the terminal width; pin it so the text is stable
    monkeypatch.setenv("COLUMNS", "100")


def test_commands_are_registered_in_pipeline_order() -> None:
    assert tuple(spec.name for spec in COMMANDS) == EXPECTED_COMMANDS


def test_every_command_has_non_empty_help() -> None:
    assert all(spec.help.strip() for spec in COMMANDS)


def test_top_level_help_lists_every_command() -> None:
    help_text = build_parser().format_help()
    for name in EXPECTED_COMMANDS:
        assert name in help_text


@pytest.mark.parametrize("name", EXPECTED_COMMANDS)
def test_each_subcommand_help_shows_the_common_flags(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as info:
        parser.parse_args([name, "--help"])
    assert info.value.code == 0
    help_text = capsys.readouterr().out
    for flag in COMMON_FLAGS:
        assert flag in help_text
