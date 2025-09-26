"""Version metadata must agree across pyproject.toml, _version.py and CHANGELOG."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from hypoarena._version import VERSION_TUPLE, __version__

ROOT = Path(__file__).resolve().parents[1]


def project_table() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def changelog_headings() -> list[str]:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.startswith("## ")]


def test_pyproject_version_matches_module() -> None:
    assert project_table()["version"] == __version__


def test_version_tuple_matches_version_string() -> None:
    joined = ".".join(str(part) for part in VERSION_TUPLE)
    assert joined == __version__


def test_changelog_documents_the_current_version() -> None:
    assert f"## {__version__}" in changelog_headings()


def test_changelog_lists_the_current_version_first() -> None:
    assert changelog_headings()[0] == f"## {__version__}"


def test_console_script_points_at_the_cli_module() -> None:
    scripts = project_table()["scripts"]
    assert scripts == {"hypoarena": "hypoarena.cli:main"}
