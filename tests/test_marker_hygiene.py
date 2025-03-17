"""Markers used in the suite must be declared in pyproject.toml.

A misspelled marker name would otherwise be silently ignored, letting a slow test
run inside the fast suite; the drift is caught here instead.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILTIN_MARKERS = {
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
}
MARKER_PATTERN = re.compile(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")


def declared_markers() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        entries = tomllib.load(handle)["tool"]["pytest"]["ini_options"]["markers"]
    return {entry.split(":", 1)[0]: entry for entry in entries}


def used_markers() -> set[str]:
    used: set[str] = set()
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        used.update(MARKER_PATTERN.findall(path.read_text(encoding="utf-8")))
    return used - BUILTIN_MARKERS


def test_every_used_marker_is_declared() -> None:
    assert used_markers() <= set(declared_markers())


def test_declared_markers_have_a_description() -> None:
    for name, entry in declared_markers().items():
        assert entry.startswith(f"{name}: "), entry
        assert len(entry.split(":", 1)[1].strip()) > 10, entry
