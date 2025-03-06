"""Golden snapshot of packaging/tool configuration.

These assertions exist so that dependency, marker, linter and Makefile drift is a
deliberate decision rather than an accident.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def pyproject() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def project() -> dict[str, Any]:
    return pyproject()["project"]


def tool() -> dict[str, Any]:
    return pyproject()["tool"]


def makefile() -> str:
    return (ROOT / "Makefile").read_text(encoding="utf-8")


def test_numpy_is_the_only_runtime_dependency() -> None:
    assert project()["dependencies"] == ["numpy>=1.26"]


def test_torch_is_optional_and_never_a_runtime_dependency() -> None:
    assert "torch" in project()["optional-dependencies"]
    assert not any("torch" in dependency for dependency in project()["dependencies"])


def test_dev_extra_ships_the_build_backend_for_no_isolation_builds() -> None:
    dev = project()["optional-dependencies"]["dev"]
    assert any(name.startswith("hatchling") for name in dev)
    assert any(name.startswith("build") for name in dev)
    assert any(name.startswith("pytest") for name in dev)
    assert any(name.startswith("ruff") for name in dev)
    assert any(name.startswith("mypy") for name in dev)


def test_python_floor_is_311() -> None:
    assert project()["requires-python"] == ">=3.11"


def test_mypy_does_not_pin_python_version() -> None:
    assert "python_version" not in tool()["mypy"]


def test_ruff_line_length_is_the_formatter_default() -> None:
    assert tool()["ruff"]["line-length"] == 88


def test_declared_markers_cover_slow_and_model() -> None:
    markers = tool()["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("slow:") for marker in markers)
    assert any(marker.startswith("model:") for marker in markers)


def test_makefile_declares_every_required_target() -> None:
    for target in (
        "build",
        "test",
        "test-all",
        "format",
        "format-check",
        "lint",
        "typecheck",
        "release",
    ):
        assert f"\n{target}:" in makefile(), target


def test_makefile_recipes_are_pinned() -> None:
    text = makefile()
    assert "\n\t.venv/bin/python -m build --wheel --no-isolation\n" in text
    assert '\n\t.venv/bin/python -m pytest -q -m "not slow"\n' in text
    assert "\n\t.venv/bin/python -m pytest -q\n" in text
    assert (
        "\n\t.venv/bin/ruff format --check src tests examples scripts && "
        ".venv/bin/ruff check src tests examples scripts\n"
    ) in text


def test_test_all_has_no_marker_filter() -> None:
    body = makefile().split("\ntest-all:\n", 1)[1].split("\n\n", 1)[0]
    assert re.search(r"""-m\s+["']""", body) is None


def test_fast_test_target_excludes_slow() -> None:
    body = makefile().split("\ntest:\n", 1)[1].split("\n\n", 1)[0]
    assert body.strip() == '.venv/bin/python -m pytest -q -m "not slow"'
