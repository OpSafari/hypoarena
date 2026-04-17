"""Golden snapshots of the interface modules' public API (reports, cli).

These two modules are the human-facing surface: reports renders documents and
cli dispatches subcommands. Pinning their public symbols keeps the rendered API
and the command registry from drifting by accident.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any


def defined_public_names(module: Any) -> tuple[str, ...]:
    """Return the sorted public classes/functions defined in ``module``."""
    names = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if isinstance(obj, ModuleType):
            continue
        defined_here = getattr(obj, "__module__", None) == module.__name__
        if (isinstance(obj, type) or callable(obj)) and defined_here:
            names.append(name)
    return tuple(sorted(names))


def test_reports_public_api() -> None:
    from hypoarena import reports

    assert defined_public_names(reports) == (
        "beliefs_table",
        "cost_table",
        "counts_table",
        "dedup_table",
        "escape_html",
        "escape_markdown_cell",
        "evolution_table",
        "grounding_table",
        "html_table",
        "markdown_table",
        "ranking_table",
        "recovered_links",
        "recovery_table",
        "render_html",
        "render_markdown",
        "write_reports",
    )


def test_cli_public_api() -> None:
    from hypoarena import cli

    assert defined_public_names(cli) == ("CommandSpec", "build_parser", "main")
