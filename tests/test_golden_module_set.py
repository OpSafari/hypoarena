"""Golden over the package's public module set and clean imports.

The set of importable submodules is part of the public surface; pinning it means
a new, removed or renamed module is deliberate. Core modules must import without
the optional torch extra, while ``ranker_torch`` imports only when torch is
present.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import hypoarena

EXPECTED_MODULES = (
    "agents",
    "artifacts",
    "belief",
    "cli",
    "codec",
    "config",
    "corpus",
    "cost",
    "debate",
    "dedup",
    "errors",
    "evolve",
    "graph",
    "grounding",
    "http_agent",
    "ids",
    "ranker",
    "ranker_torch",
    "reports",
    "runner",
    "schema",
    "serialize",
    "stages",
    "synthetic",
    "text",
    "tournament",
)

CORE_MODULES = tuple(name for name in EXPECTED_MODULES if name != "ranker_torch")


def public_modules() -> tuple[str, ...]:
    return tuple(
        sorted(
            module.name
            for module in pkgutil.iter_modules(hypoarena.__path__)
            if not module.name.startswith("_")
        )
    )


def test_the_public_module_set_is_pinned() -> None:
    assert public_modules() == EXPECTED_MODULES


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_submodules_import_without_torch(name: str) -> None:
    assert importlib.import_module(f"hypoarena.{name}") is not None


def test_ranker_torch_imports_when_torch_is_present() -> None:
    pytest.importorskip("torch", reason="optional torch extra not installed")
    assert importlib.import_module("hypoarena.ranker_torch") is not None
