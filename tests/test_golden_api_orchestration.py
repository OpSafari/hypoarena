"""Golden snapshots of the orchestration modules' public API surface.

Each assertion pins the classes and functions a module *defines* (re-exported
imports are excluded via ``__module__``), so adding, removing or renaming a
public symbol is a deliberate, reviewed change rather than silent drift.
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


def test_config_public_api() -> None:
    from hypoarena import config

    assert defined_public_names(config) == ("RunConfig",)


def test_artifacts_public_api() -> None:
    from hypoarena import artifacts

    assert defined_public_names(artifacts) == (
        "ArtifactStore",
        "RunMetadata",
        "check_artifact_name",
        "check_stage_name",
    )


def test_cost_public_api() -> None:
    from hypoarena import cost

    assert defined_public_names(cost) == ("CostEntry", "CostLedger")


def test_stages_public_api() -> None:
    from hypoarena import stages

    assert defined_public_names(stages) == ("RunSummary", "StageResult")


def test_runner_public_api() -> None:
    from hypoarena import runner

    assert defined_public_names(runner) == (
        "Pipeline",
        "RunState",
        "claim_from_proposal",
        "debate_from_line",
        "debate_line",
        "default_agents",
        "recovered_links",
        "relation_from_statement",
        "run_report",
    )
