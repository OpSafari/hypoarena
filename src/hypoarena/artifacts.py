"""Run artifacts: a directory layout, atomic writes and stage checkpoints.

Layout for a run ``<root>/<run_id>/``::

    run.json                 run metadata (see :mod:`hypoarena.artifacts`)
    <stage>.jsonl            one artifact per pipeline stage
    checkpoints/<stage>.done marker files used to resume

Writes go through a temporary file plus ``os.replace``, so an interrupted run
never leaves a half-written artifact behind. Names are validated because they
become file names: a caller-supplied stage name must not be able to escape the
run directory.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from hypoarena.config import (
    STAGES,
)
from hypoarena.errors import (
    ArtifactError,
    ValidationError,
)
from hypoarena.serialize import (
    dumps_line,
    iter_lines,
    loads_line,
    write_lines,
)

ARTIFACT_PATTERN = re.compile(r"[A-Za-z0-9_.-]+\.(?:jsonl|json)")
CHECKPOINT_SUFFIX = ".done"


def check_artifact_name(name: str) -> str:
    """Validate an artifact file name and return it."""
    if not ARTIFACT_PATTERN.fullmatch(name):
        raise ValidationError(
            "artifact names must be flat file names ending in .jsonl or .json",
            name=name,
        )
    return name


def check_stage_name(stage: str) -> str:
    """Validate a pipeline stage name used for checkpoints."""
    if stage not in STAGES:
        raise ValidationError(
            "unknown pipeline stage", stage=stage, allowed=list(STAGES)
        )
    return stage


class ArtifactStore:
    """Reads and writes the artifacts of one run."""

    def __init__(self, root: Path | str, run_id: str) -> None:
        if (
            not run_id.strip()
            or "/" in run_id
            or "\\" in run_id
            or run_id in (".", "..")
        ):
            raise ValidationError("run_id must be a single path segment", run_id=run_id)
        self.run_id = run_id
        self.root = Path(root) / run_id

    @property
    def checkpoint_dir(self) -> Path:
        """Directory holding stage completion markers."""
        return self.root / "checkpoints"

    def path(self, name: str) -> Path:
        """Return the absolute path of one artifact."""
        return self.root / check_artifact_name(name)

    def exists(self, name: str) -> bool:
        """True when an artifact file is present."""
        return self.path(name).is_file()

    def write_lines(self, name: str, lines: Iterable[str]) -> int:
        """Write JSONL lines atomically; return how many were written."""
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        return write_lines(lines, target)

    def read_lines(self, name: str) -> list[str]:
        """Read an artifact's lines, raising when it is missing."""
        if not self.exists(name):
            raise ArtifactError(
                "artifact not found", name=name, path=str(self.path(name))
            )
        return list(iter_lines(self.path(name)))

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        """Write one JSON object as a single canonical line."""
        self.write_lines(name, [dumps_line(payload)])
        return self.path(name)

    def read_json(self, name: str) -> dict[str, Any]:
        """Read one JSON object artifact."""
        lines = self.read_lines(name)
        if len(lines) != 1:
            raise ArtifactError(
                "json artifact must hold exactly one line", name=name, count=len(lines)
            )
        return loads_line(lines[0], field=name)

    def listing(self) -> tuple[str, ...]:
        """Return the artifact names present in this run, sorted."""
        if not self.root.is_dir():
            return ()
        return tuple(
            sorted(path.name for path in self.root.iterdir() if path.is_file())
        )

    def mark_stage(self, stage: str) -> Path:
        """Record that a stage completed."""
        check_stage_name(stage)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        marker = self.checkpoint_dir / f"{stage}{CHECKPOINT_SUFFIX}"
        marker.write_text(f"{stage}\n", encoding="utf-8")
        return marker

    def stage_done(self, stage: str) -> bool:
        """True when a stage has a completion marker."""
        check_stage_name(stage)
        return (self.checkpoint_dir / f"{stage}{CHECKPOINT_SUFFIX}").is_file()

    def clear_stage(self, stage: str) -> bool:
        """Remove one stage marker; return whether it existed."""
        check_stage_name(stage)
        marker = self.checkpoint_dir / f"{stage}{CHECKPOINT_SUFFIX}"
        if not marker.is_file():
            return False
        marker.unlink()
        return True

    def completed_stages(self) -> tuple[str, ...]:
        """Return the stages with markers, in pipeline order."""
        return tuple(stage for stage in STAGES if self.stage_done(stage))

    def clear_checkpoints(self) -> int:
        """Remove every marker; return how many were deleted."""
        return sum(1 for stage in STAGES if self.clear_stage(stage))
