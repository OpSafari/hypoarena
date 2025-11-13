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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hypoarena._version import __version__
from hypoarena.config import (
    STAGES,
    RunConfig,
)
from hypoarena.errors import (
    ArtifactError,
    ValidationError,
)
from hypoarena.schema import (
    SCHEMA_VERSION,
)
from hypoarena.serialize import (
    dumps_line,
    iter_lines,
    loads_line,
    write_lines,
)

ARTIFACT_PATTERN = re.compile(r"[A-Za-z0-9_.-]+\.(?:jsonl|json)")
METADATA_ARTIFACT = "run.json"
METADATA_KEYS = (
    "run_id",
    "package_version",
    "schema_version",
    "config_fingerprint",
    "seed",
    "stages",
    "created_at",
)
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

    def write_metadata(self, metadata: RunMetadata) -> Path:
        """Write the run metadata document."""
        return self.write_json(METADATA_ARTIFACT, metadata.as_dict())

    def read_metadata(self) -> RunMetadata:
        """Read the run metadata document."""
        return RunMetadata.from_dict(self.read_json(METADATA_ARTIFACT))

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


@dataclass(frozen=True)
class RunMetadata:
    """What has to be recorded to interpret a run's artifacts later.

    ``created_at`` is *never* read from the clock: a caller that wants a
    timestamp supplies one. Leaving it out is what makes two runs of the same
    configuration produce byte-identical artifacts, which the resume tests rely
    on.
    """

    run_id: str
    package_version: str
    schema_version: str
    config_fingerprint: str
    seed: int
    stages: tuple[str, ...]
    created_at: str | None = None

    @classmethod
    def from_config(
        cls, config: RunConfig, *, created_at: str | None = None
    ) -> RunMetadata:
        """Build metadata for a configuration."""
        return cls(
            run_id=config.run_id,
            package_version=__version__,
            schema_version=SCHEMA_VERSION,
            config_fingerprint=config.fingerprint(),
            seed=config.seed,
            stages=config.stages,
            created_at=created_at,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready view."""
        return {
            "run_id": self.run_id,
            "package_version": self.package_version,
            "schema_version": self.schema_version,
            "config_fingerprint": self.config_fingerprint,
            "seed": self.seed,
            "stages": list(self.stages),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunMetadata:
        """Rebuild metadata from its JSON view, rejecting unknown keys."""
        unknown = sorted(set(payload) - set(METADATA_KEYS))
        if unknown:
            raise ValidationError("unknown run metadata keys", unknown=unknown)
        return cls(
            run_id=str(payload["run_id"]),
            package_version=str(payload["package_version"]),
            schema_version=str(payload["schema_version"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            seed=int(payload["seed"]),
            stages=tuple(str(stage) for stage in payload["stages"]),
            created_at=(
                None
                if payload.get("created_at") is None
                else str(payload["created_at"])
            ),
        )
