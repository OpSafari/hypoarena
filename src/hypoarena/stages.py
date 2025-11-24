"""Records describing what a pipeline stage did and how a run ended.

These live apart from :mod:`hypoarena.runner` so the serialization layer can
decode them without importing the pipeline — a dependency cycle between the two
would make both unusable.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
)


@dataclass(frozen=True)
class StageResult:
    """What one stage did."""

    stage: str
    records: int
    artifacts: tuple[str, ...]
    skipped: bool = False

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValidationError("stage result needs a stage name")
        if self.records < 0:
            raise ValidationError("records must be >= 0", records=self.records)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view."""
        return {
            "stage": self.stage,
            "records": self.records,
            "artifacts": list(self.artifacts),
            "skipped": self.skipped,
        }


@dataclass(frozen=True)
class RunSummary:
    """The outcome of a whole run."""

    run_id: str
    config_fingerprint: str
    stages: tuple[StageResult, ...]
    completed: bool

    @property
    def executed(self) -> tuple[str, ...]:
        """Stages that actually ran (not skipped by a resume)."""
        return tuple(stage.stage for stage in self.stages if not stage.skipped)

    @property
    def skipped(self) -> tuple[str, ...]:
        """Stages skipped because a checkpoint existed."""
        return tuple(stage.stage for stage in self.stages if stage.skipped)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view."""
        return {
            "run_id": self.run_id,
            "config_fingerprint": self.config_fingerprint,
            "completed": self.completed,
            "stages": [stage.as_dict() for stage in self.stages],
            "executed": list(self.executed),
            "skipped": list(self.skipped),
        }

    def signature(self) -> str:
        """Return a digest over the summary."""
        return content_hash(self.as_dict())
