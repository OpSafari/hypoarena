"""Pipeline orchestration: stages, artifacts, checkpoints and accounting.

The runner is the only place that knows the *order* of the discovery loop. Each
stage reads the run state, writes one JSONL artifact and marks a checkpoint, so a
run can be stopped and resumed without recomputing finished work — and, because
nothing reads the clock and every iteration order is sorted, a resumed run
produces byte-identical artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from hypoarena.agents import (
    ScriptedAgent,
)
from hypoarena.artifacts import (
    ArtifactStore,
    RunMetadata,
)
from hypoarena.config import (
    RunConfig,
)
from hypoarena.cost import (
    CostLedger,
)
from hypoarena.errors import (
    ConfigError,
    ValidationError,
)
from hypoarena.graph import (
    HypothesisGraph,
)
from hypoarena.ids import (
    content_hash,
)

DEFAULT_AGENT_QUALITIES: tuple[float, ...] = (0.9, 0.5, 0.2)
SUMMARY_ARTIFACT = "summary.json"


def default_agents(seed: int) -> tuple[ScriptedAgent, ...]:
    """Return the offline agent set a run uses when none is supplied.

    One scripted agent per quality tier, named so audit trails and the cost
    ledger can tell them apart. Nothing here contacts a service.
    """
    return tuple(
        ScriptedAgent(f"scripted-{quality:.1f}", quality=quality)
        for quality in DEFAULT_AGENT_QUALITIES
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


@dataclass
class RunState:
    """Mutable state handed from stage to stage within one run."""

    corpus_hash: str = ""
    graph: HypothesisGraph = field(default_factory=HypothesisGraph)
    candidates: tuple[object, ...] = ()
    reports: tuple[object, ...] = ()
    dedup: object | None = None
    debates: tuple[object, ...] = ()
    tournament: object | None = None
    evolution: tuple[object, ...] = ()
    beliefs: tuple[object, ...] = ()
    truth: object | None = None
    corpus: object | None = None


class Pipeline:
    """Runs the configured stages in order against an artifact store."""

    def __init__(
        self,
        config: RunConfig,
        store: ArtifactStore,
        *,
        ledger: CostLedger | None = None,
        agents: Sequence[ScriptedAgent] | None = None,
    ) -> None:
        if store.run_id != config.run_id:
            raise ConfigError(
                "artifact store and run configuration disagree",
                store=store.run_id,
                config=config.run_id,
            )
        self.config = config
        self.store = store
        self.ledger = ledger or CostLedger()
        self.agents = (
            tuple(agents) if agents is not None else default_agents(config.seed)
        )
        self.state = RunState()
        if not self.agents:
            raise ConfigError("a pipeline needs at least one agent")

    def handlers(self) -> dict[str, Callable[[], StageResult]]:
        """Return the stage implementations this pipeline knows."""
        return {}

    def run(self, *, resume: bool = False) -> RunSummary:
        """Execute the configured stages and write the run summary."""
        results: list[StageResult] = []
        handlers = self.handlers()
        for stage in self.config.stages:
            if stage not in handlers:
                raise ConfigError("stage has no implementation", stage=stage)
            if resume and self.store.stage_done(stage):
                self.restore(stage)
                results.append(StageResult(stage, 0, (), skipped=True))
                continue
            results.append(handlers[stage]())
            self.store.mark_stage(stage)
        summary = RunSummary(
            run_id=self.config.run_id,
            config_fingerprint=self.config.fingerprint(),
            stages=tuple(results),
            completed=True,
        )
        self.store.write_metadata(RunMetadata.from_config(self.config))
        self.store.write_json(SUMMARY_ARTIFACT, summary.as_dict())
        return summary

    def restore(self, stage: str) -> None:
        """Reload the state a completed stage produced (used when resuming)."""
        raise NotImplementedError(f"{type(self).__name__} must implement restore()")
