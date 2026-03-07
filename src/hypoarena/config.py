"""Run configuration: one object that fully determines a pipeline run.

A run is reproducible when its configuration is: every knob that can change an
artifact lives here, and :meth:`RunConfig.fingerprint` digests all of them so two
runs can be compared by hash instead of by diffing directories. Nothing in this
module reads the clock or the environment, which is what lets a resumed run
produce byte-identical artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace

from hypoarena.belief import (
    BeliefConfig,
)
from hypoarena.debate import (
    DebateConfig,
)
from hypoarena.dedup import (
    DedupConfig,
)
from hypoarena.errors import (
    ValidationError,
)
from hypoarena.grounding import (
    VerifierConfig,
)
from hypoarena.ids import (
    content_hash,
)
from hypoarena.synthetic import (
    SyntheticConfig,
)
from hypoarena.tournament import (
    TournamentConfig,
)

STAGES: tuple[str, ...] = (
    "corpus",
    "generate",
    "verify",
    "dedup",
    "debate",
    "rank",
    "evolve",
    "accumulate",
    "report",
)
DEFAULT_RUN_ID = "run"


@dataclass(frozen=True)
class RunConfig:
    """Everything a pipeline run depends on."""

    seed: int = 270106
    stages: tuple[str, ...] = STAGES
    run_id: str = DEFAULT_RUN_ID
    corpus: SyntheticConfig = field(default_factory=SyntheticConfig)
    grounding: VerifierConfig = field(default_factory=VerifierConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    tournament: TournamentConfig = field(default_factory=TournamentConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValidationError("a run needs at least one stage")
        unknown = [stage for stage in self.stages if stage not in STAGES]
        if unknown:
            raise ValidationError(
                "unknown pipeline stages", unknown=unknown, allowed=list(STAGES)
            )
        if len(set(self.stages)) != len(self.stages):
            raise ValidationError(
                "pipeline stages contain duplicates", stages=list(self.stages)
            )
        if not self.run_id.strip():
            raise ValidationError("run_id must not be blank")
        if "/" in self.run_id or "\\" in self.run_id or self.run_id in (".", ".."):
            raise ValidationError(
                "run_id must be a single path segment", run_id=self.run_id
            )

    def stage_index(self, stage: str) -> int:
        """Return the position of a stage in this run's order."""
        try:
            return self.stages.index(stage)
        except ValueError:
            raise ValidationError(
                "stage is not part of this run", stage=stage, stages=list(self.stages)
            ) from None

    def stages_from(self, stage: str) -> tuple[str, ...]:
        """Return this stage and everything after it."""
        return self.stages[self.stage_index(stage) :]

    def with_stages(self, stages: Sequence[str]) -> RunConfig:
        """Return a copy restricted to the given stages."""
        return replace(self, stages=tuple(stages))

    def fingerprint(self) -> str:
        """Return a digest covering every setting that affects artifacts."""
        return content_hash(
            {
                "seed": self.seed,
                "stages": list(self.stages),
                "run_id": self.run_id,
                "corpus": content_hash(asdict(self.corpus)),
                "grounding": self.grounding.fingerprint(),
                "dedup": self.dedup.fingerprint(),
                "debate": self.debate.fingerprint(),
                "tournament": self.tournament.fingerprint(),
                "belief": self.belief.fingerprint(),
            }
        )
