"""Pipeline orchestration: stages, artifacts, checkpoints and accounting.

The runner is the only place that knows the *order* of the discovery loop. Each
stage reads the run state, writes one JSONL artifact and marks a checkpoint, so a
run can be stopped and resumed without recomputing finished work — and, because
nothing reads the clock and every iteration order is sorted, a resumed run
produces byte-identical artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from hypoarena.agents import (
    ScriptedAgent,
    Usage,
)
from hypoarena.artifacts import (
    ArtifactStore,
    RunMetadata,
)
from hypoarena.belief import (
    BeliefState,
)
from hypoarena.config import (
    RunConfig,
)
from hypoarena.corpus import (
    Corpus,
)
from hypoarena.cost import (
    CostLedger,
)
from hypoarena.debate import (
    DebateResult,
    corpus_context,
)
from hypoarena.dedup import (
    DedupReport,
)
from hypoarena.errors import (
    ConfigError,
    ValidationError,
)
from hypoarena.evolve import (
    EvolutionStep,
)
from hypoarena.graph import (
    HypothesisGraph,
)
from hypoarena.grounding import (
    GroundingReport,
)
from hypoarena.ids import (
    content_hash,
    make_id,
)
from hypoarena.schema import (
    Claim,
    PredictedRelation,
    Provenance,
    Scope,
)
from hypoarena.serialize import (
    claim_from_line,
    claim_to_line,
    corpus_from_text,
    corpus_to_lines,
    graph_from_lines,
    graph_to_lines,
    truth_from_lines,
    truth_to_lines,
)
from hypoarena.synthetic import (
    PlantedTruth,
    SyntheticBundle,
    build_bundle,
)
from hypoarena.text import (
    content_tokens,
    normalize,
)
from hypoarena.tournament import (
    TournamentResult,
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
    corpus: Corpus | None = None
    truth: PlantedTruth | None = None
    bundle: SyntheticBundle | None = None
    candidates: tuple[Claim, ...] = ()
    reports: tuple[GroundingReport, ...] = ()
    dedup: DedupReport | None = None
    debates: tuple[DebateResult, ...] = ()
    tournament: TournamentResult | None = None
    evolution: tuple[EvolutionStep, ...] = ()
    beliefs: tuple[BeliefState, ...] = ()


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
        self.usage_snapshot: dict[str, Usage] = {}
        if not self.agents:
            raise ConfigError("a pipeline needs at least one agent")

    def handlers(self) -> dict[str, Callable[[], StageResult]]:
        """Return the stage implementations this pipeline knows."""
        return {
            "corpus": self.stage_corpus,
            "generate": self.stage_generate,
        }

    def record_usage(self, stage: str) -> None:
        """Record each agent's usage *delta* for one stage.

        Agents are reused across stages, so recording their cumulative counters
        would double count. The pipeline keeps a snapshot per agent and books only
        what changed since the previous stage.
        """
        for agent in self.agents:
            previous = self.usage_snapshot.get(agent.name, Usage())
            calls = agent.usage.calls - previous.calls
            if calls <= 0:
                self.usage_snapshot[agent.name] = Usage(
                    agent.usage.calls,
                    agent.usage.prompt_tokens,
                    agent.usage.completion_tokens,
                    dict(agent.usage.by_task),
                )
                continue
            delta = Usage(
                calls=calls,
                prompt_tokens=agent.usage.prompt_tokens - previous.prompt_tokens,
                completion_tokens=(
                    agent.usage.completion_tokens - previous.completion_tokens
                ),
                by_task={
                    task: count - previous.by_task.get(task, 0)
                    for task, count in agent.usage.by_task.items()
                    if count - previous.by_task.get(task, 0) > 0
                },
            )
            self.usage_snapshot[agent.name] = Usage(
                agent.usage.calls,
                agent.usage.prompt_tokens,
                agent.usage.completion_tokens,
                dict(agent.usage.by_task),
            )
            self.ledger.record(stage, agent.name, delta)

    def stage_corpus(self) -> StageResult:
        """Generate the synthetic corpus and the graph derived from it."""
        bundle = build_bundle(replace(self.config.corpus, seed=self.config.seed))
        self.state.bundle = bundle
        self.state.corpus = bundle.corpus
        self.state.truth = bundle.truth
        self.state.graph = bundle.graph()
        self.state.corpus_hash = bundle.corpus_hash
        self.store.write_lines("corpus.jsonl", corpus_to_lines(bundle.corpus))
        self.store.write_lines("truth.jsonl", truth_to_lines(bundle.truth))
        self.store.write_lines("graph.jsonl", graph_to_lines(self.state.graph))
        return StageResult("corpus", len(bundle.corpus), CORPUS_ARTIFACTS)

    def stage_generate(self) -> StageResult:
        """Collect one proposal per agent and add the usable ones to the graph."""
        if self.state.corpus is None:
            raise ConfigError("generate needs the corpus stage to have run")
        context = corpus_context(self.state.corpus, limit=GENERATE_CONTEXT_LINES)
        candidates: list[Claim] = []
        for agent in self.agents:
            response = agent.propose(self.config.debate.proposal_prompt, context)
            claim = claim_from_proposal(
                response.text, agent=agent.name, seed=self.config.seed
            )
            if claim is None or self.state.graph.has_claim(claim.claim_id):
                continue
            candidates.append(claim)
            self.state.graph.add_claim(claim)
        self.record_usage("generate")
        self.state.candidates = tuple(candidates)
        self.store.write_lines(
            CANDIDATE_ARTIFACT, [claim_to_line(claim) for claim in candidates]
        )
        self.store.write_lines("graph.jsonl", graph_to_lines(self.state.graph))
        return StageResult(
            "generate", len(candidates), (CANDIDATE_ARTIFACT, "graph.jsonl")
        )

    def restore_corpus(self) -> None:
        """Reload corpus, truth and graph artifacts after a resume."""
        corpus = corpus_from_text("".join(self.store.read_lines("corpus.jsonl")))
        self.state.corpus = corpus
        self.state.truth = truth_from_lines(self.store.read_lines("truth.jsonl"))
        self.state.graph = graph_from_lines(self.store.read_lines("graph.jsonl"))
        self.state.corpus_hash = corpus.signature()

    def restore_generate(self) -> None:
        """Reload the candidate claims and the graph they were added to."""
        self.state.candidates = tuple(
            claim_from_line(line) for line in self.store.read_lines(CANDIDATE_ARTIFACT)
        )
        self.state.graph = graph_from_lines(self.store.read_lines("graph.jsonl"))

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
        loaders: dict[str, Callable[[], None]] = {
            "corpus": self.restore_corpus,
            "generate": self.restore_generate,
        }
        loader = loaders.get(stage)
        if loader is None:
            raise ConfigError("stage has no restore implementation", stage=stage)
        loader()


CORPUS_ARTIFACTS = ("corpus.jsonl", "truth.jsonl", "graph.jsonl")
CANDIDATE_ARTIFACT = "candidates.jsonl"
PROPOSAL_SCOPE = "synthetic corpus"
RELATION_KEYWORDS: tuple[tuple[str, PredictedRelation], ...] = (
    ("increases", PredictedRelation.INCREASES),
    ("upregulates", PredictedRelation.INCREASES),
    ("elevates", PredictedRelation.INCREASES),
    ("decreases", PredictedRelation.DECREASES),
    ("reduces", PredictedRelation.DECREASES),
    ("enables", PredictedRelation.ENABLES),
    ("is required for", PredictedRelation.ENABLES),
    ("inhibits", PredictedRelation.INHIBITS),
    ("blocks", PredictedRelation.INHIBITS),
    ("causes", PredictedRelation.CAUSES),
    ("triggers", PredictedRelation.CAUSES),
)


def relation_from_statement(statement: str) -> PredictedRelation:
    """Map a statement's wording onto a predicted relation.

    This is a documented keyword heuristic, not a parser: the first matching
    keyword wins, and wording that matches nothing stays ``associates`` rather
    than guessing a direction.
    """
    lowered = normalize(statement)
    for keyword, relation in RELATION_KEYWORDS:
        if keyword in lowered:
            return relation
    return PredictedRelation.ASSOCIATES


def claim_from_proposal(statement: str, *, agent: str, seed: int) -> Claim | None:
    """Turn one agent proposal into a claim, or ``None`` when it is unusable.

    The variables are the first and last distinct content tokens of the
    proposal — a deliberate, inspectable choice. Proposals carry no citations, so
    they enter the graph ungrounded and the verify stage is what grades them.
    """
    tokens = list(dict.fromkeys(content_tokens(statement)))
    if len(tokens) < 2:
        return None
    subject, target = tokens[0], tokens[-1]
    if subject == target:
        return None
    return Claim(
        claim_id=make_id("clm", "proposed", seed, agent, statement),
        statement=" ".join(statement.split()),
        subject=subject,
        object=target,
        relation=relation_from_statement(statement),
        scope=Scope(population=PROPOSAL_SCOPE),
        citations=(),
        provenance=Provenance(
            origin="agent",
            agent_id=make_id("agt", agent),
            seed=seed,
            notes="proposed",
        ),
    )


GENERATE_CONTEXT_LINES = 6
