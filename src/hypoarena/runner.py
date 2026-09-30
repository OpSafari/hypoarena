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
from typing import Any

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
    accumulate_graph,
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
    DebateLoop,
    DebateResult,
    apply_debate,
    corpus_context,
)
from hypoarena.dedup import (
    DedupReport,
    DuplicateFinder,
)
from hypoarena.errors import (
    ConfigError,
)
from hypoarena.evolve import (
    EvolutionStep,
)
from hypoarena.graph import (
    HypothesisGraph,
)
from hypoarena.grounding import (
    GroundingReport,
    GroundingVerifier,
    summarize_reports,
)
from hypoarena.ids import make_id
from hypoarena.reports import write_reports
from hypoarena.schema import (
    Claim,
    PredictedRelation,
    Provenance,
    Scope,
)
from hypoarena.serialize import (
    belief_from_lines,
    belief_to_lines,
    claim_from_line,
    claim_to_line,
    corpus_from_text,
    corpus_to_lines,
    cost_ledger_from_dict,
    cost_ledger_to_dict,
    debate_result_from_dict,
    debate_result_to_dict,
    dedup_report_from_lines,
    dedup_report_to_lines,
    dumps_line,
    evolution_from_lines,
    evolution_to_lines,
    graph_from_lines,
    graph_to_lines,
    grounding_reports_from_lines,
    grounding_reports_to_lines,
    loads_line,
    tournament_from_lines,
    tournament_to_lines,
    truth_from_lines,
    truth_to_lines,
)
from hypoarena.stages import (
    RunSummary,
    StageResult,
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
    FeatureJudge,
    Tournament,
    TournamentResult,
    graph_contradiction_counts,
)

DEFAULT_AGENT_QUALITIES: tuple[float, ...] = (0.9, 0.5, 0.2)
SUMMARY_ARTIFACT = "summary.json"
COST_ARTIFACT = "cost.json"


def default_agents(seed: int) -> tuple[ScriptedAgent, ...]:
    """Return the offline agent set a run uses when none is supplied.

    One scripted agent per quality tier, named so audit trails and the cost
    ledger can tell them apart. Nothing here contacts a service.
    """
    return tuple(
        ScriptedAgent(f"scripted-{quality:.1f}", quality=quality)
        for quality in DEFAULT_AGENT_QUALITIES
    )


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
            "verify": self.stage_verify,
            "dedup": self.stage_dedup,
            "debate": self.stage_debate,
            "rank": self.stage_rank,
            "evolve": self.stage_evolve,
            "accumulate": self.stage_accumulate,
            "report": self.stage_report,
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

    def stage_verify(self) -> StageResult:
        """Grade every claim in the graph against the corpus."""
        if self.state.corpus is None:
            raise ConfigError("verify needs the corpus stage to have run")
        verifier = GroundingVerifier(self.state.corpus, self.config.grounding)
        reports = verifier.verify_graph(self.state.graph)
        self.state.reports = reports
        self.store.write_lines(GROUNDING_ARTIFACT, grounding_reports_to_lines(reports))
        return StageResult("verify", len(reports), (GROUNDING_ARTIFACT,))

    def stage_dedup(self) -> StageResult:
        """Cluster claim statements that restate each other."""
        texts = {claim.claim_id: claim.statement for claim in self.state.graph.claims}
        report = DuplicateFinder(self.config.dedup).report(texts)
        self.state.dedup = report
        self.store.write_lines(DEDUP_ARTIFACT, dedup_report_to_lines(report))
        return StageResult("dedup", len(report.clusters), (DEDUP_ARTIFACT,))

    def restore_verify(self) -> None:
        """Reload grounding reports after a resume."""
        self.state.reports = grounding_reports_from_lines(
            self.store.read_lines(GROUNDING_ARTIFACT)
        )

    def restore_dedup(self) -> None:
        """Reload the dedup report after a resume."""
        self.state.dedup = dedup_report_from_lines(
            self.store.read_lines(DEDUP_ARTIFACT)
        )

    def debate_loop(self) -> DebateLoop:
        """Build the debate loop from this pipeline's agents."""
        proposer = self.agents[0]
        critics = list(self.agents[1 : 1 + self.config.debate.critics]) or [proposer]
        return DebateLoop(proposer, critics, self.agents[-1], self.config.debate)

    def stage_debate(self) -> StageResult:
        """Debate the agent-proposed claims and write the revisions back.

        Only proposals are debated: the corpus-derived claims are already grounded
        in a specific span, and rewriting their statements would invalidate the
        citation they carry.
        """
        if self.state.corpus is None:
            raise ConfigError("debate needs the corpus stage to have run")
        context = corpus_context(self.state.corpus, limit=GENERATE_CONTEXT_LINES)
        loop = self.debate_loop()
        proposed = [
            claim
            for claim in self.state.graph.claims
            if claim.provenance.notes == "proposed"
        ][:DEBATE_CLAIM_LIMIT]
        debates: list[DebateResult] = []
        for claim in proposed:
            result = loop.run(context)
            apply_debate(
                self.state.graph, claim.claim_id, result, agent_id=loop.reviser.name
            )
            debates.append(result)
        self.record_usage("debate")
        self.state.debates = tuple(debates)
        self.store.write_lines(
            DEBATE_ARTIFACT, [debate_line(result) for result in debates]
        )
        self.store.write_lines("graph.jsonl", graph_to_lines(self.state.graph))
        return StageResult("debate", len(debates), (DEBATE_ARTIFACT, "graph.jsonl"))

    def stage_rank(self) -> StageResult:
        """Run the tournament over every claim currently in the graph."""
        reports = {report.claim_id: report for report in self.state.reports}
        judge = FeatureJudge(
            reports=reports,
            contradictions=graph_contradiction_counts(self.state.graph),
        )
        tournament = Tournament(judge, self.config.tournament).run(
            self.state.graph.claims
        )
        self.state.tournament = tournament
        self.store.write_lines(TOURNAMENT_ARTIFACT, tournament_to_lines(tournament))
        return StageResult("rank", len(tournament.matches), (TOURNAMENT_ARTIFACT,))

    def restore_debate(self) -> None:
        """Reload debate transcripts and the revised graph."""
        self.state.debates = tuple(
            debate_from_line(line) for line in self.store.read_lines(DEBATE_ARTIFACT)
        )
        self.state.graph = graph_from_lines(self.store.read_lines("graph.jsonl"))

    def restore_rank(self) -> None:
        """Reload the tournament audit trail."""
        self.state.tournament = tournament_from_lines(
            self.store.read_lines(TOURNAMENT_ARTIFACT)
        )

    def stage_evolve(self) -> StageResult:
        """Expand the graph with evolved claims and record every generation."""
        engine = self.config.evolution.engine()
        steps = engine.run(self.state.graph, self.config.evolution.generations)
        self.state.evolution = steps
        flat = [record for step in steps for record in step.accepted]
        self.store.write_lines(EVOLUTION_ARTIFACT, evolution_to_lines(steps))
        self.store.write_lines("graph.jsonl", graph_to_lines(self.state.graph))
        return StageResult("evolve", len(flat), (EVOLUTION_ARTIFACT, "graph.jsonl"))

    def stage_accumulate(self) -> StageResult:
        """Accumulate beliefs for every claim in the graph."""
        beliefs = accumulate_graph(self.state.graph, self.config.belief)
        self.state.beliefs = beliefs
        self.store.write_lines(
            BELIEF_ARTIFACT, belief_to_lines(beliefs, self.config.belief)
        )
        return StageResult("accumulate", len(beliefs), (BELIEF_ARTIFACT,))

    def stage_report(self) -> StageResult:
        """Write the run report as JSON plus rendered Markdown and HTML."""
        payload = run_report(self)
        self.store.write_json(REPORT_ARTIFACT, payload)
        rendered = write_reports(self.store, payload)
        recovered = payload["recovered"]
        count = recovered["planted"] if isinstance(recovered, dict) else 0
        return StageResult("report", int(count), (REPORT_ARTIFACT, *rendered))

    def restore_evolve(self) -> None:
        """Reload evolution steps and the evolved graph."""
        self.state.evolution = evolution_from_lines(
            self.store.read_lines(EVOLUTION_ARTIFACT)
        )
        self.state.graph = graph_from_lines(self.store.read_lines("graph.jsonl"))

    def restore_accumulate(self) -> None:
        """Reload belief states."""
        self.state.beliefs, _ = belief_from_lines(
            self.store.read_lines(BELIEF_ARTIFACT)
        )

    def restore_report(self) -> None:
        """The report artifact is terminal; nothing to reload."""

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
        if resume:
            self.restore_cost()
        for stage in self.config.stages:
            if stage not in handlers:
                raise ConfigError("stage has no implementation", stage=stage)
            if resume and self.store.stage_done(stage):
                self.restore(stage)
                results.append(StageResult(stage, 0, (), skipped=True))
                continue
            results.append(handlers[stage]())
            self.store.mark_stage(stage)
            self.store.write_json(COST_ARTIFACT, cost_ledger_to_dict(self.ledger))
        summary = RunSummary(
            run_id=self.config.run_id,
            config_fingerprint=self.config.fingerprint(),
            stages=tuple(results),
            completed=True,
        )
        self.store.write_metadata(RunMetadata.from_config(self.config))
        self.store.write_json(SUMMARY_ARTIFACT, summary.as_dict())
        return summary

    def restore_cost(self) -> None:
        """Reload previously booked token counts when resuming a run.

        Without this a resumed run would report only the tokens spent in the
        current process, so its report would differ from a straight run's. The
        accounting belongs to the run, not to the process that happens to finish
        it.
        """
        if not self.store.exists(COST_ARTIFACT):
            return
        self.ledger = cost_ledger_from_dict(self.store.read_json(COST_ARTIFACT))

    def restore(self, stage: str) -> None:
        """Reload the state a completed stage produced (used when resuming)."""
        loaders: dict[str, Callable[[], None]] = {
            "corpus": self.restore_corpus,
            "generate": self.restore_generate,
            "verify": self.restore_verify,
            "dedup": self.restore_dedup,
            "debate": self.restore_debate,
            "rank": self.restore_rank,
            "evolve": self.restore_evolve,
            "accumulate": self.restore_accumulate,
            "report": self.restore_report,
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


GROUNDING_ARTIFACT = "grounding.jsonl"
DEDUP_ARTIFACT = "dedup.jsonl"


DEBATE_ARTIFACT = "debates.jsonl"
TOURNAMENT_ARTIFACT = "tournament.jsonl"
DEBATE_CLAIM_LIMIT = 6


def debate_line(result: DebateResult) -> str:
    """Return one JSONL line holding a whole debate transcript."""
    return dumps_line({"record": "debate", "debate": debate_result_to_dict(result)})


def debate_from_line(line: str) -> DebateResult:
    """Decode one debate line."""
    payload = loads_line(line, field="debates")
    return debate_result_from_dict(payload["debate"], field="debates.debate")


EVOLUTION_ARTIFACT = "evolution.jsonl"
BELIEF_ARTIFACT = "beliefs.jsonl"
REPORT_ARTIFACT = "report.json"
REPORT_TOP = 5
REPORT_LIMITATIONS: tuple[str, ...] = (
    "Every corpus, claim and evidence item in this run is synthetic, generated "
    "with planted ground truth; no real literature was read or cited.",
    "Rankings express the configured judge's rubric preferences, not an "
    "assessment of scientific truth.",
    "Grounding flags come from lexical heuristics: exact span match, entity "
    "overlap, negation cues and numeric agreement.",
    "Belief posteriors follow a stipulated likelihood model and the selected "
    "contradiction policy; they are bookkeeping, not calibrated probabilities.",
    "Token counts are recorded for accounting only. Nothing in this run is "
    "priced, billed or sent to an external service.",
)


def recovered_links(
    truth: PlantedTruth, graph: HypothesisGraph
) -> tuple[tuple[str, bool], ...]:
    """Report, per planted link, whether a matching claim is in the graph.

    Matching is on the normalized ``(subject, relation, object)`` key, which is
    the same key the generator plants — so this is a real recovery check against
    known ground truth, not a similarity guess.
    """
    present = {
        (normalize(claim.subject), claim.relation.value, normalize(claim.object))
        for claim in graph.claims
    }
    return tuple(
        (link.statement, link.key() in present)
        for chain in truth.chains
        for link in chain.links
    )


def run_report(pipeline: Pipeline) -> dict[str, Any]:
    """Assemble the run report payload, including its limitations section."""
    state = pipeline.state
    stats = state.graph.stats()
    grounding = (
        summarize_reports(state.reports).as_dict()
        if state.reports
        else summarize_reports([]).as_dict()
    )
    recovered = (
        recovered_links(state.truth, state.graph) if state.truth is not None else ()
    )
    tournament = state.tournament
    return {
        "run_id": pipeline.config.run_id,
        "config_fingerprint": pipeline.config.fingerprint(),
        "corpus_hash": state.corpus_hash,
        "counts": {
            "documents": len(state.corpus) if state.corpus is not None else 0,
            "claims": stats.claims,
            "evidence": stats.evidence,
            "links": stats.links,
            "edges": stats.edges,
        },
        "grounding": grounding,
        "dedup": state.dedup.as_dict() if state.dedup is not None else None,
        "ranking": list(tournament.standings()[:REPORT_TOP]) if tournament else [],
        "beliefs": [
            {"claim_id": item.claim_id, "posterior": round(item.posterior, 6)}
            for item in sorted(
                state.beliefs, key=lambda item: (-item.posterior, item.claim_id)
            )[:REPORT_TOP]
        ],
        "evolution": {
            "generations": len(state.evolution),
            "accepted": sum(step.accepted_count for step in state.evolution),
            "rejected": sum(len(step.rejected) for step in state.evolution),
        },
        "recovered": {
            "planted": len(recovered),
            "recovered": sum(1 for _, found in recovered if found),
            "rate": round(sum(1 for _, found in recovered if found) / len(recovered), 4)
            if recovered
            else 0.0,
            "links": [
                {"statement": statement, "recovered": found}
                for statement, found in recovered
            ],
        },
        "cost": pipeline.ledger.totals(),
        "limitations": list(REPORT_LIMITATIONS),
    }
