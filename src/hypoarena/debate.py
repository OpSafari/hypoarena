"""The generate–critique–revise debate loop.

The loop is deliberately thin: it drives agents that satisfy
:class:`~hypoarena.agents.DiscoveryAgent`, records every exchange and stops when
revision reaches a fixed point. Nothing here judges the *content* of a
hypothesis — ranking happens in the tournament, and grounding happens in the
verifier — so the loop behaves identically for scripted, replayed and HTTP
agents.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from hypoarena.agents import (
    AgentResponse,
    DiscoveryAgent,
    Usage,
    merge_usage,
)
from hypoarena.corpus import (
    Corpus,
)
from hypoarena.errors import (
    ValidationError,
)
from hypoarena.graph import (
    HypothesisGraph,
)
from hypoarena.ids import (
    content_hash,
)
from hypoarena.schema import (
    Claim,
    Provenance,
)
from hypoarena.text import (
    normalize,
    normalize_whitespace,
    sentence_split,
)


@dataclass(frozen=True)
class Critique:
    """One critic's verdict on a statement."""

    agent: str
    text: str
    request_id: str

    def __post_init__(self) -> None:
        if not self.agent.strip():
            raise ValidationError("critique must name its agent")
        if not self.text.strip():
            raise ValidationError("critique text must not be blank")

    @classmethod
    def from_response(cls, response: AgentResponse) -> Critique:
        """Build a critique from an agent response."""
        return cls(
            agent=response.agent, text=response.text, request_id=response.request_id
        )


@dataclass(frozen=True)
class DebateTurn:
    """One round: the statement in, the critiques, and the statement out."""

    round_index: int
    statement: str
    critiques: tuple[Critique, ...]
    revised: str

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValidationError(
                "round_index must be >= 0", round_index=self.round_index
            )
        if not self.statement.strip():
            raise ValidationError("a debate turn needs a non-blank statement")

    @property
    def changed(self) -> bool:
        """True when revision produced a different statement (normalized)."""
        return normalize(self.revised) != normalize(self.statement)

    def transcript(self) -> tuple[str, ...]:
        """Return human-readable lines for this turn."""
        lines = [f"round {self.round_index}: {self.statement}"]
        lines.extend(
            f"  critique[{critique.agent}]: {critique.text}"
            for critique in self.critiques
        )
        lines.append(f"  revised: {self.revised}")
        return tuple(lines)


DEFAULT_PROPOSAL_PROMPT = "Propose one hypothesis supported by the context."
DEFAULT_CRITIQUE_PROMPT = "Critique this hypothesis: {statement}"


@dataclass(frozen=True)
class DebateConfig:
    """How many rounds to run and what to ask the agents.

    ``critique_prompt`` must contain ``{statement}`` so the statement under
    debate is part of the instruction. Revision follows the adapter convention
    instead: the statement is the prompt and the critiques are the context, which
    is what ``replay_transcript`` and the recorded fixtures already assume.
    """

    rounds: int = 2
    critics: int = 2
    stop_on_unchanged: bool = True
    proposal_prompt: str = DEFAULT_PROPOSAL_PROMPT
    critique_prompt: str = DEFAULT_CRITIQUE_PROMPT

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValidationError("debate rounds must be >= 1", rounds=self.rounds)
        if self.critics < 1:
            raise ValidationError(
                "a debate needs at least one critic", critics=self.critics
            )
        for name in ("proposal_prompt", "critique_prompt"):
            if not getattr(self, name).strip():
                raise ValidationError(f"{name} must not be blank")
        for name in ("critique_prompt",):
            if "{statement}" not in getattr(self, name):
                raise ValidationError(
                    f"{name} must contain the {{statement}} placeholder",
                    value=getattr(self, name),
                )

    def critique_prompt_for(self, statement: str) -> str:
        """Render the critique prompt for one statement."""
        return self.critique_prompt.format(statement=statement)

    def revise_prompt_for(self, statement: str) -> str:
        """Return the revision prompt for one statement.

        Following the adapter convention documented above, revision uses the
        statement itself as the prompt and supplies the critiques as context, so
        there is no separate template to render here.
        """
        return statement

    def fingerprint(self) -> str:
        """Return a digest of the configuration for run metadata."""
        return content_hash(
            {
                "rounds": self.rounds,
                "critics": self.critics,
                "stop_on_unchanged": self.stop_on_unchanged,
                "proposal_prompt": self.proposal_prompt,
                "critique_prompt": self.critique_prompt,
            }
        )


@dataclass(frozen=True)
class DebateResult:
    """Everything one debate produced, including its accounting."""

    proposal: str
    final_statement: str
    turns: tuple[DebateTurn, ...]
    converged: bool
    rounds_run: int
    agents: tuple[str, ...]
    usage: dict[str, object]
    context: tuple[str, ...]
    config_fingerprint: str

    def transcript(self) -> tuple[str, ...]:
        """Return the full transcript as readable lines."""
        lines = [f"proposal: {self.proposal}"]
        for turn in self.turns:
            lines.extend(turn.transcript())
        lines.append(f"final: {self.final_statement}")
        lines.append(
            f"converged={self.converged} rounds={self.rounds_run} "
            f"agents={','.join(self.agents)}"
        )
        return tuple(lines)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready summary (without the full transcript)."""
        return {
            "proposal": self.proposal,
            "final_statement": self.final_statement,
            "rounds_run": self.rounds_run,
            "converged": self.converged,
            "agents": list(self.agents),
            "context_lines": len(self.context),
            "config_fingerprint": self.config_fingerprint,
            "usage": self.usage,
        }

    def signature(self) -> str:
        """Return a digest over the transcript, for run-to-run comparison."""
        return content_hash(list(self.transcript()))


class DebateLoop:
    """Drives one proposer, N critics and one reviser through the rounds."""

    def __init__(
        self,
        proposer: DiscoveryAgent,
        critics: Sequence[DiscoveryAgent],
        reviser: DiscoveryAgent | None = None,
        config: DebateConfig | None = None,
    ) -> None:
        for role, agent in (("proposer", proposer), ("reviser", reviser or proposer)):
            if not isinstance(agent, DiscoveryAgent):
                raise ValidationError(
                    f"{role} must satisfy the DiscoveryAgent protocol",
                    role=role,
                    got=type(agent).__name__,
                )
        critic_list = tuple(critics)
        if not critic_list:
            raise ValidationError("a debate needs at least one critic")
        for index, critic in enumerate(critic_list):
            if not isinstance(critic, DiscoveryAgent):
                raise ValidationError(
                    "critics must satisfy the DiscoveryAgent protocol",
                    index=index,
                    got=type(critic).__name__,
                )
        self.proposer = proposer
        self.critics = critic_list[: (config or DebateConfig()).critics] or critic_list
        self.reviser = reviser or proposer
        self.config = config or DebateConfig()
        if len(critic_list) < self.config.critics:
            raise ValidationError(
                "not enough critics for the configured debate",
                available=len(critic_list),
                required=self.config.critics,
            )

    @property
    def agent_names(self) -> tuple[str, ...]:
        """Distinct agent names taking part, in role order."""
        seen: list[str] = []
        for agent in (self.proposer, *self.critics, self.reviser):
            if agent.name not in seen:
                seen.append(agent.name)
        return tuple(seen)

    def run(self, context: Sequence[str] = ()) -> DebateResult:
        """Run the debate and return its result.

        The loop stops early when a revision changes nothing and
        ``stop_on_unchanged`` is set; ``converged`` reports whether that fixed
        point was reached inside the round budget. Revision follows the adapter
        convention: the statement is the prompt and the critiques are context, so
        an agent that returns the statement unchanged really has converged.
        """
        lines = tuple(context)
        baseline = self.usage_snapshot()
        proposal = self.proposer.propose(self.config.proposal_prompt, lines)
        statement = proposal.text
        turns: list[DebateTurn] = []
        converged = False
        for index in range(self.config.rounds):
            critiques = tuple(
                Critique.from_response(
                    critic.critique(self.config.critique_prompt_for(statement), lines)
                )
                for critic in self.critics
            )
            revision = self.reviser.revise(
                statement, tuple(critique.text for critique in critiques)
            )
            turn = DebateTurn(
                round_index=index,
                statement=statement,
                critiques=critiques,
                revised=revision.text,
            )
            turns.append(turn)
            changed = turn.changed
            statement = revision.text if changed else statement
            if not changed:
                converged = True
                if self.config.stop_on_unchanged:
                    break
        return DebateResult(
            proposal=proposal.text,
            final_statement=statement,
            turns=tuple(turns),
            converged=converged,
            rounds_run=len(turns),
            agents=self.agent_names,
            usage=self.usage_summary(baseline),
            context=lines,
            config_fingerprint=self.config.fingerprint(),
        )

    def distinct_agents(self) -> list[DiscoveryAgent]:
        """Return the loop's agents without duplicates, in role order."""
        distinct: list[DiscoveryAgent] = []
        for agent in (self.proposer, *self.critics, self.reviser):
            if not any(agent is item for item in distinct):
                distinct.append(agent)
        return distinct

    def usage_snapshot(self) -> dict[str, Usage]:
        """Copy every agent's counters, for delta accounting."""
        return {
            agent.name: Usage(
                agent.usage.calls,
                agent.usage.prompt_tokens,
                agent.usage.completion_tokens,
                dict(agent.usage.by_task),
            )
            for agent in self.distinct_agents()
        }

    @staticmethod
    def usage_delta(current: Usage, base: Usage) -> Usage:
        """Return what an agent used since ``base`` was taken."""
        return Usage(
            calls=current.calls - base.calls,
            prompt_tokens=current.prompt_tokens - base.prompt_tokens,
            completion_tokens=current.completion_tokens - base.completion_tokens,
            by_task={
                task: count - base.by_task.get(task, 0)
                for task, count in current.by_task.items()
                if count - base.by_task.get(task, 0) > 0
            },
        )

    def usage_summary(
        self, since: Mapping[str, Usage] | None = None
    ) -> dict[str, object]:
        """Return per-agent and total usage for one debate.

        The summary is a delta against a snapshot taken when the debate started,
        so a transcript serializes the same way no matter what the agents did
        earlier in the process. That is what lets a resumed run reproduce its
        artifacts byte for byte.
        """
        baseline = since or {}
        agents = self.distinct_agents()
        deltas = [
            self.usage_delta(agent.usage, baseline.get(agent.name, Usage()))
            for agent in agents
        ]
        return {
            "agents": {
                agent.name: delta.as_dict()
                for agent, delta in zip(agents, deltas, strict=True)
            },
            "total": merge_usage(*deltas).as_dict(),
        }


MIN_CONTEXT_CHARS = 20
CONTEXT_ELLIPSIS = "..."


def corpus_context(
    corpus: Corpus, *, limit: int = 6, max_chars: int = 200
) -> tuple[str, ...]:
    """Return context lines from a corpus, one per document.

    Documents are visited in identifier order and only their first sentence is
    used (truncated with an ellipsis when it is longer than ``max_chars``), so
    the same corpus always yields the same context and a debate transcript can
    be reproduced from the corpus alone.
    """
    if limit < 1:
        raise ValidationError("context limit must be >= 1", limit=limit)
    if max_chars < MIN_CONTEXT_CHARS:
        raise ValidationError(
            "context max_chars is too small",
            max_chars=max_chars,
            minimum=MIN_CONTEXT_CHARS,
        )
    lines: list[str] = []
    for document in corpus.documents:
        if len(lines) >= limit:
            break
        sentences = sentence_split(document.text)
        line = sentences[0] if sentences else normalize_whitespace(document.text)
        if len(line) > max_chars:
            cut = max_chars - len(CONTEXT_ELLIPSIS)
            line = line[:cut].rstrip() + CONTEXT_ELLIPSIS
        lines.append(line)
    return tuple(lines)


def claim_context(claims: Iterable[Claim], *, limit: int = 6) -> tuple[str, ...]:
    """Return the statements of existing claims as context for a new debate."""
    if limit < 1:
        raise ValidationError("context limit must be >= 1", limit=limit)
    return tuple(claim.statement for claim in list(claims)[:limit])


def revised_claim(
    claim: Claim, result: DebateResult, *, agent_id: str | None = None
) -> Claim:
    """Return a copy of ``claim`` whose statement is the debate's outcome.

    Citations are preserved so grounding can be re-checked after revision, and
    provenance records the debate: origin ``agent``, the proposer's name (or
    ``agent_id``), generation incremented, the original claim as parent and the
    round count in ``notes``.
    """
    if not result.final_statement.strip():
        raise ValidationError("debate produced no statement")
    if not result.agents:
        raise ValidationError("debate result names no agents")
    provenance = Provenance(
        origin="agent",
        agent_id=agent_id or result.agents[0],
        generation=claim.provenance.generation + 1,
        parents=(claim.claim_id,),
        seed=claim.provenance.seed,
        corpus_hash=claim.provenance.corpus_hash,
        notes=f"debate:{result.rounds_run}",
    )
    return replace(claim, statement=result.final_statement, provenance=provenance)


def apply_debate(
    graph: HypothesisGraph,
    claim_id: str,
    result: DebateResult,
    *,
    agent_id: str | None = None,
) -> Claim:
    """Revise one claim inside a graph, keeping its links and edges."""
    revised = revised_claim(graph.claim(claim_id), result, agent_id=agent_id)
    graph.replace_claim(revised)
    graph.validate()
    return revised
