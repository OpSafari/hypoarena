"""Model-agnostic agent adapters for the discovery loop.

Three adapters ship with the toolkit and all of them are offline:

* :class:`ScriptedAgent` derives its replies from the request alone, with a
  ``quality`` knob that plants a skill order for tournament tests;
* :class:`ReplayAgent` serves responses recorded in JSONL fixtures;
* :class:`~hypoarena.http_agent.HttpAgent` speaks the OpenAI-compatible chat
  completions format and is exercised only against a local loopback mock.

Every adapter returns the same records and reports token counts, so cost
accounting works identically no matter where a reply came from. Nothing here
measures a real model: the adapters exist to make the loop itself testable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from hypoarena.errors import (
    AdapterError,
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
    make_id,
)
from hypoarena.text import (
    content_tokens,
)

AGENT_TASKS = ("propose", "critique", "revise", "judge")


@dataclass(frozen=True)
class AgentRequest:
    """One call into an agent.

    ``context`` carries the supporting text (corpus excerpts, prior critiques)
    as plain lines; ``request_id`` is derived from the content so that replay
    fixtures and audit trails can match requests without extra bookkeeping.
    """

    task: str
    prompt: str
    context: tuple[str, ...] = ()
    agent: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    request_id: str = ""

    def __post_init__(self) -> None:
        if self.task not in AGENT_TASKS:
            raise ValidationError(
                "unknown agent task", task=self.task, allowed=list(AGENT_TASKS)
            )
        if not self.prompt.strip():
            raise ValidationError("agent prompt must not be blank")
        if self.temperature < 0.0:
            raise ValidationError(
                "temperature must be >= 0", temperature=self.temperature
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValidationError(
                "max_tokens must be >= 1 when set", max_tokens=self.max_tokens
            )
        for line in self.context:
            if not isinstance(line, str):
                raise ValidationError(
                    "agent context lines must be strings", got=type(line).__name__
                )

    def fingerprint(self) -> str:
        """Return a digest identifying this request's content."""
        return content_hash(
            {"task": self.task, "prompt": self.prompt, "context": list(self.context)}
        )

    def with_id(self) -> AgentRequest:
        """Return a copy whose ``request_id`` is derived from its content."""
        if self.request_id:
            return self
        identifier = make_id("req", self.task, self.prompt, tuple(self.context))
        return AgentRequest(
            task=self.task,
            prompt=self.prompt,
            context=self.context,
            agent=self.agent,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_id=identifier,
        )


@dataclass(frozen=True)
class AgentResponse:
    """One reply from an agent, with the token counts it reported."""

    request_id: str
    text: str
    agent: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "scripted"
    finish_reason: str = "stop"

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValidationError(
                "token counts must be >= 0",
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            )
        if not self.agent.strip():
            raise ValidationError("agent responses must name their agent")

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Usage:
    """Accumulated call and token counters for one agent."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_task: dict[str, int] = field(default_factory=dict)

    def record(self, task: str, response: AgentResponse) -> None:
        """Add one response to the counters."""
        self.calls += 1
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.by_task[task] = self.by_task.get(task, 0) + 1

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens seen so far."""
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view for run metadata and reports."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "by_task": dict(sorted(self.by_task.items())),
        }


def count_words(text: str) -> int:
    """Return a whitespace word count, used as a tokenizer-free token proxy.

    Adapters that talk to a real service report that service's counts instead;
    this proxy exists so offline adapters still produce non-zero, reproducible
    accounting for the runner's cost hooks.
    """
    return len(text.split())


@runtime_checkable
class DiscoveryAgent(Protocol):
    """The surface the generate-debate-evolve loop depends on.

    Keeping the protocol structural means a test double only has to provide the
    four members below; nothing has to inherit from this package.
    """

    name: str
    usage: Usage

    def respond(self, request: AgentRequest) -> AgentResponse: ...

    def propose(self, prompt: str, context: Sequence[str] = ()) -> AgentResponse: ...

    def critique(self, prompt: str, context: Sequence[str] = ()) -> AgentResponse: ...

    def revise(self, prompt: str, context: Sequence[str] = ()) -> AgentResponse: ...


class BaseAgent:
    """Template implementation of :class:`DiscoveryAgent`.

    Subclasses implement :meth:`respond` only. The base class builds the request
    (including its derived identifier), checks that the reply belongs to the
    request that was sent, and records usage — so token accounting cannot be
    forgotten by a new adapter.
    """

    def __init__(self, name: str) -> None:
        if not name.strip():
            raise ValidationError("agent name must not be blank")
        self.name = name
        self.usage = Usage()

    def respond(self, request: AgentRequest) -> AgentResponse:
        """Produce a reply for one request; adapters must override this."""
        raise NotImplementedError(f"{type(self).__name__} must implement respond()")

    def propose(self, prompt: str, context: Sequence[str] = ()) -> AgentResponse:
        """Ask the agent for a new hypothesis."""
        return self.run("propose", prompt, context)

    def critique(self, prompt: str, context: Sequence[str] = ()) -> AgentResponse:
        """Ask the agent to critique a hypothesis."""
        return self.run("critique", prompt, context)

    def revise(self, prompt: str, context: Sequence[str] = ()) -> AgentResponse:
        """Ask the agent to revise a hypothesis in light of a critique."""
        return self.run("revise", prompt, context)

    def run(self, task: str, prompt: str, context: Sequence[str] = ()) -> AgentResponse:
        """Build a request, call :meth:`respond` and record the usage."""
        request = AgentRequest(
            task=task, prompt=prompt, context=tuple(context), agent=self.name
        ).with_id()
        response = self.respond(request)
        if response.request_id != request.request_id:
            raise AdapterError(
                "adapter replied to a different request",
                agent=self.name,
                expected=request.request_id,
                got=response.request_id,
            )
        self.usage.record(task, response)
        return response


QUALITY_TIERS = ("vague", "focused", "mechanistic")
VAGUE_PROPOSAL = "an intervention is associated with an outcome"


def quality_tier(quality: float) -> str:
    """Map a quality score in ``[0, 1]`` onto a named behaviour tier."""
    if not 0.0 <= quality <= 1.0:
        raise ValidationError("quality must lie within [0, 1]", quality=quality)
    if quality < 1 / 3:
        return QUALITY_TIERS[0]
    if quality < 2 / 3:
        return QUALITY_TIERS[1]
    return QUALITY_TIERS[2]


class ScriptedAgent(BaseAgent):
    """Deterministic agent whose replies depend only on the request.

    ``quality`` selects a behaviour tier: *vague* agents ignore the context and
    return a generic statement, *focused* agents name the entities they were
    given, and *mechanistic* agents add a mechanism and a measurement clause.
    That ladder is what tournament tests use to plant a skill order — no model is
    involved, and the ordering is exact by construction.
    """

    def __init__(
        self,
        name: str = "scripted",
        *,
        quality: float = 0.5,
        model_tag: str = "scripted",
    ) -> None:
        super().__init__(name)
        self.quality = quality
        self.tier = quality_tier(quality)
        self.model_tag = model_tag

    def respond(self, request: AgentRequest) -> AgentResponse:
        """Dispatch on the request's task and count words as token proxy."""
        handler = self.handlers().get(request.task)
        if handler is None:
            raise AdapterError(
                "scripted agent cannot handle this task",
                agent=self.name,
                task=request.task,
            )
        text = handler(request)
        prompt_words = count_words(request.prompt) + sum(
            count_words(line) for line in request.context
        )
        return AgentResponse(
            request_id=request.request_id,
            text=text,
            agent=self.name,
            prompt_tokens=prompt_words,
            completion_tokens=count_words(text),
            model=self.model_tag,
        )

    def handlers(self) -> dict[str, Callable[[AgentRequest], str]]:
        """Return the task handlers this agent implements."""
        return {
            "propose": self.propose_text,
            "critique": self.critique_text,
            "revise": self.revise_text,
        }

    def context_terms(self, request: AgentRequest) -> tuple[str, ...]:
        """Return the content tokens of the supplied context lines."""
        return tuple(content_tokens(" ".join(request.context)))

    def propose_text(self, request: AgentRequest) -> str:
        """Build a proposal whose specificity follows the quality tier.

        Specificity is monotone in ``quality`` by construction: each tier keeps
        what the previous one said and adds a clause. Tournament tests rely on
        that ordering, so the tiers must not be reordered casually.
        """
        terms = self.context_terms(request)
        if self.tier == "vague" or len(terms) < 2:
            return VAGUE_PROPOSAL
        subject = terms[0]
        target = next((term for term in reversed(terms) if term != subject), "")
        if not target:
            return VAGUE_PROPOSAL
        if self.tier == "focused":
            return f"{subject} increases {target} in the assayed population"
        middle = next(
            (term for term in terms if term not in (subject, target)), "the pathway"
        )
        return (
            f"{subject} increases {target} through {middle} in the assayed "
            "population, measured by dose response"
        )

    def critique_text(self, request: AgentRequest) -> str:
        """Produce a critique whose specificity follows the quality tier.

        A vague critique is content-free, a focused one names the entity the
        claim is about and what is missing, and a mechanistic one lists all three
        gaps (mechanism, assay, scope). The ladder is what makes critique quality
        measurable without a model in the loop.
        """
        terms = self.context_terms(request)
        if self.tier == "vague":
            return "the claim needs more support"
        subject = terms[0] if terms else "the subject"
        if self.tier == "focused":
            return f"the claim about {subject} does not name a testable assay"
        return (
            f"the claim about {subject} names no mechanism, no assay and a scope "
            "wider than the cited evidence"
        )

    def revise_text(self, request: AgentRequest) -> str:
        """Revise the statement carried in ``prompt``.

        The critique travels in ``context``; a vague agent returns the statement
        unchanged (which the evolution loop treats as no progress), a focused one
        narrows the scope, and a mechanistic one also names the measurement that
        would settle the claim.
        """
        statement = " ".join(request.prompt.split())
        if self.tier == "vague":
            return f"{statement} (unchanged)"
        if self.tier == "focused":
            return f"{statement} in the assayed population"
        return f"{statement} in the assayed population, measured by dose response"
