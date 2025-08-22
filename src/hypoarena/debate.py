"""The generate–critique–revise debate loop.

The loop is deliberately thin: it drives agents that satisfy
:class:`~hypoarena.agents.DiscoveryAgent`, records every exchange and stops when
revision reaches a fixed point. Nothing here judges the *content* of a
hypothesis — ranking happens in the tournament, and grounding happens in the
verifier — so the loop behaves identically for scripted, replayed and HTTP
agents.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypoarena.agents import (
    AgentResponse,
)
from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
)
from hypoarena.text import (
    normalize,
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
DEFAULT_REVISE_PROMPT = "Revise this hypothesis using the critiques: {statement}"


@dataclass(frozen=True)
class DebateConfig:
    """How many rounds to run and what to ask the agents.

    ``critic_prompt`` and ``revise_prompt`` must contain ``{statement}`` so the
    statement under debate is always part of the request; that is what makes a
    transcript reproducible from its context alone.
    """

    rounds: int = 2
    critics: int = 2
    stop_on_unchanged: bool = True
    proposal_prompt: str = DEFAULT_PROPOSAL_PROMPT
    critique_prompt: str = DEFAULT_CRITIQUE_PROMPT
    revise_prompt: str = DEFAULT_REVISE_PROMPT

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValidationError("debate rounds must be >= 1", rounds=self.rounds)
        if self.critics < 1:
            raise ValidationError(
                "a debate needs at least one critic", critics=self.critics
            )
        for name in ("proposal_prompt", "critique_prompt", "revise_prompt"):
            if not getattr(self, name).strip():
                raise ValidationError(f"{name} must not be blank")
        for name in ("critique_prompt", "revise_prompt"):
            if "{statement}" not in getattr(self, name):
                raise ValidationError(
                    f"{name} must contain the {{statement}} placeholder",
                    value=getattr(self, name),
                )

    def critique_prompt_for(self, statement: str) -> str:
        """Render the critique prompt for one statement."""
        return self.critique_prompt.format(statement=statement)

    def revise_prompt_for(self, statement: str) -> str:
        """Render the revision prompt for one statement."""
        return self.revise_prompt.format(statement=statement)

    def fingerprint(self) -> str:
        """Return a digest of the configuration for run metadata."""
        return content_hash(
            {
                "rounds": self.rounds,
                "critics": self.critics,
                "stop_on_unchanged": self.stop_on_unchanged,
                "proposal_prompt": self.proposal_prompt,
                "critique_prompt": self.critique_prompt,
                "revise_prompt": self.revise_prompt,
            }
        )
