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
