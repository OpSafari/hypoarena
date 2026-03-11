"""Token accounting for pipeline runs.

The ledger records *counts*, never prices: an offline pipeline has no price list,
and inventing one would put a fabricated number into run artifacts. Entries are
keyed by stage and agent so a report can show where the tokens went, and the
totals are the same numbers the adapters reported — nothing is estimated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hypoarena.agents import (
    Usage,
)
from hypoarena.errors import (
    ValidationError,
)

LEDGER_ENTRY_KEYS = (
    "stage",
    "agent",
    "calls",
    "prompt_tokens",
    "completion_tokens",
)


@dataclass(frozen=True)
class CostEntry:
    """One agent's contribution to one stage."""

    stage: str
    agent: str
    calls: int
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValidationError("cost entry needs a stage")
        if not self.agent.strip():
            raise ValidationError("cost entry needs an agent")
        for name in ("calls", "prompt_tokens", "completion_tokens"):
            if getattr(self, name) < 0:
                raise ValidationError(
                    f"{name} must be >= 0", **{name: getattr(self, name)}
                )

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int | str]:
        """Return a JSON-ready view."""
        return {
            "stage": self.stage,
            "agent": self.agent,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class CostLedger:
    """Accumulates cost entries across a run."""

    entries: list[CostEntry] = field(default_factory=list)

    def record(self, stage: str, agent: str, usage: Usage) -> CostEntry:
        """Record one agent's usage for a stage."""
        entry = CostEntry(
            stage=stage,
            agent=agent,
            calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        self.entries.append(entry)
        return entry

    def totals(self) -> dict[str, int]:
        """Return run-wide counters."""
        return {
            "entries": len(self.entries),
            "calls": sum(entry.calls for entry in self.entries),
            "prompt_tokens": sum(entry.prompt_tokens for entry in self.entries),
            "completion_tokens": sum(entry.completion_tokens for entry in self.entries),
            "total_tokens": sum(entry.total_tokens for entry in self.entries),
        }

    def by_stage(self) -> dict[str, dict[str, int]]:
        """Return counters grouped by stage, in insertion order of first use."""
        grouped: dict[str, dict[str, int]] = {}
        for entry in self.entries:
            bucket = grouped.setdefault(
                entry.stage,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            bucket["calls"] += entry.calls
            bucket["prompt_tokens"] += entry.prompt_tokens
            bucket["completion_tokens"] += entry.completion_tokens
            bucket["total_tokens"] += entry.total_tokens
        return grouped

    def for_stage(self, stage: str) -> tuple[CostEntry, ...]:
        """Return the entries recorded for one stage."""
        return tuple(entry for entry in self.entries if entry.stage == stage)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view for run artifacts."""
        return {
            "entries": [entry.as_dict() for entry in self.entries],
            "totals": self.totals(),
            "by_stage": self.by_stage(),
        }
