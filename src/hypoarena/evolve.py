"""Evolution operators over the hypothesis graph.

Five operators turn existing claims into new candidates: narrowing a scope,
substituting a variable, flipping a predicted relation, crossing two parents and
decomposing a compound statement. Each one returns brand-new immutable records —
nothing is mutated in place — and each records its parents in provenance, so a
derived claim can always be traced back.

Operators never produce invalid claims: the schema validates on construction, and
the engine re-validates the graph after inserting a child. The property tests in
``tests/test_evolve_validity.py`` check exactly that over seeded random graphs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    make_id,
)
from hypoarena.schema import (
    Claim,
    Provenance,
)

OPERATORS: tuple[str, ...] = (
    "narrow_scope",
    "substitute_variable",
    "flip_relation",
    "crossover",
    "decompose",
)
VARIABLE_SLOTS: tuple[str, ...] = ("subject", "object")


def check_operator(name: str) -> None:
    """Reject an unknown operator name."""
    if name not in OPERATORS:
        raise ValidationError(
            "unknown evolution operator", operator=name, allowed=list(OPERATORS)
        )


def evolved_claim_id(
    operator: str, parents: Sequence[str], child: str, *, extra: object = ""
) -> str:
    """Derive a deterministic identifier for an evolved claim."""
    check_operator(operator)
    if not parents:
        raise ValidationError("evolved claims need at least one parent")
    return make_id("clm", operator, tuple(parents), child, extra)


def evolved_provenance(
    operator: str, parents: Sequence[Claim], *, seed: int | None = None
) -> Provenance:
    """Build provenance for a child of ``parents`` under ``operator``.

    The generation is one more than the most advanced parent, so a lineage that
    is evolved repeatedly keeps counting up instead of staying flat.
    """
    check_operator(operator)
    if not parents:
        raise ValidationError("evolved claims need at least one parent")
    return Provenance(
        origin="evolved",
        generation=max(parent.provenance.generation for parent in parents) + 1,
        parents=tuple(parent.claim_id for parent in parents),
        seed=seed if seed is not None else parents[0].provenance.seed,
        corpus_hash=parents[0].provenance.corpus_hash,
        notes=operator,
    )


@dataclass(frozen=True)
class EvolutionRecord:
    """One accepted application of an operator."""

    operator: str
    parents: tuple[str, ...]
    child: Claim
    rationale: str

    def __post_init__(self) -> None:
        check_operator(self.operator)
        if not self.parents:
            raise ValidationError("an evolution record needs at least one parent")
        if not self.rationale.strip():
            raise ValidationError("an evolution record needs a rationale")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view for artifacts and reports."""
        return {
            "operator": self.operator,
            "parents": list(self.parents),
            "child_id": self.child.claim_id,
            "child_statement": self.child.statement,
            "rationale": self.rationale,
            "generation": self.child.provenance.generation,
        }


@dataclass(frozen=True)
class Rejection:
    """One candidate an operator produced but the pipeline refused."""

    operator: str
    parents: tuple[str, ...]
    reason: str
    statement: str = ""
    nearest: str | None = None
    similarity: float = 0.0

    def __post_init__(self) -> None:
        check_operator(self.operator)
        if not self.reason.strip():
            raise ValidationError("a rejection needs a reason")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view for artifacts and reports."""
        return {
            "operator": self.operator,
            "parents": list(self.parents),
            "reason": self.reason,
            "statement": self.statement,
            "nearest": self.nearest,
            "similarity": round(self.similarity, 6),
        }
