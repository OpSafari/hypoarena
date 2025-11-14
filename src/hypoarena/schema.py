"""Versioned schemas for claims, citations, evidence and provenance.

Design rules used throughout this module:

* every record is a frozen dataclass, so hashes and equality are structural;
* ``__post_init__`` performs validation, so an invalid record cannot exist —
  constructors raise instead of returning a broken object;
* ``to_dict``/``from_dict`` are exact inverses and go through
  :mod:`hypoarena.codec`, which rejects unknown keys;
* only top-level records (claims, evidence) carry ``schema_version``; nested
  value objects stay version-free to keep payloads small.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum, unique

from hypoarena.errors import ValidationError
from hypoarena.ids import canonical_json
from hypoarena.text import normalize

SCHEMA_VERSION = "1.0"


@unique
class PredictedRelation(StrEnum):
    """Direction a claim predicts between its subject and object variables."""

    INCREASES = "increases"
    DECREASES = "decreases"
    ENABLES = "enables"
    INHIBITS = "inhibits"
    CAUSES = "causes"
    ASSOCIATES = "associates"


@unique
class EvidencePolarity(StrEnum):
    """How an evidence item bears on the claim it is attached to."""

    SUPPORT = "support"
    REFUTE = "refute"
    NEUTRAL = "neutral"


@unique
class ClaimRelation(StrEnum):
    """Typed edges between two claims in the hypothesis graph."""

    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    REFINES = "refines"


DIRECTIONAL_RELATIONS: frozenset[PredictedRelation] = frozenset(
    {
        PredictedRelation.INCREASES,
        PredictedRelation.DECREASES,
        PredictedRelation.ENABLES,
        PredictedRelation.INHIBITS,
        PredictedRelation.CAUSES,
    }
)
RELATION_OPPOSITES: dict[PredictedRelation, PredictedRelation] = {
    PredictedRelation.INCREASES: PredictedRelation.DECREASES,
    PredictedRelation.DECREASES: PredictedRelation.INCREASES,
    PredictedRelation.ENABLES: PredictedRelation.INHIBITS,
    PredictedRelation.INHIBITS: PredictedRelation.ENABLES,
}


def is_directional(relation: PredictedRelation) -> bool:
    """True when the relation asserts a direction rather than mere association."""
    return relation in DIRECTIONAL_RELATIONS


def opposite_relation(relation: PredictedRelation) -> PredictedRelation | None:
    """Return the opposing relation, or ``None`` when there is none.

    ``causes`` and ``associates`` have no opposite in this schema: negating a
    causal claim is done by attaching refuting evidence, not by flipping the
    relation label.
    """
    return RELATION_OPPOSITES.get(relation)


def relations_conflict(first: PredictedRelation, second: PredictedRelation) -> bool:
    """True when two relations over the same variable pair cannot both hold."""
    return opposite_relation(first) is second


@dataclass(frozen=True)
class Scope:
    """The population and conditions under which a claim is asserted to hold.

    ``population`` is the system the claim is about (a cell type, an organism, a
    dataset); ``conditions`` are additional restrictions such as ``"hypoxia"``.
    Narrowing a scope never removes conditions, which is what makes
    :meth:`is_narrower_than` a partial order usable by the refinement edge type.
    """

    population: str
    conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.population.strip():
            raise ValidationError("scope population must not be blank")
        for condition in self.conditions:
            if not condition.strip():
                raise ValidationError(
                    "scope conditions must not be blank", conditions=self.conditions
                )
        if len(set(self.conditions)) != len(self.conditions):
            raise ValidationError(
                "scope conditions contain duplicates", conditions=self.conditions
            )

    def narrowed(self, condition: str) -> Scope:
        """Return a copy restricted by ``condition``; narrowing is idempotent."""
        if not condition.strip():
            raise ValidationError("narrowing condition must not be blank")
        if condition in self.conditions:
            return self
        return replace(self, conditions=(*self.conditions, condition))

    def is_narrower_than(self, other: Scope) -> bool:
        """True when this scope adds at least one condition to ``other``."""
        return self.population == other.population and set(other.conditions) < set(
            self.conditions
        )

    def signature(self) -> str:
        """Return an order-insensitive canonical signature for dedup."""
        return canonical_json(
            {
                "population": normalize(self.population),
                "conditions": sorted(
                    normalize(condition) for condition in self.conditions
                ),
            }
        )
