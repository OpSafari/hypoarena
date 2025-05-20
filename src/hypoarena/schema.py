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

import re
from dataclasses import dataclass, replace
from enum import StrEnum, unique

from hypoarena.errors import ValidationError
from hypoarena.ids import canonical_json, is_valid_id
from hypoarena.text import normalize

SCHEMA_VERSION = "1.0"
_HEX_PATTERN = re.compile(r"[0-9a-f]{8,64}")


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


DOCUMENT_ID_PREFIX = "doc"


@dataclass(frozen=True)
class Citation:
    """A half-open character span in a corpus document plus the quoted text.

    ``start``/``end`` are offsets into the document text and ``quote`` is what
    the claim asserts the document says. The schema enforces the cheap structural
    rules (ordered offsets, non-blank quote, quote no longer than the span);
    whether the quote really occurs at those offsets is the grounding verifier's
    job, because that needs the corpus.
    """

    document_id: str
    start: int
    end: int
    quote: str

    def __post_init__(self) -> None:
        if not is_valid_id(self.document_id) or not self.document_id.startswith(
            f"{DOCUMENT_ID_PREFIX}_"
        ):
            raise ValidationError(
                "citation document id is malformed",
                document_id=self.document_id,
                expected_prefix=f"{DOCUMENT_ID_PREFIX}_",
            )
        if self.start < 0:
            raise ValidationError("citation start must be >= 0", start=self.start)
        if self.end <= self.start:
            raise ValidationError(
                "citation span must be non-empty", start=self.start, end=self.end
            )
        if not self.quote.strip():
            raise ValidationError("citation quote must not be blank")
        if len(self.quote) > self.end - self.start:
            raise ValidationError(
                "citation quote is longer than its span",
                quote_length=len(self.quote),
                span_length=self.end - self.start,
            )

    @property
    def length(self) -> int:
        """Number of characters covered by the span."""
        return self.end - self.start

    def key(self) -> tuple[str, int, int]:
        """Return the identity used to deduplicate citations."""
        return (self.document_id, self.start, self.end)

    def overlaps(self, other: Citation) -> bool:
        """True when both citations touch the same document and character range."""
        return (
            self.document_id == other.document_id
            and self.start < other.end
            and other.start < self.end
        )


PROVENANCE_ORIGINS = frozenset({"synthetic", "agent", "user", "evolved"})


@dataclass(frozen=True)
class Provenance:
    """Where a record came from, with enough detail to replay its creation.

    ``origin`` is one of :data:`PROVENANCE_ORIGINS`. Agent-produced records must
    name their agent; evolved records must name their parents. ``seed`` and
    ``corpus_hash`` make a synthetic record reproducible from its inputs alone.
    """

    origin: str
    agent_id: str | None = None
    generation: int = 0
    parents: tuple[str, ...] = ()
    seed: int | None = None
    corpus_hash: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in PROVENANCE_ORIGINS:
            raise ValidationError(
                "unknown provenance origin",
                origin=self.origin,
                allowed=sorted(PROVENANCE_ORIGINS),
            )
        if self.origin == "agent" and not (self.agent_id or "").strip():
            raise ValidationError("agent provenance requires an agent_id")
        if self.origin == "evolved" and not self.parents:
            raise ValidationError("evolved provenance requires at least one parent")
        if self.generation < 0:
            raise ValidationError("generation must be >= 0", generation=self.generation)
        for parent in self.parents:
            if not is_valid_id(parent):
                raise ValidationError(
                    "provenance parent id is malformed", parent=parent
                )
        if self.corpus_hash is not None and not _HEX_PATTERN.fullmatch(
            self.corpus_hash
        ):
            raise ValidationError(
                "corpus_hash must be 8-64 hex characters", corpus_hash=self.corpus_hash
            )
        if self.notes is not None and not self.notes.strip():
            raise ValidationError("provenance notes must not be blank when present")

    @property
    def is_reproducible(self) -> bool:
        """True when a seed and corpus hash are recorded for synthetic origins."""
        return (
            self.origin == "synthetic"
            and self.seed is not None
            and bool(self.corpus_hash)
        )
