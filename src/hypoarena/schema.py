"""Versioned schemas for claims, citations, evidence and provenance.

Design rules used throughout this module:

* every record is a frozen dataclass, so hashes and equality are structural;
* ``__post_init__`` performs validation, so an invalid record cannot exist —
  constructors raise instead of returning a broken object;
* serialization lives in :mod:`hypoarena.serialize`, which is the single place
  that knows the wire format and rejects unknown keys;
* only top-level records (claims, evidence) carry ``schema_version``; nested
  value objects stay version-free to keep payloads small.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum, unique

from hypoarena.errors import ValidationError
from hypoarena.ids import canonical_json, content_hash, is_valid_id
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


CLAIM_ID_PREFIX = "clm"
EVIDENCE_ID_PREFIX = "evd"


def check_record_id(identifier: str, prefix: str, kind: str) -> None:
    """Validate a record id: well formed and carrying the expected prefix."""
    if not is_valid_id(identifier) or not identifier.startswith(f"{prefix}_"):
        raise ValidationError(
            f"{kind} id is malformed",
            identifier=identifier,
            expected_prefix=f"{prefix}_",
        )


def check_unique_citations(citations: Sequence[Citation], kind: str) -> None:
    """Reject repeated spans inside one record's citation list."""
    seen: set[tuple[str, int, int]] = set()
    for citation in citations:
        if not isinstance(citation, Citation):
            raise ValidationError(
                f"{kind} citations must be Citation objects", kind=kind
            )
        if citation.key() in seen:
            raise ValidationError(
                f"{kind} cites the same span twice",
                kind=kind,
                span=list(citation.key()),
            )
        seen.add(citation.key())


@dataclass(frozen=True)
class Claim:
    """A falsifiable statement relating two variables within a scope.

    ``subject`` and ``object`` are the claim's variables (for example a gene and
    a phenotype); ``relation`` is the predicted direction between them. A claim
    may be uncited — the grounding verifier grades that instead of the schema
    rejecting it, because proposals start ungrounded and earn citations later.
    """

    claim_id: str
    statement: str
    subject: str
    object: str
    relation: PredictedRelation
    scope: Scope
    citations: tuple[Citation, ...] = ()
    mechanism: str | None = None
    provenance: Provenance = field(default_factory=lambda: Provenance(origin="user"))

    def __post_init__(self) -> None:
        check_record_id(self.claim_id, CLAIM_ID_PREFIX, "claim")
        if not self.statement.strip():
            raise ValidationError(
                "claim statement must not be blank", claim_id=self.claim_id
            )
        if not self.subject.strip() or not self.object.strip():
            raise ValidationError(
                "claim variables must not be blank", claim_id=self.claim_id
            )
        if normalize(self.subject) == normalize(self.object):
            raise ValidationError(
                "claim subject and object must differ",
                claim_id=self.claim_id,
                subject=self.subject,
            )
        if not isinstance(self.relation, PredictedRelation):
            raise ValidationError(
                "claim relation must be a PredictedRelation",
                claim_id=self.claim_id,
                got=type(self.relation).__name__,
            )
        if not isinstance(self.scope, Scope):
            raise ValidationError(
                "claim scope must be a Scope",
                claim_id=self.claim_id,
                got=type(self.scope).__name__,
            )
        check_unique_citations(self.citations, "claim")
        if self.mechanism is not None and not self.mechanism.strip():
            raise ValidationError(
                "claim mechanism must not be blank when present", claim_id=self.claim_id
            )
        if not isinstance(self.provenance, Provenance):
            raise ValidationError(
                "claim provenance must be a Provenance", claim_id=self.claim_id
            )

    @property
    def variables(self) -> tuple[str, str]:
        """The ordered variable pair this claim relates."""
        return (self.subject, self.object)

    @property
    def is_cited(self) -> bool:
        """True when at least one corpus span backs the claim."""
        return bool(self.citations)

    def normalized_statement(self) -> str:
        """Return the statement after the standard normalization chain."""
        return normalize(self.statement)

    def signature(self) -> str:
        """Return a content signature for dedup and novelty checks.

        The signature covers the normalized statement, both variables, the
        relation and the scope signature — deliberately not the citations or the
        identifier, so restating the same hypothesis with new sources still
        counts as a duplicate.
        """
        return content_hash(
            {
                "statement": normalize(self.statement),
                "subject": normalize(self.subject),
                "object": normalize(self.object),
                "relation": self.relation.value,
                "scope": self.scope.signature(),
            }
        )


@dataclass(frozen=True)
class Evidence:
    """A graded observation drawn from the corpus.

    ``strength`` is a confidence weight in ``[0, 1]``; ``polarity`` says whether
    the observation supports or refutes the claim it is linked to. Neutral
    evidence must carry zero strength, because a neutral item contributes no
    weight to belief updating and pretending otherwise would silently bias the
    posterior. Every evidence item must cite at least one span: unverifiable
    assertions belong in a claim, not in the evidence set.
    """

    evidence_id: str
    statement: str
    polarity: EvidencePolarity
    strength: float
    citations: tuple[Citation, ...]
    method: str
    provenance: Provenance
    effect_size: float | None = None
    sample_size: int | None = None

    def __post_init__(self) -> None:
        check_record_id(self.evidence_id, EVIDENCE_ID_PREFIX, "evidence")
        if not self.statement.strip():
            raise ValidationError(
                "evidence statement must not be blank", evidence_id=self.evidence_id
            )
        if not isinstance(self.polarity, EvidencePolarity):
            raise ValidationError(
                "evidence polarity must be an EvidencePolarity",
                evidence_id=self.evidence_id,
                got=type(self.polarity).__name__,
            )
        if not isinstance(self.strength, float) or not math.isfinite(self.strength):
            raise ValidationError(
                "evidence strength must be a finite float",
                evidence_id=self.evidence_id,
                strength=self.strength,
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValidationError(
                "evidence strength must lie within [0, 1]",
                evidence_id=self.evidence_id,
                strength=self.strength,
            )
        if self.polarity is EvidencePolarity.NEUTRAL and self.strength != 0.0:
            raise ValidationError(
                "neutral evidence must have zero strength",
                evidence_id=self.evidence_id,
                strength=self.strength,
            )
        if not self.citations:
            raise ValidationError(
                "evidence must cite at least one corpus span",
                evidence_id=self.evidence_id,
            )
        check_unique_citations(self.citations, "evidence")
        if not self.method.strip():
            raise ValidationError(
                "evidence method must not be blank", evidence_id=self.evidence_id
            )
        if not isinstance(self.provenance, Provenance):
            raise ValidationError(
                "evidence provenance must be a Provenance",
                evidence_id=self.evidence_id,
            )
        if self.sample_size is not None and self.sample_size < 1:
            raise ValidationError(
                "evidence sample_size must be >= 1",
                evidence_id=self.evidence_id,
                sample_size=self.sample_size,
            )
        if self.effect_size is not None and not math.isfinite(self.effect_size):
            raise ValidationError(
                "evidence effect_size must be finite",
                evidence_id=self.evidence_id,
                effect_size=self.effect_size,
            )

    @property
    def weighted_polarity(self) -> float:
        """Signed weight in ``[-1, 1]`` used by belief accumulation."""
        if self.polarity is EvidencePolarity.SUPPORT:
            return self.strength
        if self.polarity is EvidencePolarity.REFUTE:
            return -self.strength
        return 0.0


CANONICAL_RELATION_VERBS: dict[PredictedRelation, str] = {
    PredictedRelation.INCREASES: "increases",
    PredictedRelation.DECREASES: "decreases",
    PredictedRelation.ENABLES: "enables",
    PredictedRelation.INHIBITS: "inhibits",
    PredictedRelation.CAUSES: "causes",
    PredictedRelation.ASSOCIATES: "is associated with",
}


def canonical_verb(relation: PredictedRelation) -> str:
    """Return the deterministic surface verb for a relation.

    Living in the schema (rather than in the corpus generator) matters: evolution
    operators need the same verb table when they rewrite a statement, and a
    second copy would be free to drift.
    """
    return CANONICAL_RELATION_VERBS[relation]


def canonical_statement(subject: str, relation: PredictedRelation, target: str) -> str:
    """Return the canonical statement for a subject/relation/target triple."""
    if not subject.strip() or not target.strip():
        raise ValidationError("canonical statement needs non-blank variables")
    return f"{subject} {canonical_verb(relation)} {target}"
