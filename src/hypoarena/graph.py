"""The hypothesis–evidence graph.

The graph is the shared state of a discovery run: claims as nodes, evidence
attached to the claims it bears on, and typed claim-to-claim relations
(``entails`` / ``contradicts`` / ``refines``). Structural rules are enforced at
mutation time *and* re-checkable in bulk via :meth:`HypothesisGraph.validate`, so
every operator in the toolkit can prove it preserved validity.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypoarena.errors import (
    DuplicateIdError,
    GraphInvariantError,
    UnknownReferenceError,
    ValidationError,
)
from hypoarena.ids import is_valid_id
from hypoarena.schema import (
    CLAIM_ID_PREFIX,
    Claim,
    ClaimRelation,
    Evidence,
)


@dataclass(frozen=True)
class ClaimEdge:
    """A typed directed relation between two claim identifiers."""

    source: str
    target: str
    relation: ClaimRelation
    note: str | None = None

    def __post_init__(self) -> None:
        for label, identifier in (("source", self.source), ("target", self.target)):
            if not is_valid_id(identifier) or not identifier.startswith(
                f"{CLAIM_ID_PREFIX}_"
            ):
                raise ValidationError(
                    f"edge {label} must be a claim id",
                    identifier=identifier,
                    side=label,
                    expected_prefix=f"{CLAIM_ID_PREFIX}_",
                )
        if not isinstance(self.relation, ClaimRelation):
            raise ValidationError(
                "edge relation must be a ClaimRelation",
                got=type(self.relation).__name__,
            )
        if self.source == self.target:
            raise ValidationError("edge endpoints must differ", source=self.source)
        if self.note is not None and not self.note.strip():
            raise ValidationError("edge note must not be blank when present")

    def key(self) -> tuple[str, str, str]:
        """Identity used to detect duplicate edges."""
        return (self.source, self.target, self.relation.value)

    def endpoints(self) -> tuple[str, str]:
        """The ordered endpoint pair, ignoring the relation."""
        return (self.source, self.target)

    def touches(self, claim_id: str) -> bool:
        """True when the edge starts or ends at ``claim_id``."""
        return claim_id in (self.source, self.target)


class HypothesisGraph:
    """Mutable collection of claims, evidence, links and typed claim relations.

    Identifiers are unique per record type: adding a second claim with an
    existing id raises :class:`~hypoarena.errors.DuplicateIdError` even when the
    payload is identical, because a silent overwrite would destroy provenance.
    Iteration order is always sorted by identifier, so serialized output and
    derived statistics do not depend on insertion order.
    """

    def __init__(self) -> None:
        self._claims: dict[str, Claim] = {}
        self._evidence: dict[str, Evidence] = {}
        self._links: dict[str, list[str]] = {}
        self._backlinks: dict[str, list[str]] = {}

    def add_claim(self, claim: Claim) -> Claim:
        """Insert ``claim`` and return it."""
        if not isinstance(claim, Claim):
            raise ValidationError(
                "graph claims must be Claim objects", got=type(claim).__name__
            )
        if claim.claim_id in self._claims:
            raise DuplicateIdError(claim.claim_id, "claim")
        self._claims[claim.claim_id] = claim
        return claim

    def has_claim(self, claim_id: str) -> bool:
        """True when a claim with this identifier is present."""
        return claim_id in self._claims

    def claim(self, claim_id: str) -> Claim:
        """Return the claim with ``claim_id`` or raise ``UnknownReferenceError``."""
        try:
            return self._claims[claim_id]
        except KeyError:
            raise UnknownReferenceError(claim_id, "claim") from None

    @property
    def claim_ids(self) -> tuple[str, ...]:
        """All claim identifiers in sorted order."""
        return tuple(sorted(self._claims))

    @property
    def claims(self) -> tuple[Claim, ...]:
        """All claims ordered by identifier."""
        return tuple(self._claims[claim_id] for claim_id in self.claim_ids)

    def __len__(self) -> int:
        return len(self._claims)

    def __contains__(self, claim_id: object) -> bool:
        return isinstance(claim_id, str) and claim_id in self._claims

    def add_evidence(self, evidence: Evidence) -> Evidence:
        """Insert ``evidence`` and return it."""
        if not isinstance(evidence, Evidence):
            raise ValidationError(
                "graph evidence must be Evidence objects", got=type(evidence).__name__
            )
        if evidence.evidence_id in self._evidence:
            raise DuplicateIdError(evidence.evidence_id, "evidence")
        self._evidence[evidence.evidence_id] = evidence
        return evidence

    def has_evidence(self, evidence_id: str) -> bool:
        """True when an evidence item with this identifier is present."""
        return evidence_id in self._evidence

    def evidence(self, evidence_id: str) -> Evidence:
        """Return the evidence item or raise ``UnknownReferenceError``."""
        try:
            return self._evidence[evidence_id]
        except KeyError:
            raise UnknownReferenceError(evidence_id, "evidence") from None

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """All evidence identifiers in sorted order."""
        return tuple(sorted(self._evidence))

    @property
    def evidence_items(self) -> tuple[Evidence, ...]:
        """All evidence items ordered by identifier."""
        return tuple(self._evidence[item] for item in self.evidence_ids)

    def link_evidence(self, claim_id: str, evidence_id: str) -> None:
        """Attach an evidence item to a claim.

        Both endpoints must already exist, and a link cannot be created twice:
        double counting one observation would silently inflate its weight during
        belief accumulation.
        """
        self.claim(claim_id)
        self.evidence(evidence_id)
        linked = self._links.setdefault(claim_id, [])
        if evidence_id in linked:
            raise GraphInvariantError(
                "evidence is already linked to this claim",
                claim_id=claim_id,
                evidence_id=evidence_id,
            )
        linked.append(evidence_id)
        self._backlinks.setdefault(evidence_id, []).append(claim_id)

    def unlink_evidence(self, claim_id: str, evidence_id: str) -> None:
        """Remove a link, raising when it does not exist."""
        linked = self._links.get(claim_id, [])
        if evidence_id not in linked:
            raise UnknownReferenceError(
                f"{claim_id}->{evidence_id}", "evidence link", owner=claim_id
            )
        linked.remove(evidence_id)
        self._backlinks[evidence_id].remove(claim_id)

    def evidence_for(self, claim_id: str) -> tuple[Evidence, ...]:
        """Evidence attached to a claim, in link order."""
        self.claim(claim_id)
        return tuple(self._evidence[item] for item in self._links.get(claim_id, ()))

    def claims_for_evidence(self, evidence_id: str) -> tuple[str, ...]:
        """Claim identifiers an evidence item is attached to, sorted."""
        self.evidence(evidence_id)
        return tuple(sorted(self._backlinks.get(evidence_id, ())))

    @property
    def link_count(self) -> int:
        """Total number of claim–evidence links."""
        return sum(len(items) for items in self._links.values())
