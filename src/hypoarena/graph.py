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
    UnknownReferenceError,
    ValidationError,
)
from hypoarena.ids import is_valid_id
from hypoarena.schema import (
    CLAIM_ID_PREFIX,
    Claim,
    ClaimRelation,
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
