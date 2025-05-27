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
        self._edges: dict[str, list[ClaimEdge]] = {}
        self._incoming: dict[str, list[ClaimEdge]] = {}

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

    def add_edge(
        self,
        source: str,
        target: str,
        relation: ClaimRelation,
        note: str | None = None,
    ) -> ClaimEdge:
        """Insert a typed relation between two claims that must already exist.

        Duplicate edges are rejected rather than merged, and ``contradicts`` is
        treated as symmetric: adding ``b contradicts a`` after ``a contradicts b``
        is the same assertion, not a new one.
        """
        self.claim(source)
        self.claim(target)
        edge = ClaimEdge(source, target, relation, note=note)
        if self.has_edge(source, target, relation):
            raise GraphInvariantError("edge already exists", edge=list(edge.key()))
        if relation is ClaimRelation.CONTRADICTS and self.has_edge(
            target, source, relation
        ):
            raise GraphInvariantError(
                "contradiction edges are symmetric", edge=list(edge.key())
            )
        self._edges.setdefault(source, []).append(edge)
        self._incoming.setdefault(target, []).append(edge)
        return edge

    def has_edge(self, source: str, target: str, relation: ClaimRelation) -> bool:
        """True when exactly this typed relation is present."""
        wanted = (source, target, relation.value)
        return any(edge.key() == wanted for edge in self._edges.get(source, ()))

    def edge_between(
        self, source: str, target: str, relation: ClaimRelation
    ) -> ClaimEdge:
        """Return the matching edge or raise ``UnknownReferenceError``."""
        for edge in self._edges.get(source, ()):
            if edge.key() == (source, target, relation.value):
                return edge
        raise UnknownReferenceError(
            f"{source}->{target}:{relation.value}", "edge", owner=source
        )

    @property
    def edges(self) -> tuple[ClaimEdge, ...]:
        """All edges in canonical (source, target, relation) order."""
        return tuple(
            sorted(
                (edge for group in self._edges.values() for edge in group),
                key=lambda edge: edge.key(),
            )
        )

    @property
    def edge_count(self) -> int:
        """Number of typed claim relations."""
        return sum(len(group) for group in self._edges.values())

    def remove_edge(self, edge: ClaimEdge) -> None:
        """Delete an edge from both indexes."""
        group = self._edges.get(edge.source, [])
        if edge not in group:
            raise UnknownReferenceError(
                "->".join(edge.endpoints()), "edge", owner=edge.source
            )
        group.remove(edge)
        self._incoming[edge.target].remove(edge)

    def outgoing(
        self, claim_id: str, relation: ClaimRelation | None = None
    ) -> tuple[ClaimEdge, ...]:
        """Edges starting at ``claim_id``, optionally filtered by relation."""
        self.claim(claim_id)
        edges = self._edges.get(claim_id, ())
        if relation is not None:
            edges = [edge for edge in edges if edge.relation is relation]
        return tuple(sorted(edges, key=lambda edge: edge.key()))

    def incoming(
        self, claim_id: str, relation: ClaimRelation | None = None
    ) -> tuple[ClaimEdge, ...]:
        """Edges ending at ``claim_id``, optionally filtered by relation."""
        self.claim(claim_id)
        edges = self._incoming.get(claim_id, ())
        if relation is not None:
            edges = [edge for edge in edges if edge.relation is relation]
        return tuple(sorted(edges, key=lambda edge: edge.key()))

    def targets(
        self, claim_id: str, relation: ClaimRelation | None = None
    ) -> tuple[str, ...]:
        """Sorted identifiers reachable in one outgoing step."""
        return tuple(
            sorted({edge.target for edge in self.outgoing(claim_id, relation)})
        )

    def sources(
        self, claim_id: str, relation: ClaimRelation | None = None
    ) -> tuple[str, ...]:
        """Sorted identifiers with an incoming step into ``claim_id``."""
        return tuple(
            sorted({edge.source for edge in self.incoming(claim_id, relation)})
        )

    def neighbors(self, claim_id: str) -> tuple[str, ...]:
        """Sorted identifiers connected in either direction."""
        return tuple(sorted(set(self.targets(claim_id)) | set(self.sources(claim_id))))

    def has_path(
        self,
        source: str,
        target: str,
        relations: frozenset[ClaimRelation] | None = None,
    ) -> bool:
        """True when ``target`` is reachable from ``source`` over outgoing edges.

        Traversal is breadth-first over a deterministic frontier (sorted
        identifiers), so the result never depends on insertion order. A relation
        filter restricts which edge types may be used; ``source == target`` is
        reported as reachable only when a real cycle exists.
        """
        self.claim(source)
        self.claim(target)
        frontier = [source]
        seen = {source}
        while frontier:
            current = frontier.pop(0)
            for edge in self.outgoing(current):
                if relations is not None and edge.relation not in relations:
                    continue
                if edge.target == target:
                    return True
                if edge.target not in seen:
                    seen.add(edge.target)
                    frontier.append(edge.target)
        return False

    def roots(self) -> tuple[str, ...]:
        """Claims with no incoming relations, sorted."""
        return tuple(cid for cid in self.claim_ids if not self._incoming.get(cid))

    def leaves(self) -> tuple[str, ...]:
        """Claims with no outgoing relations, sorted."""
        return tuple(cid for cid in self.claim_ids if not self._edges.get(cid))
