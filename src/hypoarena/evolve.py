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
from dataclasses import dataclass, replace

from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    make_id,
)
from hypoarena.schema import (
    Citation,
    Claim,
    Provenance,
    Scope,
    canonical_verb,
    opposite_relation,
)
from hypoarena.text import (
    normalize,
    normalize_whitespace,
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


def narrow_scope(claim: Claim, condition: str, *, seed: int | None = None) -> Claim:
    """Return a child claim restricted by an additional scope condition.

    Narrowing is the safest way to make a hypothesis more testable: the child
    says strictly less, so it cannot contradict its parent. Citations are
    inherited unchanged — the child still rests on the same evidence, which is
    exactly why the grounding verifier should be re-run after evolving.

    Narrowing by a condition the claim already carries is a no-op and returns the
    parent unchanged, so repeated application cannot pile up redundant children.
    """
    scope = claim.scope.narrowed(condition)
    if scope is claim.scope:
        return claim
    statement = f"{claim.statement} under {condition}"
    child = replace(
        claim,
        claim_id=evolved_claim_id(
            "narrow_scope", (claim.claim_id,), statement, extra=condition
        ),
        statement=statement,
        scope=scope,
        provenance=evolved_provenance("narrow_scope", [claim], seed=seed),
    )
    if not child.scope.is_narrower_than(claim.scope):
        raise ValidationError(  # pragma: no cover - defensive invariant
            "narrowing did not produce a stricter scope", claim_id=claim.claim_id
        )
    return child


def substitute_variable(
    claim: Claim, slot: str, replacement: str, *, seed: int | None = None
) -> Claim:
    """Return a child claim with one variable replaced.

    Substitution models the "does this hold for a related variable?" question —
    a paralogue, an isoform, a different readout. The statement is rewritten only
    when it mentions the replaced variable verbatim, so a child never carries a
    statement that contradicts its own variables.
    """
    if slot not in VARIABLE_SLOTS:
        raise ValidationError(
            "unknown variable slot", slot=slot, allowed=list(VARIABLE_SLOTS)
        )
    if not replacement.strip():
        raise ValidationError("replacement variable must not be blank")
    original = getattr(claim, slot)
    if normalize(original) == normalize(replacement):
        raise ValidationError(
            "substitution must change the variable",
            slot=slot,
            value=replacement,
        )
    statement = claim.statement
    if original.lower() in statement.lower():
        start = statement.lower().index(original.lower())
        statement = statement[:start] + replacement + statement[start + len(original) :]
    # A substitution that would make the claim self-referential is rejected by
    # the Claim constructor itself ("subject and object must differ"), so there
    # is no separate check here to fall out of sync with the schema.
    return replace(
        claim,
        claim_id=evolved_claim_id(
            "substitute_variable",
            (claim.claim_id,),
            statement,
            extra=f"{slot}:{replacement}",
        ),
        statement=statement,
        provenance=evolved_provenance("substitute_variable", [claim], seed=seed),
        **{slot: replacement},
    )


def flip_relation(claim: Claim, *, seed: int | None = None) -> Claim | None:
    """Return a child asserting the opposite relation, or ``None``.

    ``causes`` and ``associates`` have no opposite in this schema, so flipping
    them is refused rather than approximated: inventing a negation of a causal
    claim would produce a child the graph cannot relate to its parent.
    """
    opposite = opposite_relation(claim.relation)
    if opposite is None:
        return None
    original_verb = canonical_verb(claim.relation)
    statement = claim.statement
    lowered = statement.lower()
    if original_verb in lowered:
        start = lowered.index(original_verb)
        replacement = canonical_verb(opposite)
        if original_verb[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]
        statement = (
            statement[:start] + replacement + statement[start + len(original_verb) :]
        )
    else:
        statement = f"{claim.subject} {canonical_verb(opposite)} {claim.object}"
    return replace(
        claim,
        claim_id=evolved_claim_id(
            "flip_relation",
            (claim.claim_id,),
            statement,
            extra=opposite.value,
        ),
        statement=statement,
        relation=opposite,
        provenance=evolved_provenance("flip_relation", [claim], seed=seed),
    )


def shared_population(first: Claim, second: Claim) -> bool:
    """True when two claims are about the same population."""
    return normalize(first.scope.population) == normalize(second.scope.population)


def shared_variables(first: Claim, second: Claim) -> frozenset[str]:
    """Return the normalized variables the two claims have in common."""
    left = {normalize(first.subject), normalize(first.object)}
    right = {normalize(second.subject), normalize(second.object)}
    return frozenset(left & right)


def merge_citations(claims: Sequence[Claim]) -> tuple[Citation, ...]:
    """Return the union of citations, deduplicated by span and kept in order."""
    seen: set[tuple[str, int, int]] = set()
    merged: list[Citation] = []
    for claim in claims:
        for citation in claim.citations:
            if citation.key() in seen:
                continue
            seen.add(citation.key())
            merged.append(citation)
    return tuple(merged)


def crossover(first: Claim, second: Claim, *, seed: int | None = None) -> Claim | None:
    """Combine two parents into one child, or refuse when they are unrelated.

    The child keeps the first parent's variables, relation and population, takes
    a mechanism from whichever parent has one (preferring the second), and
    carries the union of both scope conditions and both citation sets. Crossover
    is refused when the parents share neither a variable nor a population, and
    also when the combination would relate a variable to itself: merging
    unrelated or degenerate claims would fabricate a hypothesis that follows
    from neither parent.
    """
    if first.claim_id == second.claim_id:
        return None
    if not shared_variables(first, second) or not shared_population(first, second):
        return None
    if normalize(first.subject) == normalize(second.object):
        # the child would relate a variable to itself; refuse instead of raising
        return None
    mechanism = second.mechanism or first.mechanism
    conditions = tuple(
        dict.fromkeys((*first.scope.conditions, *second.scope.conditions))
    )
    statement = f"{first.subject} {canonical_verb(first.relation)} {second.object}"
    if mechanism:
        statement = f"{statement} through {mechanism}"
    return Claim(
        claim_id=evolved_claim_id(
            "crossover", (first.claim_id, second.claim_id), statement
        ),
        statement=statement,
        subject=first.subject,
        object=second.object,
        relation=first.relation,
        scope=Scope(population=first.scope.population, conditions=conditions),
        citations=merge_citations((first, second)),
        mechanism=mechanism,
        provenance=evolved_provenance("crossover", [first, second], seed=seed),
    )


SPLIT_MARKERS: tuple[str, ...] = (" and ", "; ", " as well as ")


def split_statement(statement: str) -> tuple[str, ...]:
    """Split a compound statement into its conjuncts.

    The heuristic is deliberately simple and documented: split on ``and``,
    ``;`` and ``as well as``. It does not understand subordination, so a
    statement like "A binds B and thereby increases C" yields two parts of
    unequal usefulness — which is why every child keeps the parent's citations
    and has to be re-verified rather than trusted.
    """
    parts = [statement.strip()]
    for marker in SPLIT_MARKERS:
        expanded: list[str] = []
        for part in parts:
            expanded.extend(normalize_whitespace(piece) for piece in part.split(marker))
        parts = expanded
    return tuple(part for part in parts if part)


def decompose(claim: Claim, *, seed: int | None = None) -> tuple[Claim, ...]:
    """Split a compound claim into one child per conjunct.

    Children inherit the parent's variables, relation, scope and citations; only
    the statement changes. Fewer than two usable conjuncts means there is
    nothing to decompose, and an empty tuple is returned rather than a copy of
    the parent.
    """
    parts = split_statement(claim.statement)
    if len(parts) < 2:
        return ()
    children: list[Claim] = []
    seen: set[str] = set()
    whole = normalize(claim.statement)
    for index, part in enumerate(parts):
        key = normalize(part)
        # a conjunct that repeats the whole statement, or repeats an earlier
        # conjunct, would only add a duplicate claim
        if key == whole or key in seen:
            continue
        seen.add(key)
        children.append(
            replace(
                claim,
                claim_id=evolved_claim_id(
                    "decompose", (claim.claim_id,), part, extra=index
                ),
                statement=part,
                provenance=evolved_provenance("decompose", [claim], seed=seed),
            )
        )
    return tuple(children) if len(children) >= 2 else ()
