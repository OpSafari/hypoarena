"""Evolution records: identifiers, provenance and validation."""

from __future__ import annotations

import pytest

from helpers import sample_claim
from hypoarena.errors import ValidationError
from hypoarena.evolve import (
    OPERATORS,
    EvolutionRecord,
    Rejection,
    evolved_claim_id,
    evolved_provenance,
)


def test_operators_are_golden() -> None:
    assert OPERATORS == (
        "narrow_scope",
        "substitute_variable",
        "flip_relation",
        "crossover",
        "decompose",
    )


def test_unknown_operators_are_rejected_everywhere() -> None:
    with pytest.raises(ValidationError, match="unknown evolution operator"):
        evolved_claim_id("teleport", ["clm_0123456789ab"], "x")
    with pytest.raises(ValidationError, match="unknown evolution operator"):
        evolved_provenance("teleport", [sample_claim()])


def test_evolved_ids_are_deterministic_and_parent_scoped() -> None:
    first = evolved_claim_id("narrow_scope", ["clm_0123456789ab"], "a statement")
    assert first == evolved_claim_id(
        "narrow_scope", ["clm_0123456789ab"], "a statement"
    )
    assert first.startswith("clm_")
    assert first != evolved_claim_id("crossover", ["clm_0123456789ab"], "a statement")
    assert first != evolved_claim_id(
        "narrow_scope", ["clm_ffffffffffff"], "a statement"
    )
    with pytest.raises(ValidationError, match="at least one parent"):
        evolved_claim_id("narrow_scope", [], "x")


def test_provenance_counts_generations_from_the_parents() -> None:
    parent = sample_claim()
    provenance = evolved_provenance("narrow_scope", [parent])
    assert provenance.origin == "evolved"
    assert provenance.parents == (parent.claim_id,)
    assert provenance.generation == parent.provenance.generation + 1
    assert provenance.notes == "narrow_scope"
    assert provenance.corpus_hash == parent.provenance.corpus_hash


def test_crossover_provenance_takes_the_deepest_parent() -> None:
    first = sample_claim()
    second = sample_claim(
        claim_id="clm_ffffffffffff",
        provenance=sample_claim().provenance,
    )
    deep = evolved_provenance("crossover", [first, second])
    assert set(deep.parents) == {first.claim_id, second.claim_id}
    assert deep.generation == 2


def test_records_validate_their_content() -> None:
    child = sample_claim(claim_id="clm_222222222222")
    record = EvolutionRecord("narrow_scope", (child.claim_id,), child, "scope narrowed")
    assert record.as_dict()["operator"] == "narrow_scope"
    assert record.as_dict()["child_id"] == child.claim_id
    with pytest.raises(ValidationError, match="rationale"):
        EvolutionRecord("narrow_scope", (child.claim_id,), child, "  ")
    with pytest.raises(ValidationError, match="at least one parent"):
        EvolutionRecord("narrow_scope", (), child, "x")


def test_rejections_carry_their_evidence() -> None:
    rejection = Rejection(
        "crossover",
        ("clm_0123456789ab",),
        "not novel",
        statement="protein A increases cell growth",
        nearest="clm_ffffffffffff",
        similarity=0.93,
    )
    payload = rejection.as_dict()
    assert payload["reason"] == "not novel"
    assert payload["similarity"] == 0.93
    assert payload["nearest"] == "clm_ffffffffffff"
    with pytest.raises(ValidationError, match="reason"):
        Rejection("crossover", ("clm_0123456789ab",), " ")
