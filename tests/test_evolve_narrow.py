"""Scope narrowing: stricter scope, inherited evidence, new identity."""

from __future__ import annotations

import pytest

from helpers import sample_claim
from hypoarena.errors import ValidationError
from hypoarena.evolve import narrow_scope


def test_the_child_scope_is_strictly_narrower() -> None:
    parent = sample_claim()
    child = narrow_scope(parent, "hypoxia")
    assert child.scope.is_narrower_than(parent.scope)
    assert child.scope.conditions == parent.scope.conditions + ("hypoxia",)
    assert child.statement.endswith("under hypoxia")


def test_the_child_is_a_new_claim_with_evolved_provenance() -> None:
    parent = sample_claim()
    child = narrow_scope(parent, "hypoxia")
    assert child.claim_id != parent.claim_id
    assert child.provenance.origin == "evolved"
    assert child.provenance.parents == (parent.claim_id,)
    assert child.provenance.notes == "narrow_scope"
    assert child.provenance.generation == parent.provenance.generation + 1


def test_evidence_and_variables_are_inherited() -> None:
    parent = sample_claim()
    child = narrow_scope(parent, "hypoxia")
    assert child.citations == parent.citations
    assert (child.subject, child.object, child.relation) == (
        parent.subject,
        parent.object,
        parent.relation,
    )


def test_narrowing_is_deterministic_and_idempotent() -> None:
    parent = sample_claim()
    first = narrow_scope(parent, "hypoxia")
    assert narrow_scope(parent, "hypoxia") == first
    # narrowing by a condition that is already present returns the parent
    assert narrow_scope(first, "hypoxia") is first


def test_blank_conditions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        narrow_scope(sample_claim(), "   ")


def test_repeated_narrowing_accumulates_conditions() -> None:
    child = narrow_scope(narrow_scope(sample_claim(), "hypoxia"), "serum starved")
    assert child.scope.conditions == ("hypoxia", "serum starved")
    assert child.provenance.generation == sample_claim().provenance.generation + 2
