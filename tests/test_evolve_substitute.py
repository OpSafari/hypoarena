"""Variable substitution: statement rewriting and rejection paths."""

from __future__ import annotations

import pytest

from helpers import sample_claim
from hypoarena.errors import ValidationError
from hypoarena.evolve import substitute_variable


def test_substituting_the_subject_rewrites_the_statement() -> None:
    parent = sample_claim(
        statement="Protein A increases the expression of gene B",
        subject="protein A",
        object="gene B expression",
    )
    child = substitute_variable(parent, "subject", "kinase K1")
    assert child.subject == "kinase K1"
    assert child.statement == "kinase K1 increases the expression of gene B"
    assert child.object == parent.object


def test_substituting_the_object_keeps_the_subject() -> None:
    parent = sample_claim(
        statement="Protein A increases gene B expression",
        subject="protein A",
        object="gene B expression",
    )
    child = substitute_variable(parent, "object", "cell growth")
    assert child.object == "cell growth"
    assert child.subject == "protein A"
    assert child.statement == "Protein A increases cell growth"


def test_statements_are_left_alone_when_the_variable_is_not_verbatim() -> None:
    # documented limitation: rewriting only happens for a literal mention, so a
    # rephrased statement keeps its wording and only the variables change
    parent = sample_claim(
        statement="The promoter of gene B is bound after treatment",
        subject="protein A",
        object="gene B expression",
    )
    child = substitute_variable(parent, "object", "cell growth")
    assert child.object == "cell growth"
    assert child.statement == parent.statement


def test_the_child_has_its_own_identity_and_provenance() -> None:
    parent = sample_claim()
    child = substitute_variable(parent, "subject", "kinase K1")
    assert child.claim_id != parent.claim_id
    assert child.provenance.notes == "substitute_variable"
    assert child.provenance.parents == (parent.claim_id,)
    assert child.citations == parent.citations


def test_substitution_is_deterministic() -> None:
    parent = sample_claim()
    assert substitute_variable(parent, "subject", "kinase K1") == (
        substitute_variable(parent, "subject", "kinase K1")
    )


def test_invalid_substitutions_are_rejected() -> None:
    parent = sample_claim()
    with pytest.raises(ValidationError, match="unknown variable slot"):
        substitute_variable(parent, "relation", "kinase K1")
    with pytest.raises(ValidationError, match="blank"):
        substitute_variable(parent, "subject", "  ")
    with pytest.raises(ValidationError, match="must change"):
        substitute_variable(parent, "subject", "Protein A")


def test_self_referential_results_are_rejected_by_the_schema() -> None:
    parent = sample_claim(subject="protein A", object="gene B expression")
    with pytest.raises(ValidationError, match="must differ"):
        substitute_variable(parent, "object", "protein A")
