"""Decomposition: splitting compound statements and refusing simple ones."""

from __future__ import annotations

from helpers import sample_claim
from hypoarena.evolve import decompose, split_statement
from hypoarena.text import normalize


def test_splitting_covers_every_documented_marker() -> None:
    assert split_statement("A binds B and B drives C") == ("A binds B", "B drives C")
    assert split_statement("A binds B; B drives C") == ("A binds B", "B drives C")
    assert split_statement("A binds B as well as C") == ("A binds B", "C")
    assert split_statement("A binds B") == ("A binds B",)


def test_splitting_normalizes_whitespace_and_drops_empty_parts() -> None:
    assert split_statement("A binds B  and   B drives C") == ("A binds B", "B drives C")
    assert split_statement("   ") == ()


def test_a_compound_claim_yields_one_child_per_conjunct() -> None:
    parent = sample_claim(
        statement="Protein A binds the promoter and gene B is expressed"
    )
    children = decompose(parent)
    assert len(children) == 2
    assert children[0].statement == "Protein A binds the promoter"
    assert children[1].statement == "gene B is expressed"


def test_children_inherit_variables_scope_and_citations() -> None:
    parent = sample_claim(
        statement="Protein A binds the promoter and gene B is expressed"
    )
    for child in decompose(parent):
        assert child.subject == parent.subject
        assert child.object == parent.object
        assert child.relation is parent.relation
        assert child.scope == parent.scope
        assert child.citations == parent.citations
        assert child.provenance.notes == "decompose"
        assert child.provenance.parents == (parent.claim_id,)


def test_a_simple_claim_is_not_decomposed() -> None:
    assert decompose(sample_claim(statement="Protein A increases cell growth")) == ()


def test_children_are_distinct_and_deterministic() -> None:
    parent = sample_claim(
        statement="Protein A binds the promoter and gene B is expressed"
    )
    children = decompose(parent)
    assert len({child.claim_id for child in children}) == 2
    assert decompose(parent) == children


def test_duplicated_conjuncts_do_not_produce_children() -> None:
    parent = sample_claim(statement="Protein A binds B and protein A binds B")
    parts = split_statement(parent.statement)
    assert len(parts) == 2
    assert normalize(parts[0]) == normalize(parts[1])
    assert decompose(parent) == ()
