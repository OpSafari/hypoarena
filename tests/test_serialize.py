"""Serialization of value objects: golden dicts, roundtrips and rejections."""

from __future__ import annotations

import pytest

from hypoarena.errors import SchemaError, ValidationError
from hypoarena.schema import Citation, Provenance, Scope
from hypoarena.serialize import (
    citation_from_dict,
    citation_to_dict,
    provenance_from_dict,
    provenance_to_dict,
    scope_from_dict,
    scope_to_dict,
)


def test_scope_roundtrips_and_pins_its_dict_shape() -> None:
    scope = Scope("hek293 cells", ("hypoxia",))
    assert scope_to_dict(scope) == {
        "population": "hek293 cells",
        "conditions": ["hypoxia"],
    }
    assert scope_from_dict(scope_to_dict(scope)) == scope


def test_scope_from_dict_rejects_unknown_keys_and_bad_payloads() -> None:
    with pytest.raises(SchemaError, match="unknown keys"):
        scope_from_dict({"population": "cells", "extra": 1})
    with pytest.raises(SchemaError, match="missing"):
        scope_from_dict({"conditions": []})
    with pytest.raises(ValidationError):
        scope_from_dict({"population": " ", "conditions": []})


def test_citation_roundtrips_and_pins_its_dict_shape() -> None:
    citation = Citation("doc_0123456789ab", 10, 25, "kinase binds")
    assert citation_to_dict(citation) == {
        "document_id": "doc_0123456789ab",
        "start": 10,
        "end": 25,
        "quote": "kinase binds",
    }
    assert citation_from_dict(citation_to_dict(citation)) == citation


def test_citation_from_dict_enforces_offset_bounds() -> None:
    payload = citation_to_dict(Citation("doc_0123456789ab", 10, 25, "kinase binds"))
    payload["start"] = -1
    with pytest.raises(SchemaError, match="below the minimum"):
        citation_from_dict(payload)
    payload["start"] = 30
    payload["end"] = 25
    with pytest.raises(ValidationError, match="non-empty"):
        citation_from_dict(payload)


def test_provenance_roundtrips_with_explicit_nulls() -> None:
    provenance = Provenance(origin="synthetic", seed=7, corpus_hash="0123456789abcdef")
    encoded = provenance_to_dict(provenance)
    assert encoded == {
        "origin": "synthetic",
        "agent_id": None,
        "generation": 0,
        "parents": [],
        "seed": 7,
        "corpus_hash": "0123456789abcdef",
        "notes": None,
    }
    assert provenance_from_dict(encoded) == provenance


def test_provenance_from_dict_defers_cross_field_rules_to_the_constructor() -> None:
    payload = provenance_to_dict(Provenance(origin="user"))
    payload["origin"] = "agent"
    with pytest.raises(ValidationError, match="agent_id"):
        provenance_from_dict(payload)
    payload["origin"] = "oracle"
    with pytest.raises(ValidationError, match="unknown provenance origin"):
        provenance_from_dict(payload)
