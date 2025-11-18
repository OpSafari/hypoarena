"""ClaimEdge: construction rules, identity and endpoint helpers."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.graph import ClaimEdge
from hypoarena.schema import ClaimRelation

FIRST = "clm_0123456789ab"
SECOND = "clm_ffffffffffff"


def test_edge_accepts_valid_endpoints_and_relation() -> None:
    edge = ClaimEdge(FIRST, SECOND, ClaimRelation.ENTAILS)
    assert edge.key() == (FIRST, SECOND, "entails")
    assert edge.endpoints() == (FIRST, SECOND)
    assert edge.note is None


def test_edge_rejects_malformed_and_non_claim_endpoint_ids() -> None:
    with pytest.raises(ValidationError, match="must be a claim id") as info:
        ClaimEdge("nope", SECOND, ClaimRelation.REFINES)
    assert info.value.details["side"] == "source"
    with pytest.raises(ValidationError, match="must be a claim id") as info:
        ClaimEdge(FIRST, "evd_0123456789ab", ClaimRelation.REFINES)
    assert info.value.details["side"] == "target"
    assert info.value.details["expected_prefix"] == "clm_"


def test_edge_rejects_self_loops_and_wrong_relation_types() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        ClaimEdge(FIRST, FIRST, ClaimRelation.CONTRADICTS)
    with pytest.raises(ValidationError, match="ClaimRelation"):
        ClaimEdge(FIRST, SECOND, "entails")  # type: ignore[arg-type]


def test_edge_rejects_blank_notes_but_allows_none() -> None:
    assert ClaimEdge(FIRST, SECOND, ClaimRelation.ENTAILS, note=None).note is None
    assert ClaimEdge(FIRST, SECOND, ClaimRelation.ENTAILS, note="scope narrowed").note
    with pytest.raises(ValidationError, match="note"):
        ClaimEdge(FIRST, SECOND, ClaimRelation.ENTAILS, note="  ")


def test_touches_covers_both_endpoints_only() -> None:
    edge = ClaimEdge(FIRST, SECOND, ClaimRelation.ENTAILS)
    assert edge.touches(FIRST) and edge.touches(SECOND)
    assert not edge.touches("clm_1234567890ab")


def test_edges_compare_and_hash_structurally() -> None:
    first = ClaimEdge(FIRST, SECOND, ClaimRelation.ENTAILS, note="a")
    second = ClaimEdge(FIRST, SECOND, ClaimRelation.ENTAILS, note="a")
    assert first == second
    assert len({first, second}) == 1
    assert first != ClaimEdge(FIRST, SECOND, ClaimRelation.CONTRADICTS, note="a")
