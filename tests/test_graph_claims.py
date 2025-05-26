"""Graph claim storage: insertion, lookup, ordering and duplicate rejection."""

from __future__ import annotations

import pytest

from helpers import CLAIM_ID, OTHER_CLAIM_ID, sample_claim
from hypoarena.errors import DuplicateIdError, UnknownReferenceError, ValidationError
from hypoarena.graph import HypothesisGraph


def test_empty_graph_has_no_claims() -> None:
    graph = HypothesisGraph()
    assert len(graph) == 0
    assert graph.claims == ()
    assert graph.claim_ids == ()
    assert CLAIM_ID not in graph


def test_add_claim_stores_and_returns_the_record() -> None:
    graph = HypothesisGraph()
    claim = sample_claim()
    assert graph.add_claim(claim) is claim
    assert graph.has_claim(CLAIM_ID) is True
    assert graph.claim(CLAIM_ID) == claim
    assert CLAIM_ID in graph


def test_duplicate_claim_ids_are_rejected_even_when_identical() -> None:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    with pytest.raises(DuplicateIdError) as info:
        graph.add_claim(sample_claim(statement="a different statement"))
    assert info.value.identifier == CLAIM_ID
    assert info.value.kind == "claim"
    assert len(graph) == 1


def test_non_claim_objects_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Claim objects"):
        HypothesisGraph().add_claim("clm_0123456789ab")  # type: ignore[arg-type]


def test_lookup_of_a_missing_claim_reports_the_identifier() -> None:
    with pytest.raises(UnknownReferenceError) as info:
        HypothesisGraph().claim(CLAIM_ID)
    assert info.value.identifier == CLAIM_ID
    assert info.value.kind == "claim"


def test_claims_iterate_in_sorted_identifier_order() -> None:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_claim(sample_claim(claim_id=CLAIM_ID))
    graph.add_claim(sample_claim(claim_id="clm_222222222222"))
    assert graph.claim_ids == (CLAIM_ID, "clm_222222222222", OTHER_CLAIM_ID)
    assert [claim.claim_id for claim in graph.claims] == list(graph.claim_ids)


def test_contains_rejects_non_string_keys() -> None:
    assert 12 not in HypothesisGraph()
