"""Graph encoding: golden shape, roundtrips and strict rejection."""

from __future__ import annotations

import pytest

from helpers import (
    CLAIM_ID,
    EVIDENCE_ID,
    OTHER_CLAIM_ID,
    sample_claim,
    sample_evidence,
)
from hypoarena.errors import DuplicateIdError, SchemaError, UnknownReferenceError
from hypoarena.graph import HypothesisGraph
from hypoarena.schema import ClaimRelation
from hypoarena.serialize import (
    edge_from_dict,
    edge_to_dict,
    graph_from_dict,
    graph_to_dict,
)


def small_graph() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_evidence(sample_evidence())
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS, note="derived")
    return graph


def test_graph_dict_has_the_expected_top_level_shape() -> None:
    payload = graph_to_dict(small_graph())
    assert sorted(payload) == [
        "claims",
        "edges",
        "evidence",
        "links",
        "schema_version",
    ]
    assert payload["schema_version"] == "1.0"
    assert len(payload["claims"]) == 2
    assert payload["links"] == [{"claim_id": CLAIM_ID, "evidence_id": EVIDENCE_ID}]
    assert payload["edges"][0]["relation"] == "entails"
    assert payload["edges"][0]["note"] == "derived"


def test_graph_roundtrips_with_identical_content() -> None:
    graph = small_graph()
    restored = graph_from_dict(graph_to_dict(graph))
    assert restored.signature() == graph.signature()
    assert restored.link_pairs() == graph.link_pairs()
    assert restored.edges == graph.edges
    assert restored.validate() is None


def test_edge_dict_roundtrip_keeps_the_note() -> None:
    edge = small_graph().edges[0]
    assert edge_from_dict(edge_to_dict(edge)) == edge
    bare = edge_from_dict(
        {"source": edge.source, "target": edge.target, "relation": "entails"}
    )
    assert bare.note is None


def test_graph_from_dict_rejects_unknown_keys_and_versions() -> None:
    payload = graph_to_dict(small_graph())
    payload["owner"] = "someone"
    with pytest.raises(SchemaError, match="unknown keys"):
        graph_from_dict(payload)
    payload = graph_to_dict(small_graph())
    payload["schema_version"] = "0.9"
    with pytest.raises(SchemaError, match="unsupported schema version"):
        graph_from_dict(payload)


def test_nested_errors_report_their_position() -> None:
    payload = graph_to_dict(small_graph())
    payload["claims"][1]["statement"] = "   "
    with pytest.raises(SchemaError, match=r"claims\[1\]"):
        graph_from_dict(payload)
    payload = graph_to_dict(small_graph())
    payload["links"].append(
        {"claim_id": "clm_999999999999", "evidence_id": EVIDENCE_ID}
    )
    with pytest.raises(UnknownReferenceError):
        graph_from_dict(payload)
    payload = graph_to_dict(small_graph())
    payload["links"][0]["weight"] = 2
    with pytest.raises(SchemaError, match="unknown keys"):
        graph_from_dict(payload)


def test_duplicate_ids_inside_a_payload_are_rejected() -> None:
    payload = graph_to_dict(small_graph())
    payload["claims"].append(payload["claims"][0])
    with pytest.raises(DuplicateIdError) as info:
        graph_from_dict(payload)
    assert info.value.identifier == CLAIM_ID
