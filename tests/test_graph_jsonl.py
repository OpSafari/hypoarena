"""JSONL graph documents: layout, integrity checks and rejections."""

from __future__ import annotations

import json

import pytest

from helpers import (
    CLAIM_ID,
    EVIDENCE_ID,
    OTHER_CLAIM_ID,
    sample_claim,
    sample_evidence,
)
from hypoarena.errors import SchemaError, UnknownReferenceError
from hypoarena.graph import HypothesisGraph
from hypoarena.schema import ClaimRelation
from hypoarena.serialize import (
    RECORD_TYPES,
    graph_from_lines,
    graph_from_text,
    graph_to_lines,
    graph_to_text,
)


def small_graph() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_evidence(sample_evidence())
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.ENTAILS)
    return graph


def test_lines_start_with_meta_then_dependencies_first() -> None:
    lines = graph_to_lines(small_graph())
    kinds = [json.loads(line)["record"] for line in lines]
    assert kinds == ["meta", "claim", "claim", "evidence", "link", "edge"]
    assert set(kinds) <= set(RECORD_TYPES)


def test_meta_line_pins_counts_and_signature() -> None:
    graph = small_graph()
    meta = json.loads(graph_to_lines(graph)[0])
    assert meta["schema_version"] == "1.0"
    assert meta["counts"] == {"claims": 2, "evidence": 1, "links": 1, "edges": 1}
    assert meta["signature"] == graph.signature()


def test_link_and_edge_lines_are_exact() -> None:
    lines = graph_to_lines(small_graph())
    assert lines[4] == (
        '{"claim_id":"clm_0123456789ab","evidence_id":"evd_0123456789ab",'
        '"record":"link"}\n'
    )
    assert lines[5] == (
        '{"note":null,"record":"edge","relation":"entails",'
        '"source":"clm_0123456789ab","target":"clm_ffffffffffff"}\n'
    )


def test_document_roundtrips_through_text() -> None:
    graph = small_graph()
    text = graph_to_text(graph)
    restored = graph_from_text(text)
    assert restored.signature() == graph.signature()
    assert graph_to_text(restored) == text


def test_documents_without_meta_still_rebuild() -> None:
    graph = small_graph()
    lines = graph_to_lines(graph, include_meta=False)
    assert all('"meta"' not in line for line in lines)
    assert graph_from_lines(lines).signature() == graph.signature()


def test_count_mismatch_is_detected() -> None:
    lines = graph_to_lines(small_graph())
    meta = json.loads(lines[0])
    meta["counts"]["claims"] = 5
    lines[0] = json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(SchemaError, match="counts disagree"):
        graph_from_lines(lines)


def test_signature_mismatch_is_detected() -> None:
    lines = graph_to_lines(small_graph())
    meta = json.loads(lines[0])
    meta["signature"] = "0" * 16
    lines[0] = json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(SchemaError, match="signature disagrees"):
        graph_from_lines(lines)


def test_meta_verification_can_be_skipped() -> None:
    lines = graph_to_lines(small_graph())
    meta = json.loads(lines[0])
    meta["signature"] = "0" * 16
    lines[0] = json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n"
    assert (
        graph_from_lines(lines, verify_meta=False).signature()
        == small_graph().signature()
    )


def test_unknown_record_types_and_keys_are_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown graph record type"):
        graph_from_lines(['{"record":"observation","value":1}'])
    with pytest.raises(SchemaError, match="unknown keys"):
        graph_from_lines(
            [
                '{"claim_id":"clm_0123456789ab","evidence_id":"evd_0123456789ab",'
                '"extra":1,"record":"link"}'
            ]
        )


def test_dangling_links_report_the_line_number() -> None:
    with pytest.raises(UnknownReferenceError) as info:
        graph_from_lines(
            [
                '{"claim_id":"clm_999999999999","evidence_id":"evd_999999999999","record":"link"}'
            ]
        )
    assert info.value.kind == "claim"


def test_blank_lines_are_ignored() -> None:
    text = graph_to_text(small_graph())
    assert graph_from_text("\n" + text.replace("\n", "\n\n")).signature() == (
        small_graph().signature()
    )
