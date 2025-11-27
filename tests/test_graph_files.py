"""Graph and JSONL file I/O: atomicity, roundtrips and missing-file handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import CLAIM_ID, EVIDENCE_ID, OTHER_CLAIM_ID, sample_claim, sample_evidence
from hypoarena.errors import ArtifactError, SchemaError
from hypoarena.graph import HypothesisGraph
from hypoarena.schema import ClaimRelation
from hypoarena.serialize import (
    graph_to_text,
    read_graph,
    read_jsonl_records,
    write_graph,
    write_jsonl_records,
)


def small_graph() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add_claim(sample_claim())
    graph.add_claim(sample_claim(claim_id=OTHER_CLAIM_ID))
    graph.add_evidence(sample_evidence())
    graph.link_evidence(CLAIM_ID, EVIDENCE_ID)
    graph.add_edge(CLAIM_ID, OTHER_CLAIM_ID, ClaimRelation.REFINES)
    return graph


def test_write_then_read_restores_the_same_graph(tmp_path: Path) -> None:
    graph = small_graph()
    target = tmp_path / "graph.jsonl"
    assert write_graph(graph, target) == 6
    assert read_graph(target).signature() == graph.signature()


def test_written_bytes_match_the_in_memory_document(tmp_path: Path) -> None:
    graph = small_graph()
    target = tmp_path / "graph.jsonl"
    write_graph(graph, target)
    assert target.read_text(encoding="utf-8") == graph_to_text(graph)


def test_no_partial_file_is_left_behind(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "graph.jsonl"
    write_graph(small_graph(), target)
    assert list(tmp_path.rglob("*.partial")) == []


def test_reading_a_missing_document_raises_an_artifact_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError) as info:
        read_graph(tmp_path / "absent.jsonl")
    assert "absent.jsonl" in str(info.value)


def test_truncated_documents_are_reported(tmp_path: Path) -> None:
    target = tmp_path / "graph.jsonl"
    write_graph(small_graph(), target)
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    target.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(SchemaError, match="counts disagree"):
        read_graph(target)


def test_generic_jsonl_helpers_roundtrip_records(tmp_path: Path) -> None:
    target = tmp_path / "records.jsonl"
    records = [{"stage": "generate", "count": 2}, {"stage": "verify", "count": 1}]
    assert write_jsonl_records(records, target) == 2
    assert list(read_jsonl_records(target)) == records


def test_generic_jsonl_reader_rejects_malformed_lines(tmp_path: Path) -> None:
    target = tmp_path / "bad.jsonl"
    target.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    reader = read_jsonl_records(target)
    assert next(reader) == {"a": 1}
    with pytest.raises(SchemaError) as info:
        next(reader)
    assert info.value.details["line_number"] == 2


def test_generic_jsonl_reader_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        next(read_jsonl_records(tmp_path / "absent.jsonl"))
