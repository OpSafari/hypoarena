"""Corpus JSONL documents: layout, integrity and rejection paths."""

from __future__ import annotations

import json

import pytest

from hypoarena.corpus import Corpus, Document
from hypoarena.errors import DuplicateIdError, SchemaError
from hypoarena.serialize import (
    corpus_from_lines,
    corpus_from_text,
    corpus_to_lines,
    corpus_to_text,
)


def sample_corpus() -> Corpus:
    return Corpus(
        [
            Document(
                "doc_0123456789ab",
                "Binding study",
                "Protein A binds the promoter of gene B.",
                "synthetic:v1",
                (("chain", "c1"),),
            ),
            Document("doc_ffffffffffff", "Replication", "Binding reproduced in vivo."),
        ]
    )


def test_header_comes_first_then_documents_in_order() -> None:
    lines = corpus_to_lines(sample_corpus())
    kinds = [json.loads(line)["record"] for line in lines]
    assert kinds == ["meta", "document", "document"]
    assert json.loads(lines[0])["counts"] == {"documents": 2, "characters": 66}


def test_document_lines_are_exact() -> None:
    lines = corpus_to_lines(sample_corpus())
    assert lines[2] == (
        '{"document":{"attributes":[],"document_id":"doc_ffffffffffff","source":'
        '"unspecified","text":"Binding reproduced in vivo.","title":"Replication"},'
        '"record":"document"}\n'
    )


def test_document_roundtrips_through_text() -> None:
    corpus = sample_corpus()
    text = corpus_to_text(corpus)
    assert corpus_to_text(corpus_from_text(text)) == text
    assert corpus_from_text(text).signature() == corpus.signature()


def test_documents_without_a_header_still_rebuild() -> None:
    corpus = sample_corpus()
    lines = corpus_to_lines(corpus, include_meta=False)
    assert corpus_from_lines(lines).signature() == corpus.signature()


def test_truncated_documents_are_detected() -> None:
    lines = corpus_to_lines(sample_corpus())
    with pytest.raises(SchemaError, match="counts disagree"):
        corpus_from_lines(lines[:-1])


def test_unknown_record_types_are_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown corpus record type"):
        corpus_from_lines(['{"record":"annotation","text":"x"}'])


def test_duplicate_documents_are_rejected() -> None:
    lines = corpus_to_lines(sample_corpus())
    with pytest.raises(DuplicateIdError):
        corpus_from_lines([lines[0], lines[1], lines[1]])
