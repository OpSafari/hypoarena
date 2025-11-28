"""Citation resolution: real spans pass, fabricated ones fail loudly."""

from __future__ import annotations

import pytest

from hypoarena.corpus import Corpus, Document
from hypoarena.errors import SpanNotFoundError, UnknownReferenceError
from hypoarena.schema import Citation

DOC = "doc_0123456789ab"
OTHER = "doc_ffffffffffff"
TEXT = "Protein A binds the promoter of gene B. Binding increases expression."


def corpus() -> Corpus:
    return Corpus(
        [
            Document(DOC, "Binding study", TEXT),
            Document(OTHER, "Replication", "Binding increases expression in vivo."),
        ]
    )


BINDING = TEXT.index("Binding")


def citation(start: int, end: int, quote: str, document_id: str = DOC) -> Citation:
    return Citation(document_id, start, end, quote)


def test_resolve_returns_the_exact_quoted_text() -> None:
    assert corpus().resolve(citation(0, 9, "Protein A")) == "Protein A"
    assert corpus().resolve(citation(BINDING, BINDING + 7, "Binding")) == "Binding"


def test_resolve_rejects_fabricated_quotes() -> None:
    with pytest.raises(SpanNotFoundError) as info:
        corpus().resolve(citation(0, 9, "Protein Z"))
    details = info.value.details
    assert details["quoted"] == "Protein Z"
    assert details["found"] == "Protein A"


def test_resolve_rejects_offsets_outside_the_document() -> None:
    with pytest.raises(SpanNotFoundError, match="outside the document"):
        corpus().resolve(Citation(DOC, len(TEXT) - 2, len(TEXT) + 10, "x" * 12))


def test_resolve_reports_unknown_documents() -> None:
    with pytest.raises(UnknownReferenceError):
        corpus().resolve(citation(0, 5, "Prote", document_id="doc_999999999999"))


def test_verify_quote_accepts_real_spans_and_rejects_drifted_ones() -> None:
    store = corpus()
    assert store.verify_quote(citation(0, 9, "Protein A")) is True
    assert store.verify_quote(citation(0, 9, "Protein B")) is False
    assert store.verify_quote(citation(0, 9, "Protein A", "doc_999999999999")) is False


def test_find_quote_returns_the_first_span_only() -> None:
    assert corpus().find_quote(DOC, "Binding") == (BINDING, BINDING + 7)
    assert corpus().find_quote(DOC, "absent") is None


def test_search_quote_is_deterministic_and_corpus_wide() -> None:
    results = corpus().search_quote("Binding increases expression")
    assert [(item.document_id, item.start) for item in results] == [
        (DOC, BINDING),
        (OTHER, 0),
    ]
    assert corpus().search_quote("nothing matches") == []
