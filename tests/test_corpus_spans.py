"""Locating quotes inside documents, in both directions."""

from __future__ import annotations

from hypoarena.corpus import Document
from hypoarena.schema import Citation

TEXT = "Binding was observed. Binding was reproduced in a second assay."
DOCUMENT_ID = "doc_0123456789ab"


def document() -> Document:
    return Document(DOCUMENT_ID, "Repeated findings", TEXT)


def citation(start: int, end: int, quote: str) -> Citation:
    return Citation(DOCUMENT_ID, start, end, quote)


def test_find_all_returns_every_occurrence_in_order() -> None:
    assert document().find_all("Binding") == [0, 22]
    assert document().find_all("absent phrase") == []
    assert document().find_all("") == []


def test_locate_returns_the_first_half_open_span() -> None:
    start = TEXT.index("second assay")
    assert document().locate("Binding") == (0, 7)
    assert document().locate("second assay") == (start, start + len("second assay"))
    assert TEXT[start : start + len("second assay")] == "second assay"
    assert document().locate("nope") is None


def test_citation_spans_lists_all_matches() -> None:
    assert document().citation_spans("Binding") == [(0, 7), (22, 29)]


def test_contains_span_accepts_exact_matches_only() -> None:
    assert document().contains_span(citation(0, 7, "Binding")) is True
    assert document().contains_span(citation(0, 7, "binding")) is False
    assert document().contains_span(citation(0, 8, "Binding")) is False


def test_contains_span_rejects_foreign_documents_and_bad_offsets() -> None:
    assert document().contains_span(citation(0, 7, "Binding")) is True
    foreign = Citation("doc_ffffffffffff", 0, 7, "Binding")
    assert document().contains_span(foreign) is False
    assert document().contains_span(citation(0, len(TEXT) + 5, "Binding")) is False
