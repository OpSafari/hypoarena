"""Span resolution: real citations pass, fabricated ones are named."""

from __future__ import annotations

from hypoarena.corpus import Corpus, Document
from hypoarena.grounding import GroundingIssue, check_span
from hypoarena.schema import Citation

DOC = "doc_0123456789ab"
TEXT = "Protein A increases cell growth in HeLa cells."
START = TEXT.index("Protein A increases cell growth")
QUOTE = TEXT[START : START + len("Protein A increases cell growth")]


def corpus() -> Corpus:
    return Corpus([Document(DOC, "Synthetic finding", TEXT)])


def citation(**overrides: object) -> Citation:
    payload: dict[str, object] = {
        "document_id": DOC,
        "start": START,
        "end": START + len(QUOTE),
        "quote": QUOTE,
    }
    payload.update(overrides)
    return Citation(**payload)  # type: ignore[arg-type]


def test_a_real_span_resolves_without_issues() -> None:
    resolved, issues, detail = check_span(corpus(), citation())
    assert resolved is True
    assert issues == ()
    assert detail is None


def test_a_missing_document_is_reported() -> None:
    resolved, issues, detail = check_span(
        corpus(), citation(document_id="doc_999999999999")
    )
    assert resolved is False
    assert issues == (GroundingIssue.MISSING_DOCUMENT,)
    assert "no document" in (detail or "")


def test_offsets_outside_the_document_are_reported() -> None:
    store = corpus()
    shifted = Citation(DOC, len(TEXT) - 3, len(TEXT) + 20, "x" * 23)
    resolved, issues, detail = check_span(store, shifted)
    assert resolved is False
    assert issues == (GroundingIssue.SPAN_OUT_OF_RANGE,)
    assert "outside" in (detail or "")


def test_a_fabricated_quote_is_reported_with_both_strings() -> None:
    forged = citation(quote="Protein A decreases cell growth")
    resolved, issues, detail = check_span(corpus(), forged)
    assert resolved is False
    assert issues == (GroundingIssue.QUOTE_MISMATCH,)
    assert "Protein A increases cell growth" in (detail or "")


def test_a_shifted_span_with_the_same_quote_is_reported() -> None:
    shifted = citation(start=START + 1, end=START + 1 + len(QUOTE))
    resolved, issues, _ = check_span(corpus(), shifted)
    assert resolved is False
    assert issues == (GroundingIssue.QUOTE_MISMATCH,)
