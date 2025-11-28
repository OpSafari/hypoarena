"""Corpus statistics and the content signature used as provenance hash."""

from __future__ import annotations

from hypoarena.corpus import Corpus, Document

SHORT = Document(
    "doc_0123456789ab", "Short", "Binding observed.", source="synthetic:v1"
)
LONG = Document(
    "doc_ffffffffffff",
    "Long",
    "Binding observed. Binding reproduced in a second assay with 48 samples.",
    source="synthetic:v1",
)
OTHER_SOURCE = Document(
    "doc_222222222222", "Field note", "No effect seen.", source="notes:v1"
)


def test_stats_summarize_an_empty_corpus() -> None:
    stats = Corpus().stats()
    assert stats.as_dict() == {
        "documents": 0,
        "characters": 0,
        "shortest": 0,
        "longest": 0,
        "mean_length": 0.0,
        "sources": {},
    }


def test_stats_count_documents_characters_and_sources() -> None:
    stats = Corpus([SHORT, LONG, OTHER_SOURCE]).stats()
    assert stats.documents == 3
    assert stats.characters == SHORT.length + LONG.length + OTHER_SOURCE.length
    lengths = sorted([SHORT.length, LONG.length, OTHER_SOURCE.length])
    assert stats.shortest == lengths[0]
    assert stats.longest == lengths[-1]
    # mean length is documented as rounded to three decimals
    assert stats.mean_length == round(stats.characters / 3, 3)
    assert abs(stats.mean_length - stats.characters / 3) < 1e-3
    assert stats.sources == (("notes:v1", 1), ("synthetic:v1", 2))


def test_signature_is_stable_and_insertion_order_independent() -> None:
    forward = Corpus([SHORT, LONG, OTHER_SOURCE])
    backward = Corpus([OTHER_SOURCE, LONG, SHORT])
    assert forward.signature() == backward.signature()
    assert len(forward.signature()) == 16


def test_signature_tracks_text_and_metadata_changes() -> None:
    baseline = Corpus([SHORT]).signature()
    edited = Corpus(
        [Document(SHORT.document_id, SHORT.title, "Binding observed!", "synthetic:v1")]
    )
    retagged = Corpus(
        [Document(SHORT.document_id, SHORT.title, SHORT.text, "synthetic:v2")]
    )
    assert Corpus([SHORT]).signature() == baseline
    assert edited.signature() != baseline
    assert retagged.signature() != baseline


def test_empty_and_populated_corpus_signatures_differ() -> None:
    assert Corpus().signature() != Corpus([SHORT]).signature()
