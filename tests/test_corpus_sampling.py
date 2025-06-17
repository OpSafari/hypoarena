"""Seeded span sampling: determinism, boundaries and validation."""

from __future__ import annotations

from random import Random

import pytest

from hypoarena.corpus import Corpus, Document, sample_spans, word_spans
from hypoarena.errors import ValidationError

TEXT = (
    "Protein A binds the promoter of gene B and increases its expression. "
    "A second assay reproduced the effect with 48 samples. "
    "No binding was detected in the control extract."
)
DOCUMENT_ID = "doc_0123456789ab"


def document() -> Document:
    return Document(DOCUMENT_ID, "Binding study", TEXT)


def test_word_spans_cover_every_token() -> None:
    spans = word_spans(document())
    assert spans[0] == (0, 7)
    assert all(TEXT[start:end].strip() == TEXT[start:end] for start, end in spans)
    assert len(spans) == len(TEXT.split())


def test_sampling_is_deterministic_for_a_seed() -> None:
    first = sample_spans(document(), Random(270106), 3)
    second = sample_spans(document(), Random(270106), 3)
    assert first == second
    assert len(first) == 3


def test_different_seeds_produce_different_spans() -> None:
    assert sample_spans(document(), Random(1), 3) != sample_spans(
        document(), Random(2), 3
    )


def test_sampled_spans_resolve_against_the_corpus() -> None:
    corpus = Corpus([document()])
    for citation in sample_spans(document(), Random(7), 5):
        assert corpus.resolve(citation) == citation.quote
        assert len(citation.quote) >= 12


def test_spans_do_not_overlap_and_are_sorted() -> None:
    citations = sample_spans(document(), Random(3), 6)
    offsets = [(item.start, item.end) for item in citations]
    assert offsets == sorted(offsets)
    for index in range(len(offsets) - 1):
        assert offsets[index][1] <= offsets[index + 1][0]


def test_short_documents_yield_at_most_one_span() -> None:
    tiny = Document("doc_222222222222", "Tiny", "Short text here.")
    assert len(sample_spans(tiny, Random(1), 5, min_length=12)) == 1
    stub = Document("doc_333333333333", "Stub", "abc")
    assert sample_spans(stub, Random(1), 5, min_length=12) == []


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValidationError, match="count"):
        sample_spans(document(), Random(1), -1)
    with pytest.raises(ValidationError, match="minimum span length"):
        sample_spans(document(), Random(1), 1, min_length=0)
    with pytest.raises(ValidationError, match="maximum span length"):
        sample_spans(document(), Random(1), 1, min_length=40, max_length=10)


def test_zero_count_returns_nothing() -> None:
    assert sample_spans(document(), Random(1), 0) == []
