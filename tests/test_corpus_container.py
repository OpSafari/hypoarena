"""Corpus construction, lookup, ordering and duplicate rejection."""

from __future__ import annotations

import pytest

from hypoarena.corpus import Corpus, Document
from hypoarena.errors import DuplicateIdError, UnknownReferenceError, ValidationError

FIRST = "doc_0123456789ab"
SECOND = "doc_ffffffffffff"


def first() -> Document:
    return Document(FIRST, "First paper", "Protein A binds gene B promoter.")


def second() -> Document:
    return Document(SECOND, "Second paper", "Binding was reproduced in vivo.")


def test_corpus_accepts_documents_at_construction() -> None:
    corpus = Corpus([first(), second()])
    assert len(corpus) == 2
    assert corpus.document_ids == (FIRST, SECOND)
    assert [document.title for document in corpus] == ["First paper", "Second paper"]


def test_duplicate_documents_are_rejected() -> None:
    corpus = Corpus([first()])
    with pytest.raises(DuplicateIdError) as info:
        corpus.add_document(first())
    assert info.value.kind == "document"
    assert len(corpus) == 1


def test_non_documents_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Document objects"):
        Corpus().add_document("doc_0123456789ab")  # type: ignore[arg-type]


def test_lookup_reports_missing_identifiers() -> None:
    corpus = Corpus([first()])
    assert corpus.has_document(FIRST) is True
    assert corpus.has_document(SECOND) is False
    with pytest.raises(UnknownReferenceError) as info:
        corpus.document(SECOND)
    assert info.value.kind == "document"
    assert SECOND not in corpus
    assert 7 not in corpus


def test_text_of_returns_the_full_document_text() -> None:
    assert Corpus([first()]).text_of(FIRST) == "Protein A binds gene B promoter."
