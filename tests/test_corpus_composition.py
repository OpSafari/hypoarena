"""Subcorpora, merging and attribute filtering."""

from __future__ import annotations

import pytest

from hypoarena.corpus import Corpus, Document
from hypoarena.errors import CorpusError, UnknownReferenceError, ValidationError

FIRST = Document(
    "doc_0123456789ab",
    "Binding study",
    "Protein A binds gene B.",
    "synthetic:v1",
    (("chain", "c1"),),
)
SECOND = Document(
    "doc_ffffffffffff", "Replication", "Binding reproduced.", "synthetic:v1"
)
CONFLICTING = Document(
    "doc_0123456789ab",
    "Binding study",
    "Protein A does NOT bind gene B.",
    "synthetic:v1",
)


def test_subcorpus_selects_and_orders_documents() -> None:
    corpus = Corpus([FIRST, SECOND])
    sliced = corpus.subcorpus([SECOND.document_id, FIRST.document_id])
    assert sliced.document_ids == (FIRST.document_id, SECOND.document_id)
    assert len(corpus) == 2


def test_subcorpus_rejects_unknown_identifiers() -> None:
    with pytest.raises(UnknownReferenceError):
        Corpus([FIRST]).subcorpus(["doc_999999999999"])


def test_merge_unions_disjoint_corpora() -> None:
    merged = Corpus([FIRST]).merged(Corpus([SECOND]))
    assert merged.document_ids == (FIRST.document_id, SECOND.document_id)


def test_merge_shares_identical_documents() -> None:
    merged = Corpus([FIRST]).merged(Corpus([FIRST, SECOND]))
    assert len(merged) == 2


def test_merge_conflicts_raise_by_default() -> None:
    with pytest.raises(CorpusError, match="conflicting document content"):
        Corpus([FIRST]).merged(Corpus([CONFLICTING]))


def test_merge_conflicts_can_prefer_the_receiver() -> None:
    merged = Corpus([FIRST]).merged(Corpus([CONFLICTING]), on_conflict="prefer_self")
    assert merged.document(FIRST.document_id).text == FIRST.text


def test_merge_rejects_unknown_conflict_policies() -> None:
    with pytest.raises(ValidationError, match="unknown conflict policy"):
        Corpus([FIRST]).merged(Corpus([SECOND]), on_conflict="prefer_other")


def test_documents_for_attribute_filters_by_metadata() -> None:
    corpus = Corpus([FIRST, SECOND])
    assert [
        item.document_id for item in corpus.documents_for_attribute("chain", "c1")
    ] == [FIRST.document_id]
    assert corpus.documents_for_attribute("chain", "c2") == ()
