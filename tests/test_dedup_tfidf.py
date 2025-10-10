"""TF-IDF vectors and cosine similarity."""

from __future__ import annotations

import numpy as np
import pytest

from hypoarena.dedup import (
    TfidfVectorizer,
    cosine_matrix,
    cosine_similarity,
    term_counts,
)
from hypoarena.errors import ValidationError

FIRST = "Protein A increases cell growth in HeLa cells"
PARAPHRASE = "In HeLa cells, protein A increases the growth of cells"
UNRELATED = "Batch effects were regressed out before summarizing results"
DOCUMENTS = [FIRST, PARAPHRASE, UNRELATED]


def test_term_counts_handle_unigrams_and_bigrams() -> None:
    assert term_counts("a b a", 1) == {"a": 2, "b": 1}
    assert term_counts("a b c", 2) == {"a b": 1, "b c": 1}
    with pytest.raises(ValidationError, match="ngram_size"):
        term_counts("a", 0)


def test_vocabulary_is_sorted_and_filtered_by_document_frequency() -> None:
    vectorizer = TfidfVectorizer(min_document_frequency=2).fit(DOCUMENTS)
    assert vectorizer.vocabulary_ == tuple(sorted(vectorizer.vocabulary_))
    assert "cells" in vectorizer.vocabulary_
    assert "batch" not in vectorizer.vocabulary_
    assert len(TfidfVectorizer().fit(DOCUMENTS).vocabulary_) > len(
        vectorizer.vocabulary_
    )


def test_rows_are_unit_normalized() -> None:
    vectors = TfidfVectorizer().fit_transform(DOCUMENTS)
    assert vectors.shape == (3, len(TfidfVectorizer().fit(DOCUMENTS).vocabulary_))
    for row in vectors:
        assert float(np.linalg.norm(row)) == pytest.approx(1.0)


def test_identical_documents_score_one_and_disjoint_score_zero() -> None:
    vectors = TfidfVectorizer().fit_transform([FIRST, FIRST, UNRELATED])
    assert cosine_similarity(vectors[0], vectors[1]) == pytest.approx(1.0)
    assert cosine_similarity(vectors[0], vectors[2]) == pytest.approx(0.0, abs=1e-9)


def test_paraphrases_score_above_unrelated_pairs() -> None:
    vectors = TfidfVectorizer().fit_transform(DOCUMENTS)
    assert cosine_similarity(vectors[0], vectors[1]) > cosine_similarity(
        vectors[0], vectors[2]
    )


def test_bigram_vectors_separate_word_order() -> None:
    unigrams = TfidfVectorizer(ngram_size=1).fit_transform([FIRST, PARAPHRASE])
    bigrams = TfidfVectorizer(ngram_size=2).fit_transform([FIRST, PARAPHRASE])
    assert cosine_similarity(unigrams[0], unigrams[1]) > cosine_similarity(
        bigrams[0], bigrams[1]
    )


def test_the_matrix_is_symmetric_with_a_unit_diagonal() -> None:
    vectors = TfidfVectorizer().fit_transform(DOCUMENTS)
    matrix = cosine_matrix(vectors)
    assert matrix.shape == (3, 3)
    for index in range(3):
        assert float(matrix[index, index]) == pytest.approx(1.0)
        for other in range(3):
            assert float(matrix[index, other]) == pytest.approx(
                float(matrix[other, index])
            )


def test_empty_documents_follow_the_zero_vector_convention() -> None:
    vectors = TfidfVectorizer().fit_transform(["", "", FIRST])
    assert cosine_similarity(vectors[0], vectors[1]) == 1.0
    assert cosine_similarity(vectors[0], vectors[2]) == 0.0


def test_transform_before_fit_and_bad_shapes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="fitted"):
        TfidfVectorizer().transform(DOCUMENTS)
    with pytest.raises(ValidationError, match="ngram_size"):
        TfidfVectorizer(ngram_size=0)
    with pytest.raises(ValidationError, match="min_document_frequency"):
        TfidfVectorizer(min_document_frequency=0)
    with pytest.raises(ValidationError, match="equally shaped"):
        cosine_similarity(np.zeros(3), np.zeros(4))


def test_idf_decreases_with_document_frequency() -> None:
    vectorizer = TfidfVectorizer().fit(["alpha beta", "alpha gamma", "alpha delta"])
    positions = {term: index for index, term in enumerate(vectorizer.vocabulary_)}
    assert vectorizer.idf_[positions["alpha"]] < vectorizer.idf_[positions["beta"]]
