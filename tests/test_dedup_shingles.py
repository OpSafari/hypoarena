"""Shingling and Jaccard similarity: symmetry, bounds and boundaries."""

from __future__ import annotations

import pytest

from hypoarena.dedup import jaccard_similarity, shingles, similarity_matrix
from hypoarena.errors import ValidationError

FIRST = "Protein A increases cell growth in HeLa cells"
PARAPHRASE = "In HeLa cells, protein A increases the growth of cells"
UNRELATED = "Batch effects were regressed out before summarizing results"


def test_character_and_word_shingles() -> None:
    assert shingles("abcd", 2) == {"ab", "bc", "cd"}
    assert shingles("a b", 5) == {"ab"}
    assert shingles("protein A binds", 2, words=True) == {"protein a", "a binds"}


def test_shingle_sizes_are_validated() -> None:
    with pytest.raises(ValidationError, match="shingle size"):
        shingles("abc", 0)


def test_identical_texts_score_one() -> None:
    assert jaccard_similarity(FIRST, FIRST) == 1.0
    assert jaccard_similarity(FIRST, FIRST.lower()) == 1.0


def test_paraphrases_score_higher_than_unrelated_text() -> None:
    paraphrase = jaccard_similarity(FIRST, PARAPHRASE)
    unrelated = jaccard_similarity(FIRST, UNRELATED)
    assert paraphrase > unrelated
    assert 0.0 < paraphrase < 1.0


def test_similarity_is_symmetric_and_bounded() -> None:
    assert jaccard_similarity(FIRST, PARAPHRASE) == jaccard_similarity(
        PARAPHRASE, FIRST
    )
    for pair in ((FIRST, PARAPHRASE), (FIRST, UNRELATED), ("", "")):
        assert 0.0 <= jaccard_similarity(*pair) <= 1.0


def test_empty_texts_follow_the_documented_rule() -> None:
    assert jaccard_similarity("", "") == 1.0
    assert jaccard_similarity("", FIRST) == 0.0
    assert jaccard_similarity(FIRST, "!!!") == 0.0


def test_word_shingles_are_less_sensitive_to_word_order() -> None:
    reordered = "cell growth in HeLa cells increases Protein A"
    character = jaccard_similarity(FIRST, reordered, n=3)
    words = jaccard_similarity(FIRST, reordered, n=1, words=True)
    assert words > character


def test_the_matrix_is_symmetric_with_a_unit_diagonal() -> None:
    texts = [FIRST, PARAPHRASE, UNRELATED]
    matrix = similarity_matrix(texts)
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert all(matrix[index][index] == 1.0 for index in range(3))
    assert matrix[0][1] == matrix[1][0]
    assert matrix[0][1] > matrix[0][2]
