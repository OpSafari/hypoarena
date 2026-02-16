"""Duplicate and near-duplicate detection for hypotheses and findings.

Four levels are provided, cheapest first: exact matching on normalized text,
character/word n-gram Jaccard, TF-IDF cosine, and MinHash with LSH banding for
large candidate sets. Every function is deterministic and pure; the LSH stage
only *proposes* candidate pairs, and a cluster is never formed without verifying
the pair with the configured similarity metric.

Documented behaviour rather than promises: MinHash estimates Jaccard similarity
with a standard error of about ``1 / sqrt(num_perm)``, and LSH banding trades
false negatives (dissimilar pairs proposed) for false positives (similar pairs
missed) around the ``threshold ≈ (1 / bands) ** (1 / rows)`` inflection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
)
from hypoarena.text import (
    char_ngrams,
    normalize,
    tokenize,
    word_ngrams,
)


def normalized_signature(text: str) -> str:
    """Return the digest of a normalized text.

    Two texts that differ only by case, punctuation or whitespace share a
    signature, which is what makes exact duplicate detection robust to the
    surface variation a paraphrase cluster introduces.
    """
    return content_hash(normalize(text))


def exact_duplicates(texts: Mapping[str, str]) -> tuple[tuple[str, ...], ...]:
    """Group identifiers whose normalized texts are identical.

    Groups with a single member are omitted, and both the members and the groups
    are sorted so the result never depends on dictionary order.
    """
    groups: dict[str, list[str]] = {}
    for identifier, text in texts.items():
        groups.setdefault(normalized_signature(text), []).append(identifier)
    return tuple(
        sorted(tuple(sorted(group)) for group in groups.values() if len(group) > 1)
    )


def duplicate_rate(texts: Mapping[str, str]) -> float:
    """Return the fraction of items that share text with another item."""
    if not texts:
        return 0.0
    duplicated = sum(len(group) for group in exact_duplicates(texts))
    return duplicated / len(texts)


def shingles(text: str, n: int, *, words: bool = False) -> frozenset[str]:
    """Return the character or word n-gram set of a normalized text.

    Character shingles are the default because they survive word-order changes
    better than single tokens; word shingles (``words=True``) are what the TF-IDF
    stage uses. Short texts fall back to the whole normalized string so that very
    short inputs still compare equal to themselves.
    """
    if n < 1:
        raise ValidationError("shingle size must be >= 1", n=n)
    if words:
        return frozenset(word_ngrams(tokenize(text), n))
    return frozenset(char_ngrams(text, n))


def jaccard_similarity(
    left: str, right: str, *, n: int = 3, words: bool = False
) -> float:
    """Return the Jaccard index of two texts' shingle sets.

    Two empty texts score ``1.0`` (they are the same nothing); an empty text
    against a non-empty one scores ``0.0``.
    """
    first = shingles(left, n, words=words)
    second = shingles(right, n, words=words)
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def similarity_matrix(
    texts: Sequence[str], *, n: int = 3, words: bool = False
) -> list[list[float]]:
    """Return the pairwise Jaccard matrix of a list of texts."""
    sets = [shingles(text, n, words=words) for text in texts]
    matrix: list[list[float]] = []
    for first in sets:
        row = []
        for second in sets:
            if not first and not second:
                row.append(1.0)
            elif not first or not second:
                row.append(0.0)
            else:
                row.append(len(first & second) / len(first | second))
        matrix.append(row)
    return matrix


def term_counts(text: str, ngram_size: int) -> dict[str, int]:
    """Count word n-grams (or single tokens when ``ngram_size`` is 1)."""
    if ngram_size < 1:
        raise ValidationError("ngram_size must be >= 1", ngram_size=ngram_size)
    terms = (
        tokenize(text) if ngram_size == 1 else word_ngrams(tokenize(text), ngram_size)
    )
    counts: dict[str, int] = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1
    return counts


class TfidfVectorizer:
    """Term frequency / inverse document frequency vectors over word n-grams.

    ``idf`` uses the smoothed form ``log((1 + n_docs) / (1 + df)) + 1`` so a term
    present in every document still contributes (with weight 1) instead of
    vanishing. Rows are L2-normalized, which turns a dot product into a cosine
    similarity. ``min_document_frequency`` drops rare terms, the usual defence
    against identifiers and typos dominating the vocabulary.
    """

    def __init__(self, ngram_size: int = 1, min_document_frequency: int = 1) -> None:
        if ngram_size < 1:
            raise ValidationError("ngram_size must be >= 1", ngram_size=ngram_size)
        if min_document_frequency < 1:
            raise ValidationError(
                "min_document_frequency must be >= 1",
                min_document_frequency=min_document_frequency,
            )
        self.ngram_size = ngram_size
        self.min_document_frequency = min_document_frequency
        self.vocabulary_: tuple[str, ...] = ()
        self.idf_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.fitted_ = False

    def fit(self, documents: Sequence[str]) -> TfidfVectorizer:
        """Learn the vocabulary and inverse document frequencies."""
        frequencies: dict[str, int] = {}
        for document in documents:
            for term in set(term_counts(document, self.ngram_size)):
                frequencies[term] = frequencies.get(term, 0) + 1
        kept = sorted(
            term
            for term, count in frequencies.items()
            if count >= self.min_document_frequency
        )
        total = len(documents)
        self.vocabulary_ = tuple(kept)
        self.idf_ = np.array(
            [
                math.log((1 + total) / (1 + frequencies[term])) + 1.0
                for term in self.vocabulary_
            ],
            dtype=np.float64,
        )
        self.fitted_ = True
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Return L2-normalized TF-IDF rows for ``documents``."""
        if not self.fitted_:
            raise ValidationError("vectorizer must be fitted before transforming")
        index = {term: position for position, term in enumerate(self.vocabulary_)}
        matrix = np.zeros((len(documents), len(self.vocabulary_)), dtype=np.float64)
        for row, document in enumerate(documents):
            for term, count in term_counts(document, self.ngram_size).items():
                position = index.get(term)
                if position is not None:
                    matrix[row, position] = count * self.idf_[position]
        return l2_normalize(matrix)

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        """Fit on ``documents`` and return their vectors."""
        return self.fit(documents).transform(documents)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving zero rows untouched."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return the cosine of two equally shaped vectors.

    Two zero vectors score ``1.0`` and a zero vector against a non-zero one
    scores ``0.0``, matching the convention used by :func:`jaccard_similarity`.
    """
    if left.shape != right.shape:
        raise ValidationError(
            "cosine needs equally shaped vectors",
            left=list(left.shape),
            right=list(right.shape),
        )
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine matrix of L2-normalized rows."""
    normalized = l2_normalize(vectors)
    return normalized @ normalized.T
