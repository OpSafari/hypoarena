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


EMPTY_SIGNATURE_VALUE = 0
MAX_HASH = (1 << 61) - 1


def hash_shingle(shingle: str, permutation: int) -> int:
    """Return a deterministic 61-bit hash of one shingle under one permutation."""
    if permutation < 0:
        raise ValidationError("permutation index must be >= 0", permutation=permutation)
    return int(content_hash([permutation, shingle], length=16), 16) % MAX_HASH


def minhash_signature(
    shingle_set: frozenset[str], num_perm: int, *, seed: int = 0
) -> tuple[int, ...]:
    """Return the MinHash signature of a shingle set.

    Slot ``i`` holds the smallest hash of any shingle under permutation ``i``;
    the seed shifts the permutation family so independent runs can be compared.
    An empty set yields a signature of zeros, and two empty signatures compare
    equal — the same convention the exact similarities use.
    """
    if num_perm < 1:
        raise ValidationError("num_perm must be >= 1", num_perm=num_perm)
    if not shingle_set:
        return tuple([EMPTY_SIGNATURE_VALUE] * num_perm)
    return tuple(
        min(hash_shingle(shingle, seed * 4096 + index) for shingle in shingle_set)
        for index in range(num_perm)
    )


def text_signature(
    text: str, num_perm: int, *, n: int = 3, words: bool = False, seed: int = 0
) -> tuple[int, ...]:
    """Return the MinHash signature of a text's shingles."""
    return minhash_signature(shingles(text, n, words=words), num_perm, seed=seed)


def minhash_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    """Return the fraction of agreeing slots, an estimate of Jaccard similarity."""
    if len(left) != len(right):
        raise ValidationError(
            "signatures must have the same length",
            left=len(left),
            right=len(right),
        )
    if not left:
        raise ValidationError("signatures must not be empty")
    agreeing = sum(
        1 for first, second in zip(left, right, strict=True) if first == second
    )
    return agreeing / len(left)


def signature_standard_error(num_perm: int) -> float:
    """Return the standard error of a MinHash estimate: ``1 / sqrt(num_perm)``.

    With 64 permutations the estimate is typically within about 0.125 of the true
    Jaccard index (one standard error); 256 permutations tighten that to 0.0625.
    Callers comparing near a threshold should size ``num_perm`` from this bound
    rather than guessing.
    """
    if num_perm < 1:
        raise ValidationError("num_perm must be >= 1", num_perm=num_perm)
    return 1.0 / math.sqrt(num_perm)


def lsh_rows(num_perm: int, bands: int) -> int:
    """Return the rows per band, requiring ``num_perm`` to divide evenly."""
    if bands < 1:
        raise ValidationError("bands must be >= 1", bands=bands)
    if num_perm < 1:
        raise ValidationError("num_perm must be >= 1", num_perm=num_perm)
    if num_perm % bands:
        raise ValidationError(
            "num_perm must be divisible by bands", num_perm=num_perm, bands=bands
        )
    return num_perm // bands


def lsh_bands(signature: Sequence[int], bands: int) -> tuple[tuple[int, ...], ...]:
    """Split a signature into ``bands`` row tuples used as bucket keys."""
    rows = lsh_rows(len(signature), bands)
    return tuple(
        tuple(signature[index * rows : (index + 1) * rows]) for index in range(bands)
    )


def candidate_pairs(
    signatures: Mapping[str, tuple[int, ...]], bands: int
) -> set[tuple[str, str]]:
    """Return the pairs that share at least one band bucket.

    LSH is a *filter*: every returned pair still has to be verified with the
    configured similarity metric before it may join a cluster. The set contains
    sorted identifier pairs, so downstream iteration is deterministic.
    """
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for identifier, signature in signatures.items():
        for band_index, band in enumerate(lsh_bands(signature, bands)):
            buckets.setdefault((band_index, band), []).append(identifier)
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                pairs.add((first, second))
    return pairs


def candidate_probability(similarity: float, num_perm: int, bands: int) -> float:
    """Return the probability that a pair becomes an LSH candidate.

    For similarity ``s``, ``b`` bands of ``r`` rows the standard S-curve applies:
    ``1 - (1 - s**r)**b``. This is the formula behind the documented
    false-positive/false-negative trade-off — pairs well above the inflection are
    almost always proposed, pairs well below almost never are, and the transition
    is where threshold tuning matters.
    """
    if not 0.0 <= similarity <= 1.0:
        raise ValidationError(
            "similarity must lie within [0, 1]", similarity=similarity
        )
    rows = lsh_rows(num_perm, bands)
    return 1.0 - (1.0 - similarity**rows) ** bands


def lsh_inflection(num_perm: int, bands: int) -> float:
    """Return the similarity at which the S-curve is steepest.

    ``(1 / bands) ** (1 / rows)`` is the usual rule of thumb for the threshold a
    banding configuration approximates; choosing ``bands`` so this value matches
    the intended threshold is how the trade-off is controlled.
    """
    rows = lsh_rows(num_perm, bands)
    return (1.0 / bands) ** (1.0 / rows)
