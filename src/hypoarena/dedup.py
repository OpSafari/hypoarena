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

from collections.abc import Mapping, Sequence

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
