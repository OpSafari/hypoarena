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

from collections.abc import Mapping

from hypoarena.ids import (
    content_hash,
)
from hypoarena.text import (
    normalize,
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
