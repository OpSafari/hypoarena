"""Character vs word shingles: configuration and the behaviour they imply."""

from __future__ import annotations

import pytest

from hypoarena.dedup import (
    SHINGLE_UNITS,
    DedupConfig,
    DuplicateFinder,
    NoveltyGuard,
)
from hypoarena.errors import ValidationError

LEFT = "protein A increases cell growth"
RIGHT = "cell growth increases protein A"
OTHER = "batch effects were regressed out before summarizing"


def test_units_are_golden_and_default_to_characters() -> None:
    assert SHINGLE_UNITS == ("char", "word")
    assert DedupConfig().shingle_unit == "char"


def test_unknown_units_are_rejected_and_tracked_by_the_fingerprint() -> None:
    with pytest.raises(ValidationError, match="shingle_unit"):
        DedupConfig(shingle_unit="sentence")
    assert DedupConfig(shingle_unit="word").fingerprint() != DedupConfig().fingerprint()


def test_word_shingles_tolerate_reordering_that_characters_do_not() -> None:
    texts = {"a": LEFT, "b": RIGHT, "c": OTHER}
    words = DuplicateFinder(
        DedupConfig(method="jaccard", shingle_unit="word", ngram_size=1, threshold=0.9)
    ).find(texts)
    characters = DuplicateFinder(
        DedupConfig(method="jaccard", shingle_unit="char", threshold=0.9)
    ).find(texts)
    assert [cluster.members for cluster in words] == [("a", "b")]
    assert characters == ()


def test_word_units_apply_to_minhash_signatures_too() -> None:
    texts = {"a": LEFT, "b": RIGHT}
    clustered = DuplicateFinder(
        DedupConfig(
            method="minhash",
            shingle_unit="word",
            ngram_size=1,
            threshold=0.9,
            use_lsh=False,
        )
    ).find(texts)
    assert [cluster.members for cluster in clustered] == [("a", "b")]


def test_word_bigrams_are_stricter_than_unigrams() -> None:
    texts = {"a": LEFT, "b": RIGHT}
    unigrams = DuplicateFinder(
        DedupConfig(method="jaccard", shingle_unit="word", ngram_size=1, threshold=0.9)
    ).find(texts)
    bigrams = DuplicateFinder(
        DedupConfig(method="jaccard", shingle_unit="word", ngram_size=2, threshold=0.5)
    ).find(texts)
    assert unigrams and not bigrams


def test_the_novelty_guard_honours_the_unit() -> None:
    guard = NoveltyGuard(
        DedupConfig(method="jaccard", shingle_unit="word", ngram_size=1, threshold=0.9)
    )
    guard.observe("a", LEFT)
    assert guard.check(RIGHT).is_novel is False
    assert guard.check(OTHER).is_novel is True
