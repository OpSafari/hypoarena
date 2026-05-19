"""Every configured dedup method must cluster an identical pair reproducibly.

Sweeping all methods over the same input guards against a method silently
diverging from the others on the easiest possible case (two identical strings),
and pins the method vocabulary so a new method is a deliberate addition.
"""

from __future__ import annotations

import pytest

from hypoarena.dedup import DEDUP_METHODS, DedupConfig, DuplicateFinder
from hypoarena.errors import ValidationError

TEXTS = {
    "a": "the cat sat on the mat",
    "b": "the cat sat on the mat",
    "c": "a completely different sentence about distant stars",
}


def test_the_method_vocabulary_is_pinned() -> None:
    assert DEDUP_METHODS == ("exact", "jaccard", "tfidf", "minhash")


@pytest.mark.parametrize("method", DEDUP_METHODS)
def test_every_method_clusters_an_identical_pair(method: str) -> None:
    report = DuplicateFinder(DedupConfig(method=method)).report(TEXTS)
    assert {"a", "b"} <= set(report.duplicate_ids())


@pytest.mark.parametrize("method", DEDUP_METHODS)
def test_no_method_clusters_the_unrelated_text(method: str) -> None:
    report = DuplicateFinder(DedupConfig(method=method)).report(TEXTS)
    assert "c" not in set(report.duplicate_ids())


@pytest.mark.parametrize("method", DEDUP_METHODS)
def test_every_method_is_reproducible(method: str) -> None:
    first = DuplicateFinder(DedupConfig(method=method)).report(TEXTS)
    second = DuplicateFinder(DedupConfig(method=method)).report(TEXTS)
    assert first.duplicate_ids() == second.duplicate_ids()


def test_an_unknown_method_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DedupConfig(method="nope")
