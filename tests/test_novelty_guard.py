"""Novelty guard: admission, nearest-neighbour reporting and duplicates."""

from __future__ import annotations

import pytest

from hypoarena.dedup import DedupConfig, NoveltyGuard
from hypoarena.errors import DuplicateIdError

FIRST = "Protein A increases cell growth in HeLa cells"
NEAR = "Protein A increases cell growth in HeLa cells."
DIFFERENT = "Batch effects were regressed out before summarizing results"


def guard(**overrides: object) -> NoveltyGuard:
    config = DedupConfig(method="exact", **overrides)  # type: ignore[arg-type]
    return NoveltyGuard(config)


def test_an_empty_guard_admits_everything() -> None:
    produced = guard()
    verdict = produced.admit("a", FIRST)
    assert verdict.is_novel is True
    assert verdict.nearest is None
    assert produced.size == 1


def test_exact_repetition_is_not_novel() -> None:
    produced = guard()
    produced.admit("a", FIRST)
    verdict = produced.admit("b", NEAR)
    assert verdict.is_novel is False
    assert verdict.nearest == "a"
    assert verdict.similarity == 1.0
    assert produced.size == 1


def test_unrelated_text_is_novel() -> None:
    produced = guard()
    produced.admit("a", FIRST)
    verdict = produced.admit("b", DIFFERENT)
    assert verdict.is_novel is True
    assert verdict.nearest == "a"
    assert verdict.similarity == 0.0
    assert produced.size == 2


def test_jaccard_guard_uses_the_threshold_in_both_directions() -> None:
    loose = NoveltyGuard(DedupConfig(method="jaccard", threshold=0.95))
    loose.admit("a", FIRST)
    assert loose.admit("b", NEAR).is_novel is False
    strict = NoveltyGuard(DedupConfig(method="jaccard", threshold=1.0))
    strict.admit("a", FIRST)
    assert strict.admit(
        "b", "Protein A increases cell growth in HeLa cell lines"
    ).is_novel


def test_the_nearest_neighbour_is_reported() -> None:
    produced = NoveltyGuard(DedupConfig(method="jaccard", threshold=0.99))
    produced.observe("far", DIFFERENT)
    produced.observe("near", FIRST)
    verdict = produced.check("Protein A increases cell growth in HeLa cells!")
    assert verdict.nearest == "near"
    assert verdict.similarity > 0.5
    assert verdict.method == "jaccard"


def test_observe_rejects_duplicate_identifiers() -> None:
    produced = guard()
    produced.observe("a", FIRST)
    with pytest.raises(DuplicateIdError):
        produced.observe("a", DIFFERENT)


def test_minhash_and_tfidf_guards_agree_on_obvious_cases() -> None:
    for method in ("minhash", "tfidf"):
        produced = NoveltyGuard(DedupConfig(method=method, threshold=0.9))
        produced.admit("a", FIRST)
        assert produced.admit("b", NEAR).is_novel is False
        assert produced.admit("c", DIFFERENT).is_novel is True


def test_verdicts_serialize_with_their_evidence() -> None:
    produced = guard()
    produced.observe("a", FIRST)
    payload = produced.check(NEAR).as_dict()
    assert payload == {
        "is_novel": False,
        "nearest": "a",
        "similarity": 1.0,
        "method": "exact",
    }
