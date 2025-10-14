"""Duplicate clustering across every configured metric."""

from __future__ import annotations

import pytest

from hypoarena.dedup import DedupConfig, DuplicateFinder, union_groups
from hypoarena.errors import ValidationError

TEXTS = {
    "a": "Protein A increases cell growth in HeLa cells",
    "b": "Protein A increases cell growth in HeLa cells.",
    "c": "protein a   increases CELL growth in hela cells!",
    "d": "Batch effects were regressed out before summarizing results",
    "e": "Batch effects were regressed out before summarizing the results",
}
TRUTH = [("a", "b", "c"), ("d", "e")]


def test_union_find_groups_by_transitivity() -> None:
    groups = union_groups(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
    assert groups == [["a", "b", "c"], ["d"]]


def test_union_find_is_order_independent() -> None:
    forward = union_groups(["a", "b", "c"], [("a", "b"), ("b", "c")])
    backward = union_groups(["a", "b", "c"], [("b", "c"), ("a", "b")])
    assert forward == backward


def test_exact_method_finds_only_identical_text() -> None:
    clusters = DuplicateFinder(DedupConfig(method="exact")).find(TEXTS)
    assert [cluster.members for cluster in clusters] == [("a", "b", "c")]


def test_jaccard_method_groups_paraphrases_at_a_low_threshold() -> None:
    clusters = DuplicateFinder(DedupConfig(method="jaccard", threshold=0.6)).find(TEXTS)
    members = {cluster.members for cluster in clusters}
    assert ("a", "b", "c") in members
    assert ("d", "e") in members


def test_a_high_threshold_keeps_only_near_identical_text() -> None:
    clusters = DuplicateFinder(DedupConfig(method="jaccard", threshold=0.99)).find(
        TEXTS
    )
    assert [cluster.members for cluster in clusters] == [("a", "b", "c")]


def test_minhash_with_lsh_matches_the_verified_pairs() -> None:
    strict = DuplicateFinder(DedupConfig(method="minhash", threshold=0.95)).find(TEXTS)
    assert [cluster.members for cluster in strict] == [("a", "b", "c")]
    loose = DuplicateFinder(
        DedupConfig(method="minhash", threshold=0.55, num_perm=256, bands=64)
    ).find(TEXTS)
    assert {cluster.members for cluster in loose} >= {("a", "b", "c"), ("d", "e")}


def test_lsh_can_be_disabled_without_changing_results() -> None:
    filtered = DuplicateFinder(DedupConfig(method="minhash", threshold=0.95)).find(
        TEXTS
    )
    brute = DuplicateFinder(
        DedupConfig(method="minhash", threshold=0.95, use_lsh=False)
    ).find(TEXTS)
    assert [cluster.members for cluster in filtered] == [
        cluster.members for cluster in brute
    ]


def test_tfidf_method_groups_by_cosine_similarity() -> None:
    clusters = DuplicateFinder(DedupConfig(method="tfidf", threshold=0.9)).find(TEXTS)
    members = {cluster.members for cluster in clusters}
    assert ("a", "b", "c") in members


def test_clusters_record_their_verified_similarities() -> None:
    clusters = DuplicateFinder(DedupConfig(method="jaccard", threshold=0.6)).find(TEXTS)
    for cluster in clusters:
        assert cluster.method == "jaccard"
        assert len(cluster.similarities) >= 1
        for _left, _right, score in cluster.similarities:
            assert 0.6 <= score <= 1.0


def test_results_are_deterministic_and_sorted() -> None:
    config = DedupConfig(method="minhash", threshold=0.6, num_perm=256, bands=64)
    first = DuplicateFinder(config).find(TEXTS)
    second = DuplicateFinder(config).find(dict(reversed(list(TEXTS.items()))))
    assert [cluster.members for cluster in first] == [
        cluster.members for cluster in second
    ]
    representatives = [cluster.representative for cluster in first]
    assert representatives == sorted(representatives)


def test_small_inputs_are_handled() -> None:
    assert DuplicateFinder().find({}) == ()
    assert DuplicateFinder().find({"a": "only one"}) == ()
    with pytest.raises(ValidationError, match="unknown dedup method"):
        DedupConfig(method="magic")
