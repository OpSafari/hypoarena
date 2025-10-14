"""Dedup configuration, cluster records and pair-based metrics."""

from __future__ import annotations

import pytest

from hypoarena.dedup import (
    DEDUP_METHODS,
    DEFAULT_BANDS,
    DEFAULT_NUM_PERM,
    DEFAULT_THRESHOLD,
    ClusterMetrics,
    DedupConfig,
    DuplicateCluster,
    cluster_metrics,
)
from hypoarena.errors import ValidationError


def cluster(members: tuple[str, ...]) -> DuplicateCluster:
    return DuplicateCluster(members, members[0], "minhash")


def test_defaults_are_golden() -> None:
    config = DedupConfig()
    assert config.method == "minhash"
    assert config.threshold == DEFAULT_THRESHOLD == 0.8
    assert config.num_perm == DEFAULT_NUM_PERM == 128
    assert config.bands == DEFAULT_BANDS == 32
    assert config.use_lsh is True
    assert config.rows == 4
    assert DEDUP_METHODS == ("exact", "jaccard", "tfidf", "minhash")


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown dedup method"):
        DedupConfig(method="embeddings")
    with pytest.raises(ValidationError, match="threshold"):
        DedupConfig(threshold=0.0)
    with pytest.raises(ValidationError, match="threshold"):
        DedupConfig(threshold=1.5)
    with pytest.raises(ValidationError, match="ngram_size"):
        DedupConfig(ngram_size=0)
    with pytest.raises(ValidationError, match="divisible"):
        DedupConfig(num_perm=100, bands=32)


def test_lsh_can_be_disabled_and_the_inflection_reported() -> None:
    config = DedupConfig(num_perm=100, bands=32, use_lsh=False)
    assert config.fingerprint() != DedupConfig().fingerprint()
    assert 0.0 < DedupConfig().inflection() < 1.0


def test_clusters_validate_their_members() -> None:
    with pytest.raises(ValidationError, match="at least two"):
        DuplicateCluster(("a",), "a", "exact")
    with pytest.raises(ValidationError, match="sorted and unique"):
        DuplicateCluster(("b", "a"), "b", "exact")
    with pytest.raises(ValidationError, match="must be a member"):
        DuplicateCluster(("a", "b"), "c", "exact")


def test_cluster_helpers() -> None:
    produced = cluster(("a", "b", "c"))
    assert produced.size == 3
    assert produced.contains("b") and not produced.contains("z")
    assert produced.pairs() == (("a", "b"), ("a", "c"), ("b", "c"))
    payload = produced.as_dict()
    assert payload["representative"] == "a"
    assert payload["members"] == ["a", "b", "c"]
    assert payload["similarities"] == []


def test_metrics_count_pairs_in_both_directions() -> None:
    truth = [("a", "b", "c"), ("d", "e")]
    perfect = cluster_metrics([cluster(("a", "b", "c")), cluster(("d", "e"))], truth)
    assert perfect.as_dict() == {
        "clusters": 2,
        "members": 5,
        "pairs_found": 4,
        "pairs_expected": 4,
        "true_positives": 4,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_splitting_a_group_costs_recall_not_precision() -> None:
    metrics = cluster_metrics([cluster(("a", "b"))], [("a", "b", "c")])
    assert metrics.recall == pytest.approx(1 / 3)
    assert metrics.precision == 1.0
    assert metrics.false_negatives == 2
    assert metrics.false_positives == 0


def test_merging_groups_costs_precision_not_recall() -> None:
    metrics = cluster_metrics([cluster(("a", "b", "c"))], [("a", "b"), ("c",)])
    assert metrics.precision == pytest.approx(1 / 3)
    assert metrics.recall == 1.0
    assert metrics.false_positives == 2


def test_empty_inputs_are_handled() -> None:
    empty = cluster_metrics([], [])
    assert isinstance(empty, ClusterMetrics)
    assert empty.precision == 1.0 and empty.recall == 1.0
    assert empty.f1 == 1.0
    assert cluster_metrics([], [("a", "b")]).recall == 0.0
    assert cluster_metrics([cluster(("a", "b"))], []).precision == 0.0
