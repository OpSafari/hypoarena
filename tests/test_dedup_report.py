"""Dedup reports: aggregates, lookups and integrity rules."""

from __future__ import annotations

import pytest

from hypoarena.dedup import DedupConfig, DedupReport, DuplicateCluster, DuplicateFinder
from hypoarena.errors import ValidationError

TEXTS = {
    "a": "Protein A increases cell growth in HeLa cells",
    "b": "Protein A increases cell growth in HeLa cells.",
    "c": "kinase K1 phosphorylates protein A",
    "d": "kinase K1 phosphorylates protein A!",
}


def report() -> DedupReport:
    config = DedupConfig(method="exact")
    finder = DuplicateFinder(config)
    return DedupReport(finder.find(TEXTS), config, len(TEXTS))


def test_report_aggregates_the_pass() -> None:
    produced = report()
    assert produced.total == 4
    assert produced.duplicated == 4
    assert produced.duplicate_rate == 1.0
    assert produced.duplicate_ids() == ("a", "b", "c", "d")


def test_cluster_lookup_works_in_both_directions() -> None:
    produced = report()
    cluster = produced.cluster_of("a")
    assert cluster is not None
    assert cluster.members == ("a", "b")
    assert produced.cluster_of("missing") is None


def test_summary_dict_is_complete() -> None:
    payload = report().as_dict()
    assert sorted(payload) == [
        "clusters",
        "config_fingerprint",
        "duplicate_rate",
        "duplicated",
        "method",
        "signature",
        "threshold",
        "total",
    ]
    assert payload["method"] == "exact"
    assert payload["clusters"] == 2


def test_overlapping_clusters_are_rejected() -> None:
    config = DedupConfig(method="exact")
    first = DuplicateCluster(("a", "b"), "a", "exact")
    second = DuplicateCluster(("b", "c"), "b", "exact")
    with pytest.raises(ValidationError, match="more than one cluster"):
        DedupReport((first, second), config, 3)


def test_negative_totals_are_rejected() -> None:
    with pytest.raises(ValidationError, match="total"):
        DedupReport((), DedupConfig(), -1)


def test_an_empty_pass_has_a_zero_rate_and_a_stable_signature() -> None:
    produced = DedupReport((), DedupConfig(), 0)
    assert produced.duplicate_rate == 0.0
    assert produced.signature() == DedupReport((), DedupConfig(), 0).signature()
    assert produced.signature() != report().signature()


def test_metrics_can_be_taken_straight_from_the_report() -> None:
    produced = report()
    metrics = produced.metrics_against([("a", "b"), ("c", "d")])
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
