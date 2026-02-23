"""Golden dedup payload.

Configuration: word unigrams over content tokens only (``content_only=True``),
Jaccard metric, threshold 0.6. The scores below were produced by running the
finder on these five synthetic sentences; 0.857143 is 6/7, the overlap between
the plain statement and the version that adds one condition.
"""

from __future__ import annotations

import pytest

from hypoarena.dedup import DedupConfig, DuplicateFinder
from hypoarena.serialize import dedup_config_to_dict, dedup_report_to_dict

TEXTS = {
    "doc_a": "Protein A increases cell growth in HeLa cells",
    "doc_b": "We observed that protein A increases cell growth in HeLa cells",
    "doc_c": "Assays in HeLa cells show that protein A increases cell growth",
    "doc_d": "Protein A increases cell growth in HeLa cells under hypoxia",
    "doc_e": "kinase K1 phosphorylates protein A",
}
CONFIG = DedupConfig(
    method="jaccard",
    threshold=0.6,
    ngram_size=1,
    shingle_unit="word",
    content_only=True,
)
GOLDEN_CLUSTERS = [
    {
        "representative": "doc_a",
        "method": "jaccard",
        "size": 4,
        "members": ["doc_a", "doc_b", "doc_c", "doc_d"],
        "similarities": [
            {"left": "doc_a", "right": "doc_b", "score": 1.0},
            {"left": "doc_a", "right": "doc_c", "score": 1.0},
            {"left": "doc_a", "right": "doc_d", "score": 0.857143},
            {"left": "doc_b", "right": "doc_c", "score": 1.0},
            {"left": "doc_b", "right": "doc_d", "score": 0.857143},
            {"left": "doc_c", "right": "doc_d", "score": 0.857143},
        ],
    }
]


def report() -> object:
    return DuplicateFinder(CONFIG).report(TEXTS)


def test_clusters_match_the_golden_payload() -> None:
    produced = report()
    assert [cluster.as_dict() for cluster in produced.clusters] == GOLDEN_CLUSTERS


def test_the_unrelated_sentence_stays_out_of_the_cluster() -> None:
    produced = report()
    assert produced.cluster_of("doc_e") is None
    assert produced.duplicated == 4
    assert produced.total == 5
    assert produced.duplicate_rate == 0.8


def test_the_report_summary_is_golden() -> None:
    payload = dedup_report_to_dict(report())
    assert payload["total"] == 5
    assert payload["schema_version"] == "1.0"
    assert len(payload["clusters"]) == 1
    assert payload["config"] == dedup_config_to_dict(CONFIG)


def test_similarity_scores_are_the_measured_fractions() -> None:
    produced = report()
    scores = {
        (left, right): score for left, right, score in produced.clusters[0].similarities
    }
    # the report rounds for display; the stored value is the exact fraction
    assert scores[("doc_a", "doc_d")] == pytest.approx(6 / 7)
    assert scores[("doc_a", "doc_b")] == 1.0
