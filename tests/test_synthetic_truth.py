"""Planted truth: link enumeration, clusters and pair counts."""

from __future__ import annotations

from hypoarena.synthetic import (
    PlantedCluster,
    PlantedTruth,
    SyntheticConfig,
    generate,
    truth_of,
)


def truth(paraphrases: int = 2) -> PlantedTruth:
    return truth_of(
        generate(
            SyntheticConfig(
                seed=21, chains=2, chain_length=3, paraphrases_per_link=paraphrases
            )
        )
    )


def test_true_links_cover_every_chain_pair() -> None:
    planted = truth()
    assert len(planted.true_links()) == 2 * 2
    assert all(link.is_true for link in planted.true_links())


def test_clusters_group_the_paraphrase_documents() -> None:
    planted = truth(paraphrases=3)
    assert len(planted.clusters) == 4
    for cluster in planted.clusters:
        assert isinstance(cluster, PlantedCluster)
        assert cluster.size == 3
        assert len(set(cluster.document_ids)) == 3


def test_cluster_lookup_by_link_key() -> None:
    planted = truth()
    link = planted.true_links()[0]
    assert planted.cluster_for(link.key()).statement == link.statement
    assert planted.cluster_for(("missing", "increases", "entries")) is None


def test_paraphrase_pairs_count_combinations() -> None:
    assert truth(paraphrases=2).paraphrase_pairs() == 4
    assert truth(paraphrases=3).paraphrase_pairs() == 4 * 3


def test_cluster_pairs_are_unordered_and_sorted() -> None:
    cluster = truth().clusters[0]
    pairs = cluster.pairs()
    assert len(pairs) == 1
    assert pairs[0][0] < pairs[0][1]


def test_summary_counts_are_complete() -> None:
    planted = truth()
    summary = planted.as_dict()
    assert summary["chains"] == 2
    assert summary["true_links"] == 4
    assert summary["competing"] == 2
    assert summary["contradictions"] == 2
    assert summary["distractors"] == 4
    assert summary["clusters"] == 4
