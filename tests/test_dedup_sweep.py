"""Seeded sweeps over dedup configurations.

The invariants that must hold for every method and threshold: results never
depend on dictionary order, clusters are disjoint and sorted, raising the
threshold can only remove pairs, and content-only shingles recover at least as
many planted clusters as surface shingles do.
"""

from __future__ import annotations

import pytest

from hypoarena.dedup import DedupConfig, DuplicateFinder, cluster_metrics
from hypoarena.synthetic import SyntheticConfig, build_bundle

METHODS = ("exact", "jaccard", "tfidf", "minhash")
THRESHOLDS = (0.4, 0.6, 0.8, 1.0)
SEEDS = (11, 270106)


def texts_and_truth(seed: int) -> tuple[dict[str, str], list[tuple[str, ...]]]:
    bundle = build_bundle(
        SyntheticConfig(seed=seed, chains=2, chain_length=3, paraphrases_per_link=3)
    )
    texts = {
        finding.document_id: finding.sentence
        for finding in bundle.findings
        if finding.link.kind == "planted"
    }
    texts.update(
        {
            document_id: bundle.corpus.document(document_id).text
            for document_id in bundle.truth.distractor_ids
        }
    )
    truth = [tuple(cluster.document_ids) for cluster in bundle.truth.clusters]
    return texts, truth


def config(method: str, threshold: float, content_only: bool) -> DedupConfig:
    return DedupConfig(
        method=method,
        threshold=threshold,
        ngram_size=1 if method != "exact" else 3,
        shingle_unit="word",
        content_only=content_only,
        num_perm=128,
        bands=32,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_clusters_are_disjoint_sorted_and_reproducible(
    method: str, threshold: float
) -> None:
    texts, _ = texts_and_truth(11)
    finder = DuplicateFinder(config(method, threshold, content_only=True))
    first = finder.find(texts)
    second = finder.find(dict(reversed(list(texts.items()))))
    assert [cluster.members for cluster in first] == [
        cluster.members for cluster in second
    ]
    seen: set[str] = set()
    for cluster in first:
        assert list(cluster.members) == sorted(set(cluster.members))
        assert not (seen & set(cluster.members))
        seen.update(cluster.members)
        assert cluster.representative == cluster.members[0]


@pytest.mark.parametrize("method", METHODS)
def test_raising_the_threshold_never_adds_pairs(method: str) -> None:
    texts, _ = texts_and_truth(11)
    counts = [
        sum(
            len(cluster.pairs())
            for cluster in DuplicateFinder(
                config(method, threshold, content_only=True)
            ).find(texts)
        )
        for threshold in THRESHOLDS
    ]
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("seed", SEEDS)
def test_content_only_shingles_never_lose_recall(seed: int) -> None:
    texts, truth = texts_and_truth(seed)
    surface = cluster_metrics(
        DuplicateFinder(config("minhash", 0.6, content_only=False)).find(texts), truth
    )
    content = cluster_metrics(
        DuplicateFinder(config("minhash", 0.6, content_only=True)).find(texts), truth
    )
    assert content.recall >= surface.recall
    assert content.precision >= surface.precision


@pytest.mark.parametrize("seed", SEEDS)
def test_content_only_recovers_every_planted_cluster(seed: int) -> None:
    texts, truth = texts_and_truth(seed)
    metrics = cluster_metrics(
        DuplicateFinder(config("minhash", 0.6, content_only=True)).find(texts), truth
    )
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0


def test_exact_matching_never_groups_paraphrases() -> None:
    texts, truth = texts_and_truth(270106)
    metrics = cluster_metrics(
        DuplicateFinder(config("exact", 1.0, content_only=False)).find(texts), truth
    )
    assert metrics.pairs_found == 0
    assert metrics.recall == 0.0
