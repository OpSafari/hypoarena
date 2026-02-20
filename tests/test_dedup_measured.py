"""Measured duplicate recovery on synthetic corpora with planted clusters.

Corpus: ``SyntheticConfig(seed=270106, chains=2, chain_length=3,
paraphrases_per_link=3, distractor_documents=4)`` — 12 planted finding documents
forming 4 clusters of 3, plus 4 distractors. The numbers asserted here were
produced by running these configurations; they describe surface similarity
mechanics on synthetic text, not performance on real literature.
"""

from __future__ import annotations

import pytest

from hypoarena.dedup import DedupConfig, DuplicateFinder, cluster_metrics
from hypoarena.synthetic import SyntheticConfig, build_bundle

SEED = 270106


def corpus_texts(
    *, include_rivals: bool = False
) -> tuple[dict[str, str], list[tuple[str, ...]]]:
    bundle = build_bundle(
        SyntheticConfig(
            seed=SEED,
            chains=2,
            chain_length=3,
            paraphrases_per_link=3,
            distractor_documents=4,
        )
    )
    kinds = (
        ("planted", "competing", "contradiction") if include_rivals else ("planted",)
    )
    texts = {
        finding.document_id: finding.sentence
        for finding in bundle.findings
        if finding.link.kind in kinds
    }
    texts.update(
        {
            document_id: bundle.corpus.document(document_id).text
            for document_id in bundle.truth.distractor_ids
        }
    )
    truth = [tuple(cluster.document_ids) for cluster in bundle.truth.clusters]
    return texts, truth


def measure(config: DedupConfig, *, include_rivals: bool = False) -> object:
    texts, truth = corpus_texts(include_rivals=include_rivals)
    return cluster_metrics(DuplicateFinder(config).find(texts), truth)


def content_config(threshold: float, method: str = "minhash") -> DedupConfig:
    return DedupConfig(
        method=method,
        threshold=threshold,
        ngram_size=1,
        shingle_unit="word",
        content_only=True,
        num_perm=256,
        bands=64,
    )


def test_content_shingles_recover_every_planted_cluster() -> None:
    metrics = measure(content_config(0.6))
    assert metrics.as_dict() == {
        "clusters": 4,
        "members": 12,
        "pairs_found": 12,
        "pairs_expected": 12,
        "true_positives": 12,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_jaccard_agrees_with_the_minhash_estimate() -> None:
    assert measure(content_config(0.6, method="jaccard")).f1 == 1.0


def test_a_stricter_threshold_costs_recall_not_precision() -> None:
    metrics = measure(content_config(0.7))
    assert metrics.precision == 1.0
    assert metrics.recall == pytest.approx(0.8333, abs=1e-4)
    assert metrics.false_negatives == 2


def test_surface_shingles_are_clearly_worse_on_paraphrases() -> None:
    surface_words = measure(
        DedupConfig(
            method="minhash",
            threshold=0.6,
            ngram_size=1,
            shingle_unit="word",
            num_perm=256,
            bands=64,
        )
    )
    surface_chars = measure(
        DedupConfig(
            method="minhash",
            threshold=0.5,
            ngram_size=3,
            num_perm=256,
            bands=64,
        )
    )
    assert (surface_words.precision, surface_words.recall) == (0.75, 0.5)
    assert (surface_chars.precision, surface_chars.recall) == (0.4, 0.5)
    assert measure(content_config(0.6)).f1 > surface_words.f1


def test_exact_matching_finds_no_paraphrases() -> None:
    metrics = measure(DedupConfig(method="exact"))
    assert metrics.clusters == 0
    assert metrics.recall == 0.0
    assert metrics.precision == 1.0


def test_rival_hypotheses_are_merged_by_surface_similarity() -> None:
    # competing and contradicting findings name the same variables, so a purely
    # lexical metric cannot tell "restated" from "about the same pair"
    permissive = measure(content_config(0.6), include_rivals=True)
    stricter = measure(content_config(0.7), include_rivals=True)
    assert permissive.recall == 1.0
    assert permissive.precision == 0.5
    assert stricter.precision == 0.625
    assert stricter.false_positives < permissive.false_positives


def test_distractors_never_join_a_cluster() -> None:
    bundle = build_bundle(
        SyntheticConfig(seed=SEED, chains=2, chain_length=3, paraphrases_per_link=3)
    )
    texts, _ = corpus_texts()
    clusters = DuplicateFinder(content_config(0.6)).find(texts)
    distractors = set(bundle.truth.distractor_ids)
    assert distractors
    for cluster in clusters:
        assert not (set(cluster.members) & distractors)
