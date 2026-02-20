"""Content-only shingling: what it changes and why it matters."""

from __future__ import annotations

from hypoarena.dedup import (
    DedupConfig,
    DuplicateFinder,
    content_shingles,
    jaccard_similarity,
    shingle_set,
)

PLAIN = "gene G3 is required for metabolite M1 in mouse liver"
REPORTED = "We observed that gene G3 is required for metabolite M1 in mouse liver"
ASSAYED = "Assays in mouse liver show that gene G3 licenses metabolite M1"
UNRELATED = "kinase K1 inhibits protein A in HeLa cells"


def test_content_shingles_drop_function_and_reporting_words() -> None:
    assert content_shingles(PLAIN, 1) == content_shingles(REPORTED, 1)
    assert "observed" not in content_shingles(REPORTED, 1)
    assert "gene" in content_shingles(REPORTED, 1)


def test_content_shingles_support_word_bigrams() -> None:
    assert "gene g3" in content_shingles(PLAIN, 2)
    assert content_shingles(PLAIN, 2, words=False)


def test_paraphrases_separate_from_unrelated_findings() -> None:
    paraphrase = jaccard_similarity(
        REPORTED, ASSAYED, n=1, words=True, content_only=True
    )
    unrelated = jaccard_similarity(
        REPORTED, UNRELATED, n=1, words=True, content_only=True
    )
    surface = jaccard_similarity(REPORTED, ASSAYED, n=1, words=True)
    assert paraphrase > surface
    assert paraphrase > 0.6
    assert unrelated < 0.3


def test_shingle_set_dispatches_on_the_content_flag() -> None:
    assert shingle_set(PLAIN, 1, words=True, content_only=False) != shingle_set(
        PLAIN, 1, words=True, content_only=True
    )
    assert shingle_set(PLAIN, 3) == shingle_set(PLAIN, 3, content_only=False)


def test_content_only_is_a_configuration_option() -> None:
    assert DedupConfig().content_only is False
    assert DedupConfig(content_only=True).fingerprint() != DedupConfig().fingerprint()


def test_content_only_clustering_recovers_planted_paraphrases() -> None:
    texts = {"a": PLAIN, "b": REPORTED, "c": ASSAYED, "d": UNRELATED}
    clustered = DuplicateFinder(
        DedupConfig(
            method="jaccard",
            shingle_unit="word",
            ngram_size=1,
            content_only=True,
            threshold=0.6,
        )
    ).find(texts)
    assert [cluster.members for cluster in clustered] == [("a", "b", "c")]
    surface = DuplicateFinder(
        DedupConfig(method="jaccard", shingle_unit="word", ngram_size=1, threshold=0.6)
    ).find(texts)
    # surface shingles catch the near-identical pair but miss the reframed one
    assert [cluster.members for cluster in surface] == [("a", "b")]


def test_content_only_applies_to_minhash_signatures() -> None:
    texts = {"a": REPORTED, "b": ASSAYED}
    clustered = DuplicateFinder(
        DedupConfig(
            method="minhash",
            shingle_unit="word",
            ngram_size=1,
            content_only=True,
            threshold=0.6,
            num_perm=128,
            bands=32,
        )
    ).find(texts)
    assert [cluster.members for cluster in clustered] == [("a", "b")]
