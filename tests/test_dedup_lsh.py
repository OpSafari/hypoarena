"""LSH banding: bucketing, the S-curve and the tuning rule of thumb."""

from __future__ import annotations

from random import Random

import pytest

from hypoarena.dedup import (
    candidate_pairs,
    candidate_probability,
    lsh_bands,
    lsh_inflection,
    lsh_rows,
    minhash_signature,
    minhash_similarity,
    text_signature,
)
from hypoarena.errors import ValidationError

FIRST = "Protein A increases cell growth in HeLa cells"
PARAPHRASE = "In HeLa cells, protein A increases the growth of cells"
UNRELATED = "Batch effects were regressed out before summarizing results"


def test_band_splitting_uses_every_slot_exactly_once() -> None:
    signature = tuple(range(12))
    bands = lsh_bands(signature, 3)
    assert bands == ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
    assert lsh_rows(12, 3) == 4


def test_indivisible_configurations_are_rejected() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        lsh_rows(10, 3)
    with pytest.raises(ValidationError, match="bands"):
        lsh_rows(12, 0)
    with pytest.raises(ValidationError, match="num_perm"):
        lsh_rows(0, 1)


def test_similar_texts_share_a_band_and_unrelated_ones_do_not() -> None:
    # measured: char-3-gram Jaccard(FIRST, PARAPHRASE) = 0.622, so 32 bands of
    # two rows propose the pair with probability 1 - (1 - 0.622**2)**32 ~ 1.0
    signatures = {
        "a": text_signature(FIRST, 64),
        "b": text_signature(PARAPHRASE, 64),
        "c": text_signature(UNRELATED, 64),
    }
    pairs = candidate_pairs(signatures, bands=32)
    assert ("a", "b") in pairs
    assert ("a", "c") not in pairs
    assert ("b", "c") not in pairs


def test_candidate_pairs_are_sorted_and_unique() -> None:
    same = text_signature(FIRST, 64)
    pairs = candidate_pairs({"z": same, "a": same, "m": same}, bands=8)
    assert pairs == {("a", "m"), ("a", "z"), ("m", "z")}
    assert all(first < second for first, second in pairs)


def test_the_s_curve_is_monotone_and_bounded() -> None:
    values = [candidate_probability(value, 64, 16) for value in (0.1, 0.3, 0.5, 0.9)]
    assert values == sorted(values)
    assert 0.0 <= values[0] < values[-1] <= 1.0
    assert candidate_probability(1.0, 64, 16) == pytest.approx(1.0)
    assert candidate_probability(0.0, 64, 16) == 0.0
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        candidate_probability(1.5, 64, 16)


def test_the_inflection_matches_the_rule_of_thumb() -> None:
    assert lsh_inflection(64, 16) == pytest.approx((1 / 16) ** (1 / 4))
    # more bands means shorter buckets, so a lower similarity already collides
    assert lsh_inflection(128, 32) < lsh_inflection(128, 8)
    assert 0.0 < lsh_inflection(128, 32) < 1.0


def test_more_bands_raise_recall_for_similar_pairs() -> None:
    rng = Random(7)
    signatures = {}
    for index in range(24):
        text = " ".join(
            rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(30)
        )
        signatures[f"doc{index:02d}"] = minhash_signature(
            frozenset(text.split()), 64, seed=index
        )
    few = candidate_pairs(signatures, bands=8)
    many = candidate_pairs(signatures, bands=32)
    assert few <= many


def test_candidates_are_a_superset_of_the_verified_duplicates() -> None:
    signatures = {
        "a": text_signature(FIRST, 64),
        "b": text_signature(PARAPHRASE, 64),
        "c": text_signature(UNRELATED, 64),
    }
    pairs = candidate_pairs(signatures, bands=32)
    for first, second in pairs:
        similarity = minhash_similarity(signatures[first], signatures[second])
        assert similarity > 0.0
