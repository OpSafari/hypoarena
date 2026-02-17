"""MinHash signatures: determinism, accuracy against true Jaccard, bounds."""

from __future__ import annotations

from random import Random

import pytest

from hypoarena.dedup import (
    jaccard_similarity,
    minhash_signature,
    minhash_similarity,
    signature_standard_error,
    text_signature,
)
from hypoarena.errors import ValidationError

FIRST = "Protein A increases cell growth in HeLa cells"
PARAPHRASE = "In HeLa cells, protein A increases the growth of cells"
UNRELATED = "Batch effects were regressed out before summarizing results"


def test_signatures_are_deterministic_and_seed_sensitive() -> None:
    first = text_signature(FIRST, 64)
    assert first == text_signature(FIRST, 64)
    assert len(first) == 64
    assert first != text_signature(FIRST, 64, seed=9)


def test_identical_texts_have_identical_signatures() -> None:
    assert minhash_similarity(text_signature(FIRST, 64), text_signature(FIRST, 64)) == (
        1.0
    )


def test_empty_texts_follow_the_zero_convention() -> None:
    empty = text_signature("", 16)
    assert empty == (0,) * 16
    assert minhash_similarity(empty, empty) == 1.0


def test_estimates_track_true_jaccard_within_the_error_bound() -> None:
    rng = Random(270106)
    alphabet = [f"term-{index}" for index in range(40)]
    for _ in range(12):
        left = set(rng.sample(alphabet, rng.randrange(8, 30)))
        right = set(rng.sample(alphabet, rng.randrange(8, 30)))
        truth = len(left & right) / len(left | right)
        estimate = minhash_similarity(
            minhash_signature(frozenset(left), 256),
            minhash_signature(frozenset(right), 256),
        )
        assert abs(estimate - truth) < 5 * signature_standard_error(256)


def test_paraphrases_estimate_higher_than_unrelated_pairs() -> None:
    paraphrase = minhash_similarity(
        text_signature(FIRST, 256), text_signature(PARAPHRASE, 256)
    )
    unrelated = minhash_similarity(
        text_signature(FIRST, 256), text_signature(UNRELATED, 256)
    )
    assert paraphrase > unrelated
    assert paraphrase == pytest.approx(
        jaccard_similarity(FIRST, PARAPHRASE), abs=4 * signature_standard_error(256)
    )


def test_more_permutations_reduce_the_error_bound() -> None:
    errors = [signature_standard_error(count) for count in (16, 64, 256, 1024)]
    assert errors == sorted(errors, reverse=True)
    assert signature_standard_error(64) == pytest.approx(0.125)
    assert signature_standard_error(256) == pytest.approx(0.0625)


def test_degenerate_inputs_are_rejected() -> None:
    with pytest.raises(ValidationError, match="num_perm"):
        minhash_signature(frozenset({"a"}), 0)
    with pytest.raises(ValidationError, match="num_perm"):
        signature_standard_error(0)
    with pytest.raises(ValidationError, match="same length"):
        minhash_similarity((1, 2), (1, 2, 3))
    with pytest.raises(ValidationError, match="must not be empty"):
        minhash_similarity((), ())
    with pytest.raises(ValidationError, match="permutation"):
        from hypoarena.dedup import hash_shingle

        hash_shingle("a", -1)
