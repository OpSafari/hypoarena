"""Golden digests over the synthetic ranker dataset and its vocabulary.

The dataset and feature vocabulary are serialized outputs of the torch-free
layer; pinning their digests makes any drift in the planted generator or the
featurizer a deliberate, reviewed change. No torch is required here.
"""

from __future__ import annotations

from hypoarena.ids import content_hash
from hypoarena.ranker import (
    RankerDatasetConfig,
    RankerFeaturizer,
    synthetic_ranking_dataset,
)

DATASET_GOLDEN = "92faf4ef6e61d527"
VOCAB_GOLDEN = "791925293a1ffc68"


def dataset_payload() -> list[dict[str, object]]:
    data = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=32, seed=270106))
    return [
        {
            "id": example.example_id,
            "text": example.text,
            "score": round(example.score, 6),
            "signals": list(example.signals),
        }
        for example in data
    ]


def test_dataset_matches_the_golden_digest() -> None:
    assert content_hash(dataset_payload()) == DATASET_GOLDEN


def test_vocabulary_matches_the_golden_digest() -> None:
    examples = synthetic_ranking_dataset(
        RankerDatasetConfig(n_examples=128, seed=270106)
    )
    data = RankerFeaturizer().fit_transform(examples)
    assert content_hash(list(data.vocabulary)) == VOCAB_GOLDEN


def test_dataset_digest_is_stable_across_calls() -> None:
    assert content_hash(dataset_payload()) == content_hash(dataset_payload())
