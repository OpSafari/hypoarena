"""The torch-free featurizer is deterministic and its vocabulary is stable.

These run without torch: the feature layer is pure NumPy over the seeded
synthetic dataset, so identical input must give identical features and a sorted,
deduplicated vocabulary.
"""

from __future__ import annotations

import numpy as np

from hypoarena.ranker import (
    RankerDatasetConfig,
    RankerFeaturizer,
    synthetic_ranking_dataset,
)


def test_same_config_gives_identical_features() -> None:
    config = RankerDatasetConfig(n_examples=48, seed=3)
    first = RankerFeaturizer().fit_transform(synthetic_ranking_dataset(config))
    second = RankerFeaturizer().fit_transform(synthetic_ranking_dataset(config))
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.scores, second.scores)
    assert first.vocabulary == second.vocabulary


def test_vocabulary_is_sorted_and_deduplicated() -> None:
    data = RankerFeaturizer().fit_transform(
        synthetic_ranking_dataset(RankerDatasetConfig(n_examples=64, seed=1))
    )
    vocab = list(data.vocabulary)
    assert vocab == sorted(vocab)
    assert len(vocab) == len(set(vocab))


def test_a_different_ngram_size_changes_the_vocabulary() -> None:
    examples = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=32, seed=2))
    unigram = RankerFeaturizer(ngram_size=1).fit_transform(examples)
    bigram = RankerFeaturizer(ngram_size=2).fit_transform(examples)
    assert unigram.vocabulary != bigram.vocabulary


def test_transform_of_a_subset_keeps_the_fitted_width() -> None:
    examples = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=40, seed=4))
    featurizer = RankerFeaturizer()
    fitted = featurizer.fit_transform(examples)
    projected = featurizer.transform(examples[:5])
    assert projected.shape == (5, fitted.features.shape[1])
