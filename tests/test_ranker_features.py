"""The torch-free feature layer: TF-IDF matrix, labels and shape validation."""

from __future__ import annotations

import numpy as np
import pytest

from hypoarena.errors import ValidationError
from hypoarena.ranker import (
    RankerData,
    RankerDatasetConfig,
    RankerFeaturizer,
    synthetic_ranking_dataset,
)


def make_data(n: int = 128, seed: int = 270106) -> RankerData:
    examples = synthetic_ranking_dataset(RankerDatasetConfig(n_examples=n, seed=seed))
    return RankerFeaturizer().fit_transform(examples)


def test_feature_and_score_shapes_agree() -> None:
    data = make_data(n=128)
    assert data.features.shape == (128, len(data.vocabulary))
    assert data.scores.shape == (128,)
    assert len(data) == 128


def test_scores_match_the_dataset() -> None:
    config = RankerDatasetConfig(n_examples=64, seed=5)
    examples = synthetic_ranking_dataset(config)
    data = RankerFeaturizer().fit_transform(examples)
    assert np.allclose(data.scores, [example.score for example in examples])


def test_vocabulary_captures_the_signal_tokens() -> None:
    config = RankerDatasetConfig(n_examples=128, seed=270106)
    data = RankerFeaturizer().fit_transform(synthetic_ranking_dataset(config))
    assert set(config.signal_tokens) <= set(data.vocabulary)
    allowed = set(config.signal_tokens) | set(config.noise_tokens)
    assert set(data.vocabulary) <= allowed


def test_rows_are_l2_normalized() -> None:
    data = make_data()
    norms = np.linalg.norm(data.features, axis=1)
    assert np.allclose(norms, 1.0)


def test_featurization_is_deterministic() -> None:
    a = make_data(seed=9)
    b = make_data(seed=9)
    assert np.array_equal(a.features, b.features)
    assert a.vocabulary == b.vocabulary


def test_transform_reuses_the_fitted_vocabulary() -> None:
    config = RankerDatasetConfig(n_examples=128, seed=3)
    examples = synthetic_ranking_dataset(config)
    featurizer = RankerFeaturizer()
    featurizer.fit_transform(examples)
    projected = featurizer.transform(examples[:4])
    assert projected.shape == (4, len(featurizer.vectorizer.vocabulary_))


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RankerFeaturizer().fit_transform(())


def test_ranker_data_rejects_shape_mismatches() -> None:
    features = np.zeros((3, 2))
    with pytest.raises(ValidationError):
        RankerData(features=features, scores=np.zeros(2), vocabulary=("a", "b"))
    with pytest.raises(ValidationError):
        RankerData(features=features, scores=np.zeros(3), vocabulary=("a", "b", "c"))
    with pytest.raises(ValidationError):
        RankerData(features=np.zeros(3), scores=np.zeros(3), vocabulary=())
