"""An optional trainable ranker over TF-IDF features (needs the ``torch`` extra).

This module demonstrates that the *mechanism* of learning to rank judge rubric
scores works end to end on synthetic data: features are extracted, a small model
is trained with a fixed seed, its loss really decreases, and its calibration is
reported. It ranks **planted synthetic-quality signals, not scientific truth**,
and it never claims benchmark results for any real model or dataset.

The dataset, feature and calibration layers here depend only on NumPy and import
cleanly without torch. The trainable model lives in :mod:`hypoarena.ranker_torch`
and is only imported when the optional extra is present; every torch-dependent
test is marked ``model`` and skips cleanly when torch is absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

import numpy as np

from hypoarena.dedup import TfidfVectorizer
from hypoarena.errors import ValidationError

try:  # pragma: no cover - the import path depends on the installed extras
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def require_torch() -> object:
    """Return the torch module, or raise a clear, actionable error.

    Callers that are about to import :mod:`hypoarena.ranker_torch` use this to
    turn a bare ``ImportError`` into guidance about the optional extra.
    """
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError(
            "the trainable ranker needs the optional torch extra; "
            "install hypoarena[torch] (CPU wheels are sufficient)"
        )
    return torch


@dataclass(frozen=True)
class RankerDatasetConfig:
    """How to generate a deterministic, planted synthetic ranking dataset.

    The label is a pure function of which *signal* tokens a statement contains,
    so a small model can genuinely learn it and the loss decrease is real rather
    than a numerical accident. *Noise* tokens carry no label information.
    """

    n_examples: int = 64
    seed: int = 270106
    signal_tokens: tuple[str, ...] = ("alpha", "beta", "gamma", "delta")
    signal_weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    noise_tokens: tuple[str, ...] = (
        "system",
        "protein",
        "cell",
        "model",
        "data",
        "gene",
    )
    filler_min: int = 3
    filler_max: int = 6

    def __post_init__(self) -> None:
        if self.n_examples < 1:
            raise ValidationError("n_examples must be >= 1", n_examples=self.n_examples)
        if not self.signal_tokens:
            raise ValidationError("signal_tokens must not be empty")
        if len(self.signal_tokens) != len(self.signal_weights):
            raise ValidationError(
                "signal_tokens and signal_weights must have equal length",
                tokens=len(self.signal_tokens),
                weights=len(self.signal_weights),
            )
        if any(weight < 0 for weight in self.signal_weights):
            raise ValidationError("signal weights must be >= 0")
        if sum(self.signal_weights) <= 0:
            raise ValidationError("signal weights must sum to more than zero")
        if not self.noise_tokens:
            raise ValidationError("noise_tokens must not be empty")
        if self.filler_min < 0 or self.filler_max < self.filler_min:
            raise ValidationError(
                "filler range must satisfy 0 <= filler_min <= filler_max",
                filler_min=self.filler_min,
                filler_max=self.filler_max,
            )


@dataclass(frozen=True)
class LabeledExample:
    """One synthetic hypothesis statement with its planted judge score."""

    example_id: str
    text: str
    score: float
    signals: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValidationError("example_id must not be blank")
        if not 0.0 <= self.score <= 1.0:
            raise ValidationError("score must be within [0, 1]", score=self.score)


def planted_quality(signals: Sequence[str], config: RankerDatasetConfig) -> float:
    """Return the weighted fraction of signal tokens present, within ``[0, 1]``.

    This is the ground truth the ranker is asked to recover: it depends only on
    which signal tokens appear, never on the noise tokens.
    """
    total = sum(config.signal_weights)
    if total <= 0:
        return 0.0
    present = set(signals)
    raw = sum(
        weight
        for token, weight in zip(
            config.signal_tokens, config.signal_weights, strict=True
        )
        if token in present
    )
    return raw / total


def synthetic_ranking_dataset(
    config: RankerDatasetConfig,
) -> tuple[LabeledExample, ...]:
    """Generate a deterministic dataset whose labels are planted in the tokens.

    Every signal token is included with probability one half, the label is the
    planted quality of the chosen signals, and the surface text interleaves those
    signals with noise filler. The same seed always yields the same dataset.
    """
    rng = Random(config.seed)
    examples: list[LabeledExample] = []
    for index in range(config.n_examples):
        signals = tuple(token for token in config.signal_tokens if rng.random() < 0.5)
        score = planted_quality(signals, config)
        filler_count = rng.randint(config.filler_min, config.filler_max)
        tokens = list(signals) + [
            rng.choice(config.noise_tokens) for _ in range(filler_count)
        ]
        rng.shuffle(tokens)
        examples.append(
            LabeledExample(
                example_id=f"ex_{index:05d}",
                text=" ".join(tokens),
                score=score,
                signals=signals,
            )
        )
    return tuple(examples)


@dataclass(frozen=True)
class RankerData:
    """A feature matrix and label vector ready for training or inference.

    ``features`` holds one L2-normalized TF-IDF row per example and ``scores``
    the matching planted label in ``[0, 1]``. Shapes are validated so a mismatch
    between the matrix and the labels cannot reach the (torch) trainer.
    """

    features: np.ndarray
    scores: np.ndarray
    vocabulary: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValidationError(
                "features must be a 2-D matrix", ndim=int(self.features.ndim)
            )
        if self.scores.ndim != 1:
            raise ValidationError(
                "scores must be a 1-D vector", ndim=int(self.scores.ndim)
            )
        if self.features.shape[0] != self.scores.shape[0]:
            raise ValidationError(
                "features and scores disagree on the number of rows",
                features=int(self.features.shape[0]),
                scores=int(self.scores.shape[0]),
            )
        if len(self.vocabulary) != self.features.shape[1]:
            raise ValidationError(
                "vocabulary size disagrees with the feature width",
                vocabulary=len(self.vocabulary),
                width=int(self.features.shape[1]),
            )

    def __len__(self) -> int:
        return int(self.features.shape[0])


class RankerFeaturizer:
    """Turns labeled examples into TF-IDF features and a score vector.

    A thin, torch-free wrapper over :class:`hypoarena.dedup.TfidfVectorizer` so
    the feature layer can be tested (and reused) without the optional extra.
    """

    def __init__(self, ngram_size: int = 1, min_document_frequency: int = 1) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_size=ngram_size, min_document_frequency=min_document_frequency
        )

    def fit_transform(self, examples: Sequence[LabeledExample]) -> RankerData:
        """Learn the vocabulary on ``examples`` and return their features."""
        if not examples:
            raise ValidationError("cannot featurize an empty dataset")
        texts = [example.text for example in examples]
        features = self.vectorizer.fit_transform(texts)
        scores = np.array([example.score for example in examples], dtype=np.float64)
        return RankerData(
            features=features, scores=scores, vocabulary=self.vectorizer.vocabulary_
        )

    def transform(self, examples: Sequence[LabeledExample]) -> np.ndarray:
        """Project new examples into the fitted feature space."""
        return self.vectorizer.transform([example.text for example in examples])
