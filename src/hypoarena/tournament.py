"""Pairwise tournaments over rubric-scored hypotheses.

Ranking is the mechanism that turns many plausible hypotheses into an ordered
shortlist. This module implements it the way comparative-judgement systems do:
a judge scores both entries on a fixed rubric, the scores decide a win, loss or
draw, and an Elo/Bradley–Terry update moves the ratings. Everything is
deterministic given a seed, and every match is kept in an audit trail.

The ratings describe *relative preference under the configured judge*. They are
not a measure of scientific truth, and the tests that check order recovery use
planted qualities on synthetic claims for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
)

RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "novelty",
    "testability",
    "grounding",
    "consistency",
)
SCORE_MINIMUM = 0.0
SCORE_MAXIMUM = 1.0


@dataclass(frozen=True)
class RubricScore:
    """A judge's scores for one entry, each within ``[0, 1]``."""

    novelty: float
    testability: float
    grounding: float
    consistency: float

    def __post_init__(self) -> None:
        for dimension in RUBRIC_DIMENSIONS:
            value = getattr(self, dimension)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(
                    f"rubric {dimension} must be a number",
                    dimension=dimension,
                    got=type(value).__name__,
                )
            if not SCORE_MINIMUM <= float(value) <= SCORE_MAXIMUM:
                raise ValidationError(
                    f"rubric {dimension} must lie within [0, 1]",
                    dimension=dimension,
                    value=value,
                )

    def dimension(self, name: str) -> float:
        """Return one dimension's score by name."""
        if name not in RUBRIC_DIMENSIONS:
            raise ValidationError(
                "unknown rubric dimension",
                dimension=name,
                allowed=list(RUBRIC_DIMENSIONS),
            )
        return float(getattr(self, name))

    def as_tuple(self) -> tuple[float, ...]:
        """Return the scores in the canonical dimension order."""
        return tuple(self.dimension(name) for name in RUBRIC_DIMENSIONS)

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-ready view keyed by dimension."""
        return {name: self.dimension(name) for name in RUBRIC_DIMENSIONS}

    def mean(self) -> float:
        """Return the unweighted mean of the four dimensions."""
        return sum(self.as_tuple()) / len(RUBRIC_DIMENSIONS)


@dataclass(frozen=True)
class RubricWeights:
    """How much each rubric dimension contributes to a match verdict.

    Weights only have to be non-negative with a positive sum; :meth:`normalized`
    rescales them to sum to one so a weighted total is always within ``[0, 1]``
    and two configurations that differ only by scale behave identically.
    """

    novelty: float = 0.25
    testability: float = 0.30
    grounding: float = 0.30
    consistency: float = 0.15

    def __post_init__(self) -> None:
        for dimension in RUBRIC_DIMENSIONS:
            value = getattr(self, dimension)
            if value < 0:
                raise ValidationError(
                    f"rubric weight for {dimension} must be >= 0",
                    dimension=dimension,
                    value=value,
                )
        if self.total() <= 0:
            raise ValidationError("rubric weights must sum to more than zero")

    def weight(self, name: str) -> float:
        """Return one dimension's weight by name."""
        if name not in RUBRIC_DIMENSIONS:
            raise ValidationError(
                "unknown rubric dimension",
                dimension=name,
                allowed=list(RUBRIC_DIMENSIONS),
            )
        return float(getattr(self, name))

    def total(self) -> float:
        """Return the raw weight sum."""
        return sum(self.weight(name) for name in RUBRIC_DIMENSIONS)

    def normalized(self) -> RubricWeights:
        """Return weights rescaled to sum to one."""
        total = self.total()
        return RubricWeights(
            **{name: self.weight(name) / total for name in RUBRIC_DIMENSIONS}
        )

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-ready view keyed by dimension."""
        return {name: self.weight(name) for name in RUBRIC_DIMENSIONS}

    def fingerprint(self) -> str:
        """Return a digest of the normalized weights."""
        return content_hash(self.normalized().as_dict())


def weighted_total(score: RubricScore, weights: RubricWeights) -> float:
    """Return the weighted rubric total in ``[0, 1]``."""
    normalized = weights.normalized()
    return sum(
        score.dimension(name) * normalized.weight(name) for name in RUBRIC_DIMENSIONS
    )
