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
