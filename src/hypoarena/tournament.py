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

from dataclasses import dataclass, replace

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


DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_K_FACTOR = 32.0
DEFAULT_K_DECAY = 0.97
DEFAULT_K_FLOOR = 6.0
DEFAULT_RATING_SCALE = 400.0
DEFAULT_DRAW_MARGIN = 0.05
OUTCOMES: tuple[float, ...] = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class EloModel:
    """Bradley–Terry style pairwise rating model.

    ``expectation`` is the logistic win probability implied by two ratings;
    ``update`` moves both ratings by ``K × (score − expectation)``. Draws score
    ``0.5``, and ``draw_margin`` decides when two rubric totals are close enough
    to count as one.
    """

    initial: float = DEFAULT_INITIAL_RATING
    k_factor: float = DEFAULT_K_FACTOR
    k_decay: float = DEFAULT_K_DECAY
    k_floor: float = DEFAULT_K_FLOOR
    scale: float = DEFAULT_RATING_SCALE
    draw_margin: float = DEFAULT_DRAW_MARGIN

    def __post_init__(self) -> None:
        if self.k_factor <= 0:
            raise ValidationError("k_factor must be > 0", k_factor=self.k_factor)
        if not 0.0 < self.k_decay <= 1.0:
            raise ValidationError(
                "k_decay must lie within (0, 1]", k_decay=self.k_decay
            )
        if not 0.0 <= self.k_floor <= self.k_factor:
            raise ValidationError(
                "k_floor must lie within [0, k_factor]",
                k_floor=self.k_floor,
                k_factor=self.k_factor,
            )
        if self.scale <= 0:
            raise ValidationError("scale must be > 0", scale=self.scale)
        if not 0.0 <= self.draw_margin < 1.0:
            raise ValidationError(
                "draw_margin must lie within [0, 1)", draw_margin=self.draw_margin
            )

    def expectation(self, left: float, right: float) -> float:
        """Return the expected score of ``left`` against ``right``."""
        return 1.0 / (1.0 + 10.0 ** ((right - left) / self.scale))

    def k_for(self, played: int) -> float:
        """Return the K factor after ``played`` matches (see :mod:`docs`)."""
        if played < 0:
            raise ValidationError("played must be >= 0", played=played)
        return max(self.k_floor, self.k_factor * self.k_decay**played)

    def outcome(self, left_total: float, right_total: float) -> float:
        """Return 1.0, 0.5 or 0.0 for a left win, a draw or a left loss."""
        difference = left_total - right_total
        if abs(difference) <= self.draw_margin:
            return 0.5
        return 1.0 if difference > 0 else 0.0

    def update(
        self,
        left: float,
        right: float,
        outcome: float,
        *,
        left_played: int = 0,
        right_played: int = 0,
    ) -> tuple[float, float]:
        """Return the updated ratings for both sides of one match."""
        if outcome not in OUTCOMES:
            raise ValidationError(
                "outcome must be 0.0, 0.5 or 1.0",
                outcome=outcome,
                allowed=list(OUTCOMES),
            )
        expected = self.expectation(left, right)
        left_delta = self.k_for(left_played) * (outcome - expected)
        right_delta = self.k_for(right_played) * ((1.0 - outcome) - (1.0 - expected))
        return (left + left_delta, right + right_delta)

    def fingerprint(self) -> str:
        """Return a digest of the model settings for run metadata."""
        return content_hash(
            {
                "initial": self.initial,
                "k_factor": self.k_factor,
                "k_decay": self.k_decay,
                "k_floor": self.k_floor,
                "scale": self.scale,
                "draw_margin": self.draw_margin,
            }
        )


@dataclass(frozen=True)
class Rating:
    """One subject's standing in a tournament.

    The tally invariant ``played == wins + losses + draws`` is enforced at
    construction, so a rating table that has drifted out of sync with the audit
    trail cannot be built in the first place.
    """

    subject: str
    elo: float = DEFAULT_INITIAL_RATING
    played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValidationError("rating subject must not be blank")
        for name in ("played", "wins", "losses", "draws"):
            value = getattr(self, name)
            if value < 0:
                raise ValidationError(f"{name} must be >= 0", **{name: value})
        if self.played != self.wins + self.losses + self.draws:
            raise ValidationError(
                "rating tally disagrees with matches played",
                played=self.played,
                wins=self.wins,
                losses=self.losses,
                draws=self.draws,
            )

    @property
    def win_rate(self) -> float:
        """Wins plus half a point per draw, divided by matches played."""
        if not self.played:
            return 0.0
        return self.score_points / self.played

    @property
    def score_points(self) -> float:
        """Tournament points: one per win, a half per draw."""
        return self.wins + 0.5 * self.draws

    def advanced(self, outcome: float, elo: float) -> Rating:
        """Return the rating after one match played from this side's view."""
        if outcome not in OUTCOMES:
            raise ValidationError(
                "outcome must be 0.0, 0.5 or 1.0",
                outcome=outcome,
                allowed=list(OUTCOMES),
            )
        if outcome == 1.0:
            return replace(self, elo=elo, played=self.played + 1, wins=self.wins + 1)
        if outcome == 0.5:
            return replace(self, elo=elo, played=self.played + 1, draws=self.draws + 1)
        return replace(self, elo=elo, played=self.played + 1, losses=self.losses + 1)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view used by reports and artifacts."""
        return {
            "subject": self.subject,
            "elo": round(self.elo, 4),
            "played": self.played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(self.win_rate, 4),
            "score_points": self.score_points,
        }
