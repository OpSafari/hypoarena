"""Bayesian belief accumulation over hypotheses.

A claim starts at a prior probability; every graded evidence item attached to it
contributes a likelihood ratio; the posterior is the normalized product. Working
in odds space keeps the update multiplicative, which makes two properties easy to
state and test: evidence order does not matter, and same-polarity evidence always
moves the belief the same way.

The numbers are only as good as the likelihood model that produces them. The
default model is a documented, deliberately simple interpolation over evidence
strength — it is a bookkeeping device for a discovery pipeline, not a calibrated
statistical model of any real experimental system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
)
from hypoarena.schema import (
    Evidence,
    EvidencePolarity,
)

MIN_PROBABILITY = 1e-9
MAX_PROBABILITY = 1.0 - 1e-9
DEFAULT_PRIOR = 0.5


def check_probability(value: float, *, name: str = "probability") -> float:
    """Validate a probability and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number", got=type(value).__name__)
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValidationError(f"{name} must lie within [0, 1]", **{name: value})
    return number


def to_odds(probability: float) -> float:
    """Convert a probability in ``[0, 1]`` to odds ``p / (1 - p)``.

    The endpoints are clamped to :data:`MIN_PROBABILITY` and
    :data:`MAX_PROBABILITY` so odds stay finite; a belief of exactly 0 or 1 would
    otherwise be unrevisable, which is never what a discovery pipeline wants.
    """
    clamped = min(MAX_PROBABILITY, max(MIN_PROBABILITY, check_probability(probability)))
    return clamped / (1.0 - clamped)


def from_odds(odds: float) -> float:
    """Convert odds back to a probability."""
    if odds < 0:
        raise ValidationError("odds must be >= 0", odds=odds)
    return odds / (1.0 + odds)


def clamp_probability(probability: float) -> float:
    """Keep a probability strictly inside the open interval ``(0, 1)``."""
    return min(MAX_PROBABILITY, max(MIN_PROBABILITY, probability))


DEFAULT_SUPPORT_RATIO = 3.0
DEFAULT_REFUTE_RATIO = 3.0


@dataclass(frozen=True)
class LikelihoodModel:
    """Turns graded evidence into likelihood ratios.

    A supporting item of strength ``s`` multiplies the odds by
    ``support_ratio ** s``; a refuting item divides by ``refute_ratio ** s``;
    neutral evidence multiplies by ``neutral_ratio`` (1.0 by default, i.e. no
    effect). Strength interpolation is exponential so that half-strength evidence
    moves the belief by the square root of a full-strength step — a documented
    convention, not a measured calibration.
    """

    support_ratio: float = DEFAULT_SUPPORT_RATIO
    refute_ratio: float = DEFAULT_REFUTE_RATIO
    neutral_ratio: float = 1.0

    def __post_init__(self) -> None:
        for name in ("support_ratio", "refute_ratio"):
            value = getattr(self, name)
            if value < 1.0:
                raise ValidationError(f"{name} must be >= 1", **{name: value})
        if self.neutral_ratio <= 0:
            raise ValidationError(
                "neutral_ratio must be > 0", neutral_ratio=self.neutral_ratio
            )

    def ratio_for(self, evidence: Evidence) -> float:
        """Return the likelihood ratio contributed by one evidence item."""
        strength = evidence.strength
        if evidence.polarity is EvidencePolarity.SUPPORT:
            return self.support_ratio**strength
        if evidence.polarity is EvidencePolarity.REFUTE:
            return 1.0 / (self.refute_ratio**strength)
        return self.neutral_ratio

    def fingerprint(self) -> str:
        """Return a digest of the model for run metadata."""
        return content_hash(asdict(self))
