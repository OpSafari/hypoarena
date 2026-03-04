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

from dataclasses import asdict, dataclass, field
from enum import StrEnum, unique

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


@dataclass(frozen=True)
class BeliefState:
    """The accumulated belief in one claim, with the audit counts behind it."""

    claim_id: str
    prior: float
    posterior: float
    likelihood_ratio: float
    supporting: int = 0
    refuting: int = 0
    neutral: int = 0

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValidationError("belief state needs a claim id")
        check_probability(self.prior, name="prior")
        check_probability(self.posterior, name="posterior")
        if self.likelihood_ratio <= 0:
            raise ValidationError(
                "likelihood_ratio must be > 0", likelihood_ratio=self.likelihood_ratio
            )
        for name in ("supporting", "refuting", "neutral"):
            if getattr(self, name) < 0:
                raise ValidationError(
                    f"{name} must be >= 0", **{name: getattr(self, name)}
                )

    @property
    def updates(self) -> int:
        """How many evidence items were folded in."""
        return self.supporting + self.refuting + self.neutral

    @property
    def odds(self) -> float:
        """Posterior odds."""
        return to_odds(self.posterior)

    @property
    def movement(self) -> float:
        """Signed change from prior to posterior."""
        return self.posterior - self.prior

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view for reports and artifacts."""
        return {
            "claim_id": self.claim_id,
            "prior": round(self.prior, 6),
            "posterior": round(self.posterior, 6),
            "likelihood_ratio": round(self.likelihood_ratio, 6),
            "supporting": self.supporting,
            "refuting": self.refuting,
            "neutral": self.neutral,
            "movement": round(self.movement, 6),
        }


def update_belief(prior: float, ratio: float) -> float:
    """Apply one likelihood ratio to a probability and clamp the result."""
    check_probability(prior, name="prior")
    if ratio <= 0:
        raise ValidationError("likelihood ratio must be > 0", ratio=ratio)
    return clamp_probability(from_odds(to_odds(prior) * ratio))


DEFAULT_DOWNWEIGHT_FACTOR = 0.5


@unique
class ContradictionPolicy(StrEnum):
    """What to do when a claim also has refuting evidence attached."""

    IGNORE = "ignore"
    DOWNWEIGHT = "downweight"
    DISCOUNT = "discount"
    REJECT = "reject"


@dataclass(frozen=True)
class BeliefConfig:
    """Prior, likelihood model and contradiction handling for one pass.

    The policies differ in how much they punish a contested claim:

    * ``ignore`` — refuting items contribute nothing (useful as a baseline);
    * ``downweight`` — refuting ratios are raised to ``downweight_factor``, so
      they still push the belief down but less strongly;
    * ``discount`` — once ``contradiction_threshold`` refuting items exist, every
      ratio (supporting included) is raised to ``downweight_factor``, pulling the
      posterior back toward the prior to reflect contested evidence;
    * ``reject`` — once the threshold is reached the posterior is floored, which
      is the right choice when a contradiction invalidates the claim outright.
    """

    prior: float = DEFAULT_PRIOR
    likelihood: LikelihoodModel = field(default_factory=LikelihoodModel)
    contradiction_policy: ContradictionPolicy = ContradictionPolicy.DISCOUNT
    downweight_factor: float = DEFAULT_DOWNWEIGHT_FACTOR
    contradiction_threshold: int = 1

    def __post_init__(self) -> None:
        check_probability(self.prior, name="prior")
        if not 0.0 < self.downweight_factor <= 1.0:
            raise ValidationError(
                "downweight_factor must lie within (0, 1]",
                downweight_factor=self.downweight_factor,
            )
        if self.contradiction_threshold < 1:
            raise ValidationError(
                "contradiction_threshold must be >= 1",
                contradiction_threshold=self.contradiction_threshold,
            )

    def contested(self, refuting: int) -> bool:
        """True when the refuting count reaches the configured threshold."""
        return refuting >= self.contradiction_threshold

    def ratio_for(self, evidence: Evidence, *, refuting: int) -> float:
        """Return the ratio one item contributes under this configuration."""
        ratio = self.likelihood.ratio_for(evidence)
        if evidence.polarity is EvidencePolarity.REFUTE:
            if self.contradiction_policy is ContradictionPolicy.IGNORE:
                return 1.0
            if self.contradiction_policy is ContradictionPolicy.DOWNWEIGHT:
                return ratio**self.downweight_factor
        return ratio

    def fingerprint(self) -> str:
        """Return a digest of the configuration for run metadata."""
        return content_hash(
            {
                "prior": self.prior,
                "likelihood": self.likelihood.fingerprint(),
                "contradiction_policy": self.contradiction_policy.value,
                "downweight_factor": self.downweight_factor,
                "contradiction_threshold": self.contradiction_threshold,
            }
        )
