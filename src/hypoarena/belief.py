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

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum, unique

from hypoarena.errors import (
    UnknownReferenceError,
    ValidationError,
)
from hypoarena.graph import (
    HypothesisGraph,
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


def accumulate(
    claim_id: str,
    evidence: Sequence[Evidence],
    config: BeliefConfig | None = None,
) -> BeliefState:
    """Fold a set of evidence items into one belief state.

    Ratios multiply, so the order of ``evidence`` cannot change the posterior —
    a property the test suite checks explicitly. The ``discount`` policy needs
    the refuting count up front, so items are counted before they are applied.
    """
    settings = config or BeliefConfig()
    supporting = sum(
        1 for item in evidence if item.polarity is EvidencePolarity.SUPPORT
    )
    refuting = sum(1 for item in evidence if item.polarity is EvidencePolarity.REFUTE)
    neutral = len(evidence) - supporting - refuting
    ratio = 1.0
    for item in evidence:
        ratio *= settings.ratio_for(item, refuting=refuting)
    contested = settings.contested(refuting)
    if contested and settings.contradiction_policy is ContradictionPolicy.DISCOUNT:
        ratio **= settings.downweight_factor
    posterior = update_belief(settings.prior, ratio)
    if contested and settings.contradiction_policy is ContradictionPolicy.REJECT:
        posterior = MIN_PROBABILITY
    return BeliefState(
        claim_id=claim_id,
        prior=settings.prior,
        posterior=posterior,
        likelihood_ratio=ratio,
        supporting=supporting,
        refuting=refuting,
        neutral=neutral,
    )


DEFAULT_PRIOR_GRID: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def prior_sensitivity(
    evidence: Sequence[Evidence],
    priors: Sequence[float] = DEFAULT_PRIOR_GRID,
    config: BeliefConfig | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return ``(prior, posterior)`` pairs over a grid of priors.

    Sensitivity analysis answers the question a reviewer always asks: does the
    conclusion survive a different starting assumption? The evidence set and every
    other setting are held fixed, so the spread of the returned posteriors is
    attributable to the prior alone.
    """
    if not priors:
        raise ValidationError("a prior grid must not be empty")
    settings = config or BeliefConfig()
    pairs = []
    for prior in priors:
        adjusted = BeliefConfig(
            prior=prior,
            likelihood=settings.likelihood,
            contradiction_policy=settings.contradiction_policy,
            downweight_factor=settings.downweight_factor,
            contradiction_threshold=settings.contradiction_threshold,
        )
        pairs.append((prior, accumulate("sensitivity", evidence, adjusted).posterior))
    return tuple(pairs)


def sensitivity_spread(
    evidence: Sequence[Evidence],
    priors: Sequence[float] = DEFAULT_PRIOR_GRID,
    config: BeliefConfig | None = None,
) -> float:
    """Return the widest posterior gap the prior grid can produce."""
    posteriors = [
        posterior for _, posterior in prior_sensitivity(evidence, priors, config)
    ]
    return max(posteriors) - min(posteriors)


def is_prior_robust(
    evidence: Sequence[Evidence],
    *,
    threshold: float = 0.5,
    priors: Sequence[float] = DEFAULT_PRIOR_GRID,
    config: BeliefConfig | None = None,
) -> bool:
    """True when the posterior stays on one side of ``threshold`` for every prior."""
    posteriors = [
        posterior for _, posterior in prior_sensitivity(evidence, priors, config)
    ]
    return all(item > threshold for item in posteriors) or all(
        item < threshold for item in posteriors
    )


def accumulate_graph(
    graph: HypothesisGraph, config: BeliefConfig | None = None
) -> tuple[BeliefState, ...]:
    """Accumulate beliefs for every claim in a graph.

    Evidence is read in identifier order, so the result never depends on
    insertion order. Claims without evidence keep the prior, which keeps them
    visible in reports instead of silently disappearing.
    """
    states = []
    for claim in graph.claims:
        evidence = sorted(graph.evidence_for(claim.claim_id), key=item_key)
        states.append(accumulate(claim.claim_id, evidence, config))
    return tuple(states)


def item_key(evidence: Evidence) -> str:
    """Return the deterministic sort key for one evidence item."""
    return evidence.evidence_id


def rank_beliefs(states: Sequence[BeliefState]) -> tuple[str, ...]:
    """Return claim identifiers from the most to the least supported.

    Ties are broken by identifier so the ranking is total and reproducible.
    """
    return tuple(
        state.claim_id
        for state in sorted(
            states, key=lambda state: (-state.posterior, state.claim_id)
        )
    )


def belief_table(states: Sequence[BeliefState]) -> tuple[dict[str, object], ...]:
    """Return ranked rows ready for a report table."""
    return tuple(
        {"position": position, **state.as_dict()}
        for position, state in enumerate(
            sorted(states, key=lambda state: (-state.posterior, state.claim_id)),
            start=1,
        )
    )


def state_for(states: Sequence[BeliefState], claim_id: str) -> BeliefState:
    """Return one claim's belief state."""
    for state in states:
        if state.claim_id == claim_id:
            return state
    raise UnknownReferenceError(claim_id, "belief state")
