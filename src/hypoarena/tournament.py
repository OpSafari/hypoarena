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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import permutations
from random import Random
from typing import Protocol, runtime_checkable

from hypoarena.errors import (
    DuplicateIdError,
    UnknownReferenceError,
    ValidationError,
)
from hypoarena.graph import (
    HypothesisGraph,
)
from hypoarena.grounding import (
    GroundingReport,
)
from hypoarena.ids import (
    content_hash,
)
from hypoarena.schema import (
    Claim,
    ClaimRelation,
    is_directional,
)
from hypoarena.text import (
    content_tokens,
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


@dataclass(frozen=True)
class MatchResult:
    """One judged comparison, kept as a complete audit record.

    Totals are stored alongside the rubric scores so an auditor can recompute the
    outcome without re-running the judge, and ``seed``/indices make the exact
    position of the match inside a run recoverable.
    """

    left: str
    right: str
    left_score: RubricScore
    right_score: RubricScore
    left_total: float
    right_total: float
    outcome: float
    judge: str
    round_index: int = 0
    match_index: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("left", "right", "judge"):
            if not getattr(self, name).strip():
                raise ValidationError(f"match {name} must not be blank")
        if self.left == self.right:
            raise ValidationError(
                "a match needs two different subjects", left=self.left
            )
        if self.outcome not in OUTCOMES:
            raise ValidationError(
                "outcome must be 0.0, 0.5 or 1.0",
                outcome=self.outcome,
                allowed=list(OUTCOMES),
            )
        for name in ("left_total", "right_total"):
            value = getattr(self, name)
            if not SCORE_MINIMUM <= value <= SCORE_MAXIMUM:
                raise ValidationError(f"{name} must lie within [0, 1]", **{name: value})
        for name in ("round_index", "match_index"):
            if getattr(self, name) < 0:
                raise ValidationError(
                    f"{name} must be >= 0", **{name: getattr(self, name)}
                )

    @property
    def winner(self) -> str | None:
        """The winning subject, or ``None`` for a draw."""
        if self.outcome == 0.5:
            return None
        return self.left if self.outcome == 1.0 else self.right

    @property
    def is_draw(self) -> bool:
        """True when neither side won."""
        return self.outcome == 0.5

    @property
    def margin(self) -> float:
        """Signed difference of the weighted totals, from the left side's view."""
        return self.left_total - self.right_total

    def pair(self) -> tuple[str, str]:
        """Return the two subjects in sorted order."""
        return tuple(sorted((self.left, self.right)))  # type: ignore[return-value]

    def total_for(self, subject: str) -> float:
        """Return a subject's weighted total in this match."""
        self._require_subject(subject)
        return self.left_total if subject == self.left else self.right_total

    def outcome_for(self, subject: str) -> float:
        """Return the outcome from one subject's perspective."""
        self._require_subject(subject)
        if self.outcome == 0.5:
            return 0.5
        if subject == self.left:
            return self.outcome
        return 1.0 - self.outcome

    def _require_subject(self, subject: str) -> None:
        if subject not in (self.left, self.right):
            raise UnknownReferenceError(subject, "match subject", owner=self.left)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view of the whole audit record."""
        return {
            "left": self.left,
            "right": self.right,
            "left_score": self.left_score.as_dict(),
            "right_score": self.right_score.as_dict(),
            "left_total": round(self.left_total, 6),
            "right_total": round(self.right_total, 6),
            "outcome": self.outcome,
            "winner": self.winner,
            "judge": self.judge,
            "round_index": self.round_index,
            "match_index": self.match_index,
            "seed": self.seed,
        }


DEFAULT_UNKNOWN_QUALITY = 0.5
MAX_JUDGE_NOISE = 0.4


@runtime_checkable
class Judge(Protocol):
    """Scores one claim on the rubric; may take the opponent into account."""

    @property
    def name(self) -> str:
        """Read-only identifier so frozen dataclass judges conform."""
        ...

    def score(self, claim: Claim, *, opponent: Claim | None = None) -> RubricScore: ...


@dataclass(frozen=True)
class PlantedJudge:
    """Judge driven by a planted quality per claim identifier.

    This is the oracle used by the order-recovery tests: qualities are chosen by
    the test, and the judge adds deterministic per-pair noise so recovery is a
    measured property rather than a tautology. With ``noise=0`` the stronger claim
    always wins, which is the degenerate case used to check the update rule
    itself.
    """

    qualities: Mapping[str, float]
    noise: float = 0.0
    seed: int = 0
    name: str = "planted"

    def __post_init__(self) -> None:
        if not 0.0 <= self.noise <= MAX_JUDGE_NOISE:
            raise ValidationError(
                "judge noise must lie within [0, 0.4]", noise=self.noise
            )
        for subject, quality in self.qualities.items():
            if not 0.0 <= quality <= 1.0:
                raise ValidationError(
                    "planted quality must lie within [0, 1]",
                    subject=subject,
                    quality=quality,
                )

    def quality(self, claim: Claim) -> float:
        """Return the planted quality of a claim (0.5 when it was not planted)."""
        return float(self.qualities.get(claim.claim_id, DEFAULT_UNKNOWN_QUALITY))

    def score(self, claim: Claim, *, opponent: Claim | None = None) -> RubricScore:
        """Return rubric scores scattered deterministically around the quality."""
        rng = Random(
            f"{self.seed}:{claim.claim_id}:{opponent.claim_id if opponent else ''}"
        )
        base = self.quality(claim)
        values = []
        for _dimension in RUBRIC_DIMENSIONS:
            offset = (rng.random() - 0.5) * 2 * self.noise
            values.append(min(SCORE_MAXIMUM, max(SCORE_MINIMUM, base + offset)))
        return RubricScore(*values)


TESTABILITY_BASE = 0.4
TESTABILITY_STEP = 0.2
CONSISTENCY_PENALTY = 0.25
DEFAULT_NOVELTY_SCALE = 12


@dataclass(frozen=True)
class FeatureJudge:
    """Judge that scores inspectable features of a claim.

    Where each dimension comes from:

    * ``grounding`` — the claim's grounding score when a report is known,
      otherwise 1.0 for a cited claim and 0.0 for an uncited one;
    * ``testability`` — 0.4 plus 0.2 for a directional relation, 0.2 for a named
      mechanism and 0.2 for a narrowed scope;
    * ``novelty`` — the statement's content-token count relative to
      ``novelty_scale``, capped at 1.0;
    * ``consistency`` — 1.0 minus 0.25 per contradicting edge, floored at 0.

    Every input is inspectable, so a ranking produced by this judge can be
    explained claim by claim. It is a *preference model over features*, not an
    assessment of scientific truth.
    """

    reports: Mapping[str, GroundingReport] = field(default_factory=dict)
    contradictions: Mapping[str, int] = field(default_factory=dict)
    novelty_scale: int = DEFAULT_NOVELTY_SCALE
    name: str = "features"

    def __post_init__(self) -> None:
        if self.novelty_scale < 1:
            raise ValidationError(
                "novelty_scale must be >= 1", novelty_scale=self.novelty_scale
            )

    def grounding_for(self, claim: Claim) -> float:
        """Return the grounding dimension for one claim."""
        report = self.reports.get(claim.claim_id)
        if report is not None:
            return float(report.score)
        return 1.0 if claim.is_cited else 0.0

    def testability_for(self, claim: Claim) -> float:
        """Return the testability dimension for one claim."""
        total = TESTABILITY_BASE
        if is_directional(claim.relation):
            total += TESTABILITY_STEP
        if claim.mechanism is not None:
            total += TESTABILITY_STEP
        if claim.scope.conditions:
            total += TESTABILITY_STEP
        return min(SCORE_MAXIMUM, total)

    def novelty_for(self, claim: Claim) -> float:
        """Return the novelty dimension for one claim."""
        tokens = len(content_tokens(claim.statement))
        return min(SCORE_MAXIMUM, tokens / self.novelty_scale)

    def consistency_for(self, claim: Claim) -> float:
        """Return the consistency dimension for one claim."""
        conflicts = self.contradictions.get(claim.claim_id, 0)
        return max(SCORE_MINIMUM, 1.0 - CONSISTENCY_PENALTY * conflicts)

    def score(self, claim: Claim, *, opponent: Claim | None = None) -> RubricScore:
        """Score a claim on all four rubric dimensions."""
        return RubricScore(
            novelty=self.novelty_for(claim),
            testability=self.testability_for(claim),
            grounding=self.grounding_for(claim),
            consistency=self.consistency_for(claim),
        )


def graph_contradiction_counts(graph: HypothesisGraph) -> dict[str, int]:
    """Count the ``contradicts`` edges touching each claim in a graph."""
    counts: dict[str, int] = {}
    for edge in graph.edges:
        if edge.relation is ClaimRelation.CONTRADICTS:
            counts[edge.source] = counts.get(edge.source, 0) + 1
            counts[edge.target] = counts.get(edge.target, 0) + 1
    return counts


def round_robin_pairings(
    subjects: Sequence[str], *, repeats: int = 1, seed: int = 0
) -> tuple[tuple[str, str], ...]:
    """Return a full round-robin schedule, repeated and shuffled deterministically.

    Each pass over the pair list is one round; within a round the order is
    shuffled with a generator seeded by ``(seed, round)`` so a run is
    reproducible but not biased by identifier order. On odd-numbered repeats the
    sides are swapped, which cancels any positional advantage a judge might have.
    """
    unique = sorted(set(subjects))
    if len(unique) < 2:
        raise ValidationError(
            "a pairing schedule needs at least two subjects", count=len(unique)
        )
    if repeats < 1:
        raise ValidationError("repeats must be >= 1", repeats=repeats)
    pairs = [
        (left, right)
        for index, left in enumerate(unique)
        for right in unique[index + 1 :]
    ]
    schedule: list[tuple[str, str]] = []
    for round_index in range(repeats):
        rng = Random(f"hypoarena:tournament:{seed}:{round_index}")
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        if round_index % 2 == 1:
            shuffled = [(right, left) for left, right in shuffled]
        schedule.extend(shuffled)
    return tuple(schedule)


def pair_count(subjects: Sequence[str]) -> int:
    """Return how many matches one round contains."""
    size = len(set(subjects))
    return size * (size - 1) // 2


@dataclass(frozen=True)
class TournamentConfig:
    """Seed, schedule size and the models a tournament runs with."""

    seed: int = 0
    repeats: int = 1
    model: EloModel = field(default_factory=EloModel)
    weights: RubricWeights = field(default_factory=RubricWeights)

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValidationError("repeats must be >= 1", repeats=self.repeats)

    def fingerprint(self) -> str:
        """Return a digest covering every setting that affects the outcome."""
        return content_hash(
            {
                "seed": self.seed,
                "repeats": self.repeats,
                "model": self.model.fingerprint(),
                "weights": self.weights.fingerprint(),
            }
        )


@dataclass(frozen=True)
class TournamentResult:
    """Standings plus the complete match audit trail of one tournament."""

    subjects: tuple[str, ...]
    ratings: tuple[Rating, ...]
    matches: tuple[MatchResult, ...]
    config: TournamentConfig

    def ranking(self) -> tuple[str, ...]:
        """Return subject identifiers from best to worst rated."""
        return tuple(rating.subject for rating in self.ratings)

    def rating(self, subject: str) -> Rating:
        """Return one subject's rating."""
        for entry in self.ratings:
            if entry.subject == subject:
                return entry
        raise UnknownReferenceError(subject, "tournament subject")

    def standings(self) -> tuple[dict[str, object], ...]:
        """Return ranked rows ready for a report table."""
        return tuple(
            {"position": position, **rating.as_dict()}
            for position, rating in enumerate(self.ratings, start=1)
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready summary (matches are serialized separately)."""
        return {
            "subjects": list(self.subjects),
            "matches": len(self.matches),
            "seed": self.config.seed,
            "repeats": self.config.repeats,
            "config_fingerprint": self.config.fingerprint(),
            "standings": list(self.standings()),
        }

    def signature(self) -> str:
        """Return a digest over standings and the audit trail."""
        return content_hash(
            {
                "standings": list(self.standings()),
                "matches": [match.as_dict() for match in self.matches],
                "config": self.config.fingerprint(),
            }
        )


class Tournament:
    """Runs a judged round-robin and maintains the ratings table."""

    def __init__(self, judge: Judge, config: TournamentConfig | None = None) -> None:
        if not isinstance(judge, Judge):
            raise ValidationError(
                "judge must satisfy the Judge protocol", got=type(judge).__name__
            )
        self.judge = judge
        self.config = config or TournamentConfig()

    def run(self, claims: Sequence[Claim]) -> TournamentResult:
        """Judge every scheduled pairing and return the final standings.

        Claims are matched by identifier, so the same claim list always produces
        the same schedule; ratings start at the model's initial value and are
        updated after each match using both sides' experience (matches played).
        """
        by_id: dict[str, Claim] = {}
        for claim in claims:
            if claim.claim_id in by_id:
                raise DuplicateIdError(claim.claim_id, "claim")
            by_id[claim.claim_id] = claim
        if len(by_id) < 2:
            raise ValidationError(
                "a tournament needs at least two distinct claims", count=len(by_id)
            )
        subjects = tuple(sorted(by_id))
        model = self.config.model
        ratings = {subject: Rating(subject, model.initial) for subject in subjects}
        matches: list[MatchResult] = []
        per_round = pair_count(subjects)
        schedule = round_robin_pairings(
            subjects, repeats=self.config.repeats, seed=self.config.seed
        )
        for index, (left_id, right_id) in enumerate(schedule):
            left_claim = by_id[left_id]
            right_claim = by_id[right_id]
            left_score = self.judge.score(left_claim, opponent=right_claim)
            right_score = self.judge.score(right_claim, opponent=left_claim)
            left_total = weighted_total(left_score, self.config.weights)
            right_total = weighted_total(right_score, self.config.weights)
            outcome = model.outcome(left_total, right_total)
            left_rating = ratings[left_id]
            right_rating = ratings[right_id]
            left_elo, right_elo = model.update(
                left_rating.elo,
                right_rating.elo,
                outcome,
                left_played=left_rating.played,
                right_played=right_rating.played,
            )
            ratings[left_id] = left_rating.advanced(outcome, left_elo)
            ratings[right_id] = right_rating.advanced(1.0 - outcome, right_elo)
            matches.append(
                MatchResult(
                    left=left_id,
                    right=right_id,
                    left_score=left_score,
                    right_score=right_score,
                    left_total=left_total,
                    right_total=right_total,
                    outcome=outcome,
                    judge=self.judge.name,
                    round_index=index // per_round,
                    match_index=index,
                    seed=self.config.seed,
                )
            )
        ordered = tuple(
            sorted(ratings.values(), key=lambda rating: (-rating.elo, rating.subject))
        )
        return TournamentResult(
            subjects=subjects,
            ratings=ordered,
            matches=tuple(matches),
            config=self.config,
        )


def kendall_tau(expected: Sequence[str], actual: Sequence[str]) -> float:
    """Return Kendall's tau between two orderings of the same subject set.

    ``1.0`` means identical order, ``-1.0`` the reverse and ``0.0`` no agreement.
    Both sequences must contain exactly the same subjects, otherwise the
    comparison is meaningless and a validation error is raised.
    """
    if sorted(expected) != sorted(actual):
        raise ValidationError(
            "orderings must cover the same subjects",
            expected=len(expected),
            actual=len(actual),
        )
    if len(expected) < 2:
        return 1.0
    positions = {subject: index for index, subject in enumerate(expected)}
    concordant = 0
    discordant = 0
    for index, first in enumerate(actual):
        for second in actual[index + 1 :]:
            if positions[first] < positions[second]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def order_recovery(expected: Sequence[str], actual: Sequence[str]) -> float:
    """Return Kendall's tau rescaled to ``[0, 1]`` for reporting."""
    return (kendall_tau(expected, actual) + 1.0) / 2.0


def prefix_agreement(expected: Sequence[str], actual: Sequence[str], top: int) -> float:
    """Return the overlap of the two top-``top`` sets, within ``[0, 1]``."""
    if top < 1:
        raise ValidationError("top must be >= 1", top=top)
    size = min(top, len(expected), len(actual))
    if size == 0:
        return 1.0
    shared = set(expected[:size]) & set(actual[:size])
    return len(shared) / size


def transitivity_rate(matches: Sequence[MatchResult]) -> float:
    """Return the fraction of decided chains that are transitive.

    For every ordered triple where ``a`` beat ``b`` and ``b`` beat ``c`` and the
    ``a``/``c`` pair was also decided, the chain counts as transitive when ``a``
    beat ``c``. Draws are ignored, and a tournament without any decided chain
    scores ``1.0``.
    """
    beats: dict[str, set[str]] = {}
    decided: set[frozenset[str]] = set()
    for match in matches:
        if match.winner is None:
            continue
        loser = match.right if match.winner == match.left else match.left
        beats.setdefault(match.winner, set()).add(loser)
        decided.add(frozenset(match.pair()))
    subjects = sorted({subject for pair in decided for subject in pair})
    consistent = 0
    total = 0
    for first, second, third in permutations(subjects, 3):
        if second not in beats.get(first, set()):
            continue
        if third not in beats.get(second, set()):
            continue
        if frozenset((first, third)) not in decided:
            continue
        total += 1
        if third in beats.get(first, set()):
            consistent += 1
    return consistent / total if total else 1.0


def replay_ratings(
    matches: Sequence[MatchResult], model: EloModel | None = None
) -> tuple[Rating, ...]:
    """Recompute final ratings from an audit trail alone.

    A tournament's match list is a complete record: replaying it must reproduce
    the published standings exactly, which is the property that makes the audit
    trail worth storing.
    """
    active = model or EloModel()
    ratings: dict[str, Rating] = {}
    for match in matches:
        for subject in match.pair():
            ratings.setdefault(subject, Rating(subject, active.initial))
        left = ratings[match.left]
        right = ratings[match.right]
        left_elo, right_elo = active.update(
            left.elo,
            right.elo,
            match.outcome,
            left_played=left.played,
            right_played=right.played,
        )
        ratings[match.left] = left.advanced(match.outcome, left_elo)
        ratings[match.right] = right.advanced(match.outcome_for(match.right), right_elo)
    return tuple(
        sorted(ratings.values(), key=lambda rating: (-rating.elo, rating.subject))
    )


def step_sizes(
    matches: Sequence[MatchResult], model: EloModel | None = None
) -> tuple[float, ...]:
    """Return the mean absolute rating movement caused by each match."""
    active = model or EloModel()
    ratings: dict[str, Rating] = {}
    sizes: list[float] = []
    for match in matches:
        for subject in match.pair():
            ratings.setdefault(subject, Rating(subject, active.initial))
        left = ratings[match.left]
        right = ratings[match.right]
        left_elo, right_elo = active.update(
            left.elo,
            right.elo,
            match.outcome,
            left_played=left.played,
            right_played=right.played,
        )
        sizes.append((abs(left_elo - left.elo) + abs(right_elo - right.elo)) / 2)
        ratings[match.left] = left.advanced(match.outcome, left_elo)
        ratings[match.right] = right.advanced(match.outcome_for(match.right), right_elo)
    return tuple(sizes)


def convergence_ratio(
    matches: Sequence[MatchResult], model: EloModel | None = None
) -> float:
    """Return late-match movement divided by early-match movement.

    Values below one mean the tournament is settling: with a decaying K factor
    the second half of the schedule moves ratings less than the first half. At
    least four matches are needed for the comparison to mean anything.
    """
    sizes = step_sizes(matches, model)
    if len(sizes) < 4:
        raise ValidationError(
            "convergence needs at least four matches", count=len(sizes)
        )
    half = len(sizes) // 2
    early = sum(sizes[:half]) / half
    late = sum(sizes[half:]) / (len(sizes) - half)
    return late / early if early else 0.0


def rating_spread(result: TournamentResult) -> float:
    """Return the gap between the best and worst rating."""
    if not result.ratings:
        return 0.0
    return result.ratings[0].elo - result.ratings[-1].elo
