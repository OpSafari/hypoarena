"""Span-level grounding verification for claims.

A claim is grounded when the corpus text it cites actually says what the claim
says. The checks here are deliberately mechanical and documented: span
resolution, entity overlap, polarity cues and numeric agreement. They catch
fabricated citations, drifted numbers and negation flips; they cannot judge
whether a claim is scientifically true, and the graded flags make that limit
explicit instead of hiding it behind a boolean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum, unique

from hypoarena.corpus import (
    Corpus,
)
from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
)
from hypoarena.schema import (
    Citation,
    Claim,
)
from hypoarena.text import (
    content_tokens,
    extract_numbers,
    jaccard,
    negation_flip,
)


@unique
class GroundingFlag(StrEnum):
    """Graded outcome of verifying one claim against a corpus."""

    GROUNDED = "grounded"
    WEAK = "weakly_grounded"
    UNGROUNDED = "ungrounded"
    FABRICATED = "fabricated"


@unique
class GroundingIssue(StrEnum):
    """Individual problems found while checking a claim or citation."""

    NO_CITATIONS = "no_citations"
    MISSING_DOCUMENT = "missing_document"
    SPAN_OUT_OF_RANGE = "span_out_of_range"
    QUOTE_MISMATCH = "quote_mismatch"
    SHORT_QUOTE = "short_quote"
    LOW_ENTITY_OVERLAP = "low_entity_overlap"
    POLARITY_CONFLICT = "polarity_conflict"
    NUMERIC_MISMATCH = "numeric_mismatch"


#: Issues that mean the citation does not point at the text it claims to quote.
FABRICATING_ISSUES: frozenset[GroundingIssue] = frozenset(
    {
        GroundingIssue.MISSING_DOCUMENT,
        GroundingIssue.SPAN_OUT_OF_RANGE,
        GroundingIssue.QUOTE_MISMATCH,
    }
)
#: Issues that weaken but do not invalidate a citation.
SOFT_ISSUES: frozenset[GroundingIssue] = frozenset(
    {
        GroundingIssue.SHORT_QUOTE,
        GroundingIssue.LOW_ENTITY_OVERLAP,
        GroundingIssue.POLARITY_CONFLICT,
        GroundingIssue.NUMERIC_MISMATCH,
    }
)


def is_fabricating(issue: GroundingIssue) -> bool:
    """True when an issue means the quoted text is not where the claim says."""
    return issue in FABRICATING_ISSUES


def is_soft(issue: GroundingIssue) -> bool:
    """True when an issue downgrades a citation without invalidating it."""
    return issue in SOFT_ISSUES


@dataclass(frozen=True)
class CitationCheck:
    """The outcome of checking one citation of one claim."""

    citation: Citation
    resolved: bool
    issues: tuple[GroundingIssue, ...]
    entity_overlap: float
    claimed_numbers: tuple[float, ...]
    quoted_numbers: tuple[float, ...]
    detail: str | None = None

    @property
    def is_fabricated(self) -> bool:
        """True when the cited text is not where the citation says it is."""
        return any(is_fabricating(issue) for issue in self.issues)

    @property
    def is_clean(self) -> bool:
        """True when no issue at all was raised."""
        return not self.issues

    def span(self) -> tuple[str, int, int]:
        """Return the citation's document and offsets as a plain tuple."""
        return (self.citation.document_id, self.citation.start, self.citation.end)


def claim_terms(claim: Claim) -> set[str]:
    """Return the content tokens a claim asserts, from its variables."""
    return set(content_tokens(f"{claim.subject} {claim.object}"))


def entity_overlap(claim: Claim, quote: str) -> float:
    """Return the Jaccard overlap between a claim's variables and a quote.

    Only the claim's subject and object contribute: the statement itself usually
    repeats them plus a verb, and including it would reward verbose claims.
    """
    return jaccard(claim_terms(claim), set(content_tokens(quote)))


def check_span(
    corpus: Corpus, citation: Citation
) -> tuple[bool, tuple[GroundingIssue, ...], str | None]:
    """Resolve one citation against the corpus.

    Returns ``(resolved, issues, detail)``. The three failure modes are kept
    apart because they mean different things: a missing document is a broken
    reference, out-of-range offsets are a stale span, and a quote mismatch is the
    fabricated-citation case the verifier must never let through.
    """
    if not corpus.has_document(citation.document_id):
        return (
            False,
            (GroundingIssue.MISSING_DOCUMENT,),
            f"corpus has no document {citation.document_id}",
        )
    document = corpus.document(citation.document_id)
    if citation.start < 0 or citation.end > document.length:
        return (
            False,
            (GroundingIssue.SPAN_OUT_OF_RANGE,),
            f"offsets {citation.start}..{citation.end} outside 0..{document.length}",
        )
    found = document.text[citation.start : citation.end]
    if found != citation.quote:
        return (
            False,
            (GroundingIssue.QUOTE_MISMATCH,),
            f"document says {found!r}, claim quotes {citation.quote!r}",
        )
    return True, (), None


def check_polarity(claim: Claim, quote: str) -> tuple[GroundingIssue, ...]:
    """Flag a negation cue that appears on one side only.

    The check is symmetric on purpose: a claim asserting an effect against a
    quote that denies it, and a negated claim citing a positive finding, are both
    polarity conflicts. It is a cue-level heuristic — it detects *that* a
    statement is negated, not which part of it is.
    """
    if negation_flip(claim.statement, quote):
        return (GroundingIssue.POLARITY_CONFLICT,)
    return ()


def claimed_numbers(claim: Claim) -> tuple[float, ...]:
    """Return every number the claim asserts, in statement order."""
    return tuple(extract_numbers(f"{claim.statement} {claim.subject} {claim.object}"))


def check_numbers(
    claim: Claim, quote: str, *, tolerance: float = 0.0
) -> tuple[GroundingIssue, ...]:
    """Require every number the claim asserts to appear in the quoted text.

    Only magnitudes are compared (see :func:`hypoarena.text.extract_numbers` for
    the documented limits: units are ignored, dotted versions split). A claim
    without numbers passes vacuously; a claim with numbers the quote does not
    contain is flagged, which is what catches a drifted effect size.
    """
    claimed = claimed_numbers(claim)
    if not claimed:
        return ()
    quoted = extract_numbers(quote)
    missing = [
        value
        for value in claimed
        if not any(abs(value - item) <= tolerance for item in quoted)
    ]
    return (GroundingIssue.NUMERIC_MISMATCH,) if missing else ()


DEFAULT_MIN_ENTITY_OVERLAP = 0.34
DEFAULT_MIN_QUOTE_LENGTH = 12
DEFAULT_NUMERIC_TOLERANCE = 1e-9


@dataclass(frozen=True)
class VerifierConfig:
    """Thresholds for one grounding verification pass.

    ``min_entity_overlap`` is the Jaccard score a quote must reach against the
    claim's variables; ``min_quote_length`` rejects citations so short that any
    overlap is accidental. Both checks only ever downgrade a citation to
    ``weakly_grounded`` — they never turn a resolvable citation into a
    fabricated one.
    """

    min_entity_overlap: float = DEFAULT_MIN_ENTITY_OVERLAP
    min_quote_length: int = DEFAULT_MIN_QUOTE_LENGTH
    numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE
    polarity_checks: bool = True
    numeric_checks: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_entity_overlap <= 1.0:
            raise ValidationError(
                "min_entity_overlap must lie within [0, 1]",
                min_entity_overlap=self.min_entity_overlap,
            )
        if self.min_quote_length < 1:
            raise ValidationError(
                "min_quote_length must be >= 1", min_quote_length=self.min_quote_length
            )
        if self.numeric_tolerance < 0.0:
            raise ValidationError(
                "numeric_tolerance must be >= 0",
                numeric_tolerance=self.numeric_tolerance,
            )

    def fingerprint(self) -> str:
        """Return a digest of the configuration for run metadata."""
        return content_hash(asdict(self))
