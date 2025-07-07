"""Span-level grounding verification for claims.

A claim is grounded when the corpus text it cites actually says what the claim
says. The checks here are deliberately mechanical and documented: span
resolution, entity overlap, polarity cues and numeric agreement. They catch
fabricated citations, drifted numbers and negation flips; they cannot judge
whether a claim is scientifically true, and the graded flags make that limit
explicit instead of hiding it behind a boolean.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique

from hypoarena.corpus import (
    Corpus,
)
from hypoarena.schema import (
    Citation,
    Claim,
)
from hypoarena.text import (
    content_tokens,
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
