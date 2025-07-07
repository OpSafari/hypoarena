"""Claim-level grading: flag rules, scoring and severity ordering."""

from __future__ import annotations

from hypoarena.corpus import Corpus, Document
from hypoarena.grounding import (
    GroundingFlag,
    GroundingIssue,
    GroundingVerifier,
    citation_score,
    sort_issues,
)
from hypoarena.schema import Citation, Claim, PredictedRelation, Provenance, Scope

DOC = "doc_0123456789ab"
TEXT = "Protein A increases cell growth in HeLa cells. Batch effects were reviewed."
QUOTE = "Protein A increases cell growth"
START = TEXT.index(QUOTE)


def verifier() -> GroundingVerifier:
    return GroundingVerifier(Corpus([Document(DOC, "Findings", TEXT)]))


def claim(citations: tuple[Citation, ...]) -> Claim:
    return Claim(
        claim_id="clm_0123456789ab",
        statement="Protein A increases cell growth",
        subject="protein A",
        object="cell growth",
        relation=PredictedRelation.INCREASES,
        scope=Scope(population="HeLa cells"),
        citations=citations,
        provenance=Provenance(origin="synthetic", seed=1),
    )


def good_citation() -> Citation:
    return Citation(DOC, START, START + len(QUOTE), QUOTE)


def test_a_clean_citation_is_grounded_with_a_perfect_score() -> None:
    report = verifier().verify(claim((good_citation(),)))
    assert report.flag is GroundingFlag.GROUNDED
    assert report.is_grounded is True
    assert report.score == 1.0
    assert report.issues == ()
    assert report.worst_issue is None


def test_an_uncited_claim_is_ungrounded() -> None:
    report = verifier().verify(claim(()))
    assert report.flag is GroundingFlag.UNGROUNDED
    assert report.issues == (GroundingIssue.NO_CITATIONS,)
    assert report.score == 0.0
    assert report.checks == ()


def test_a_fabricated_citation_is_never_grounded() -> None:
    forged = Citation(DOC, START, START + len(QUOTE), "Protein A decreases cell growth")
    report = verifier().verify(claim((forged,)))
    assert report.flag is GroundingFlag.FABRICATED
    assert report.is_fabricated is True
    assert report.is_grounded is False
    assert report.worst_issue is GroundingIssue.QUOTE_MISMATCH


def filler_citation() -> Citation:
    start = TEXT.index("Batch")
    quote = TEXT[start:]
    return Citation(DOC, start, start + len(quote), quote)


def test_one_bad_citation_poisons_an_otherwise_clean_claim() -> None:
    missing = Citation("doc_999999999999", 0, 5, "aaaaa")
    report = verifier().verify(claim((good_citation(), missing)))
    assert report.flag is GroundingFlag.FABRICATED
    assert report.score == 0.5


def test_soft_issues_downgrade_to_weak() -> None:
    report = verifier().verify(claim((filler_citation(),)))
    assert report.flag is GroundingFlag.WEAK
    assert report.is_grounded is False
    assert report.is_fabricated is False
    assert GroundingIssue.LOW_ENTITY_OVERLAP in report.issues


def test_citation_scores_are_bounded() -> None:
    checks = verifier().verify(claim((good_citation(),))).checks
    assert citation_score(checks[0]) == 1.0
    weak = verifier().verify(claim((filler_citation(),))).checks
    assert citation_score(weak[0]) == 0.5


def test_issue_sorting_is_by_severity_and_deduplicated() -> None:
    ordered = sort_issues(
        [
            GroundingIssue.SHORT_QUOTE,
            GroundingIssue.QUOTE_MISMATCH,
            GroundingIssue.SHORT_QUOTE,
            GroundingIssue.POLARITY_CONFLICT,
        ]
    )
    assert ordered == (
        GroundingIssue.QUOTE_MISMATCH,
        GroundingIssue.POLARITY_CONFLICT,
        GroundingIssue.SHORT_QUOTE,
    )


def test_report_dict_shape_is_stable() -> None:
    report = verifier().verify(claim((good_citation(),)))
    assert sorted(report.as_dict()) == [
        "citations",
        "claim_id",
        "flag",
        "issues",
        "resolved",
        "score",
        "statement",
    ]
