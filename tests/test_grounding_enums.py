"""Grounding vocabulary: graded flags, issue classes and their partition."""

from __future__ import annotations

from hypoarena.grounding import (
    FABRICATING_ISSUES,
    SOFT_ISSUES,
    GroundingFlag,
    GroundingIssue,
    is_fabricating,
    is_soft,
)


def test_flags_are_golden() -> None:
    assert [flag.value for flag in GroundingFlag] == [
        "grounded",
        "weakly_grounded",
        "ungrounded",
        "fabricated",
    ]


def test_issues_are_golden() -> None:
    assert [issue.value for issue in GroundingIssue] == [
        "no_citations",
        "missing_document",
        "span_out_of_range",
        "quote_mismatch",
        "short_quote",
        "low_entity_overlap",
        "polarity_conflict",
        "numeric_mismatch",
    ]


def test_fabricating_and_soft_issues_partition_the_vocabulary() -> None:
    assert not (FABRICATING_ISSUES & SOFT_ISSUES)
    classified = FABRICATING_ISSUES | SOFT_ISSUES
    assert set(GroundingIssue) - classified == {GroundingIssue.NO_CITATIONS}


def test_predicates_agree_with_the_partitions() -> None:
    for issue in GroundingIssue:
        assert is_fabricating(issue) is (issue in FABRICATING_ISSUES)
        assert is_soft(issue) is (issue in SOFT_ISSUES)
    assert is_fabricating(GroundingIssue.QUOTE_MISMATCH) is True
    assert is_soft(GroundingIssue.NUMERIC_MISMATCH) is True
