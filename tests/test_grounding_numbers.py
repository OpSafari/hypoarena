"""Numeric agreement: matching values pass, drifted values are flagged."""

from __future__ import annotations

from helpers import sample_claim
from hypoarena.grounding import GroundingIssue, check_numbers, claimed_numbers


def test_claims_without_numbers_pass_vacuously() -> None:
    claim = sample_claim(statement="Protein A increases cell growth")
    assert claimed_numbers(claim) == ()
    assert check_numbers(claim, "Protein A increases cell growth") == ()


def test_matching_numbers_pass() -> None:
    claim = sample_claim(statement="Protein A increases cell growth by 1.5 fold")
    assert claimed_numbers(claim) == (1.5,)
    assert check_numbers(claim, "growth increased 1.5 fold in HeLa cells") == ()


def test_a_drifted_number_is_flagged() -> None:
    claim = sample_claim(statement="Protein A increases cell growth by 1.5 fold")
    assert check_numbers(claim, "growth increased 2.5 fold in HeLa cells") == (
        GroundingIssue.NUMERIC_MISMATCH,
    )


def test_a_missing_number_is_flagged() -> None:
    claim = sample_claim(statement="Protein A increases cell growth by 1.5 fold")
    assert check_numbers(claim, "growth increased in HeLa cells") == (
        GroundingIssue.NUMERIC_MISMATCH,
    )


def test_tolerance_controls_the_comparison() -> None:
    claim = sample_claim(statement="effect size 1.50")
    assert check_numbers(claim, "effect size 1.52", tolerance=0.05) == ()
    assert check_numbers(claim, "effect size 1.52", tolerance=0.01) == (
        GroundingIssue.NUMERIC_MISMATCH,
    )


def test_numbers_glued_to_variable_names_are_ignored() -> None:
    claim = sample_claim(subject="IL6", object="p53 levels", statement="no change")
    assert claimed_numbers(claim) == ()


def test_numbers_inside_variables_are_collected() -> None:
    claim = sample_claim(subject="dose 1.5", object="response 2", statement="scaled")
    assert claimed_numbers(claim) == (1.5, 2.0)
