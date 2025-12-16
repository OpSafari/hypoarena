"""Polarity consistency between a claim and the text it quotes."""

from __future__ import annotations

from helpers import sample_claim
from hypoarena.grounding import GroundingIssue, check_polarity


def test_matching_polarity_passes() -> None:
    claim = sample_claim(statement="Protein A increases cell growth")
    assert check_polarity(claim, "Protein A increases cell growth in HeLa cells") == ()


def test_a_negated_quote_conflicts_with_a_positive_claim() -> None:
    claim = sample_claim(statement="Protein A increases cell growth")
    quote = "We did not observe that protein A increases cell growth."
    assert check_polarity(claim, quote) == (GroundingIssue.POLARITY_CONFLICT,)


def test_a_negated_claim_conflicts_with_a_positive_quote() -> None:
    claim = sample_claim(statement="Protein A does not increase cell growth")
    assert check_polarity(claim, "Protein A increases cell growth") == (
        GroundingIssue.POLARITY_CONFLICT,
    )


def test_both_sides_negated_is_consistent() -> None:
    claim = sample_claim(statement="Protein A does not increase cell growth")
    quote = "No significant effect of protein A on cell growth was detected."
    assert check_polarity(claim, quote) == ()


def test_neutral_quotes_without_cues_pass() -> None:
    claim = sample_claim(statement="Protein A increases cell growth")
    assert check_polarity(claim, "Assays were run in triplicate.") == ()
