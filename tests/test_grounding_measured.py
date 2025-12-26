"""Measured grounding rates on synthetic corpora with planted truth.

The numbers asserted here were produced by running this suite against the
bundled generator; they are properties of the synthetic corpora, not of any
language model. Configuration: ``SyntheticConfig(seed=S, chains=2,
chain_length=3)`` — 16 documents, 6 gold claims (4 planted links plus 2 rivals)
and 12 evidence items per seed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hypoarena.grounding import GroundingFlag, GroundingVerifier, summarize_reports
from hypoarena.synthetic import SyntheticBundle, SyntheticConfig, build_bundle

SEEDS = (131, 977, 270106)


def bundle(seed: int) -> SyntheticBundle:
    return build_bundle(SyntheticConfig(seed=seed, chains=2, chain_length=3))


def verifier_for(produced: SyntheticBundle) -> GroundingVerifier:
    return GroundingVerifier(produced.corpus)


@pytest.mark.parametrize("seed", SEEDS)
def test_gold_claims_are_fully_grounded(seed: int) -> None:
    produced = bundle(seed)
    summary = summarize_reports(verifier_for(produced).verify_claims(produced.claims))
    assert summary.as_dict() == {
        "total": 6,
        "grounded": 6,
        "weakly_grounded": 0,
        "ungrounded": 0,
        "fabricated": 0,
        "grounded_rate": 1.0,
        "mean_score": 1.0,
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_shifted_spans_are_all_fabricated(seed: int) -> None:
    produced = bundle(seed)
    shifted = [
        replace(
            claim,
            citations=tuple(
                replace(item, start=item.start + 1, end=item.end + 1)
                for item in claim.citations
            ),
        )
        for claim in produced.claims
    ]
    summary = summarize_reports(verifier_for(produced).verify_claims(shifted))
    assert summary.fabricated == summary.total == 6
    assert summary.grounded_rate == 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_invented_quotes_are_all_fabricated(seed: int) -> None:
    produced = bundle(seed)
    forged = [
        replace(
            claim,
            citations=tuple(
                replace(item, quote="invented finding text here")
                for item in claim.citations
            ),
        )
        for claim in produced.claims
    ]
    summary = summarize_reports(verifier_for(produced).verify_claims(forged))
    assert summary.fabricated == 6
    assert summary.mean_score == 0.0


def test_uncited_claims_are_all_ungrounded() -> None:
    produced = bundle(131)
    uncited = [replace(claim, citations=()) for claim in produced.claims]
    summary = summarize_reports(verifier_for(produced).verify_claims(uncited))
    assert summary.ungrounded == 6
    assert summary.grounded == 0
    assert summary.grounded_rate == 0.0


def test_flags_cover_every_report_exactly_once() -> None:
    produced = bundle(131)
    reports = verifier_for(produced).verify_claims(produced.claims)
    summary = summarize_reports(reports)
    assert (
        summary.grounded
        + summary.weakly_grounded
        + summary.ungrounded
        + (summary.fabricated)
        == summary.total
    )
    assert {report.flag for report in reports} == {GroundingFlag.GROUNDED}


def test_generated_corpus_shape_matches_the_documented_numbers() -> None:
    assert bundle(131).summary() == {
        "documents": 16,
        "claims": 6,
        "evidence": 12,
        "planted_links": 4,
        "competing": 2,
        "contradictions": 2,
        "paraphrase_pairs": 4,
        "distractors": 4,
    }
