"""Bulk verification, graph traversal and summary aggregation."""

from __future__ import annotations

from dataclasses import replace

from hypoarena.corpus import Corpus, Document
from hypoarena.graph import HypothesisGraph
from hypoarena.grounding import (
    GroundingFlag,
    GroundingSummary,
    GroundingVerifier,
    summarize_reports,
)
from hypoarena.schema import Citation, Claim, PredictedRelation, Provenance, Scope

DOC = "doc_0123456789ab"
TEXT = "Protein A increases cell growth in HeLa cells. Batch effects were reviewed."
QUOTE = "Protein A increases cell growth"
START = TEXT.index(QUOTE)


def claim(claim_id: str, citations: tuple[Citation, ...]) -> Claim:
    return Claim(
        claim_id=claim_id,
        statement="Protein A increases cell growth",
        subject="protein A",
        object="cell growth",
        relation=PredictedRelation.INCREASES,
        scope=Scope(population="HeLa cells"),
        citations=citations,
        provenance=Provenance(origin="synthetic", seed=1),
    )


def verifier() -> GroundingVerifier:
    return GroundingVerifier(Corpus([Document(DOC, "Findings", TEXT)]))


def good() -> Citation:
    return Citation(DOC, START, START + len(QUOTE), QUOTE)


def test_verify_claims_preserves_order() -> None:
    claims = [
        claim("clm_0123456789ab", (good(),)),
        claim("clm_ffffffffffff", ()),
    ]
    reports = verifier().verify_claims(claims)
    assert [report.claim_id for report in reports] == [
        "clm_0123456789ab",
        "clm_ffffffffffff",
    ]
    assert reports[0].flag is GroundingFlag.GROUNDED
    assert reports[1].flag is GroundingFlag.UNGROUNDED


def test_verify_graph_covers_every_claim_in_sorted_order() -> None:
    graph = HypothesisGraph()
    for identifier in ("clm_ffffffffffff", "clm_0123456789ab"):
        graph.add_claim(claim(identifier, (good(),)))
    reports = verifier().verify_graph(graph)
    assert [report.claim_id for report in reports] == list(graph.claim_ids)


def test_summary_counts_every_flag() -> None:
    reports = verifier().verify_claims(
        [
            claim("clm_0123456789ab", (good(),)),
            claim("clm_111111111111", ()),
            claim(
                "clm_ffffffffffff",
                (Citation(DOC, START, START + len(QUOTE), "x" * len(QUOTE)),),
            ),
        ]
    )
    summary = summarize_reports(reports)
    assert isinstance(summary, GroundingSummary)
    assert summary.total == 3
    assert summary.grounded == 1
    assert summary.ungrounded == 1
    assert summary.fabricated == 1
    assert summary.weakly_grounded == 0
    assert summary.grounded_rate == round(1 / 3, 4)


def test_summary_of_no_reports_is_all_zero() -> None:
    summary = summarize_reports([])
    assert summary.as_dict() == {
        "total": 0,
        "grounded": 0,
        "weakly_grounded": 0,
        "ungrounded": 0,
        "fabricated": 0,
        "grounded_rate": 0.0,
        "mean_score": 0.0,
    }


def test_summary_mean_score_averages_partial_credit() -> None:
    filler_start = TEXT.index("Batch")
    filler = Citation(DOC, filler_start, len(TEXT), TEXT[filler_start:])
    reports = verifier().verify_claims(
        [claim("clm_0123456789ab", (good(),)), claim("clm_ffffffffffff", (filler,))]
    )
    summary = summarize_reports(reports)
    assert summary.weakly_grounded == 1
    assert summary.mean_score == round((1.0 + 0.5) / 2, 4)


def test_reports_are_independent_of_claim_mutation() -> None:
    original = claim("clm_0123456789ab", (good(),))
    edited = replace(original, statement="Protein A decreases cell growth")
    reports = verifier().verify_claims([original, edited])
    assert reports[0].statement != reports[1].statement
