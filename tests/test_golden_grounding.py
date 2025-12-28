"""Golden grounding payload.

The dictionary below is the exact structure the serializer must produce for one
cleanly grounded claim, including the rounded entity-overlap score. It pins the
wire format so downstream artifacts (reports, CLI output, archived runs) stay
readable across versions.
"""

from __future__ import annotations

from hypoarena.corpus import Corpus, Document
from hypoarena.grounding import GroundingFlag, GroundingVerifier
from hypoarena.schema import Citation, Claim, PredictedRelation, Provenance, Scope
from hypoarena.serialize import grounding_report_to_dict

DOC = "doc_0123456789ab"
TEXT = "Protein A increases cell growth in HeLa cells. Batch effects were reviewed."
QUOTE = "Protein A increases cell growth"

GOLDEN_REPORT = {
    "schema_version": "1.0",
    "claim_id": "clm_0123456789ab",
    "statement": "Protein A increases cell growth",
    "flag": "grounded",
    "score": 1.0,
    "issues": [],
    "checks": [
        {
            "document_id": "doc_0123456789ab",
            "start": 0,
            "end": 31,
            "quote": "Protein A increases cell growth",
            "resolved": True,
            "issues": [],
            "entity_overlap": 0.75,
            "claimed_numbers": [],
            "quoted_numbers": [],
            "detail": None,
        }
    ],
}


def grounded_report() -> object:
    claim = Claim(
        claim_id="clm_0123456789ab",
        statement=QUOTE,
        subject="protein A",
        object="cell growth",
        relation=PredictedRelation.INCREASES,
        scope=Scope(population="HeLa cells"),
        citations=(Citation(DOC, 0, len(QUOTE), QUOTE),),
        provenance=Provenance(origin="synthetic", seed=1),
    )
    verifier = GroundingVerifier(Corpus([Document(DOC, "Findings", TEXT)]))
    return verifier.verify(claim)


def test_grounded_report_matches_the_golden_payload() -> None:
    assert grounding_report_to_dict(grounded_report()) == GOLDEN_REPORT


def test_golden_flag_and_score_are_consistent() -> None:
    report = grounded_report()
    assert report.flag is GroundingFlag.GROUNDED
    assert report.score == GOLDEN_REPORT["score"]
    assert (
        report.checks[0].entity_overlap == GOLDEN_REPORT["checks"][0]["entity_overlap"]
    )


def test_golden_entity_overlap_is_the_documented_jaccard_value() -> None:
    # claim terms {protein, cell, growth} over quote tokens
    # {protein, increases, cell, growth} -> 3/4
    assert GOLDEN_REPORT["checks"][0]["entity_overlap"] == 0.75
