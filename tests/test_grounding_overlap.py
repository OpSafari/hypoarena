"""Citation check records and entity overlap scoring."""

from __future__ import annotations

from helpers import sample_citation, sample_claim
from hypoarena.grounding import (
    CitationCheck,
    GroundingIssue,
    claim_terms,
    entity_overlap,
)
from hypoarena.text import jaccard


def check(**overrides: object) -> CitationCheck:
    payload: dict[str, object] = {
        "citation": sample_citation(),
        "resolved": True,
        "issues": (),
        "entity_overlap": 0.8,
        "claimed_numbers": (),
        "quoted_numbers": (),
    }
    payload.update(overrides)
    return CitationCheck(**payload)  # type: ignore[arg-type]


def test_jaccard_is_symmetric_and_handles_empty_sets() -> None:
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_claim_terms_come_from_the_variables_only() -> None:
    claim = sample_claim(subject="protein A", object="gene B expression")
    assert claim_terms(claim) == {"protein", "gene", "b", "expression"}
    assert "increases" not in claim_terms(claim)


def test_entity_overlap_rewards_shared_entities() -> None:
    claim = sample_claim(subject="protein A", object="cell growth")
    high = entity_overlap(claim, "protein A drives cell growth in these assays")
    low = entity_overlap(claim, "unrelated batch effects were observed")
    assert high > 0.5
    assert low == 0.0


def test_citation_check_classifies_issues() -> None:
    assert check().is_clean is True
    assert check().is_fabricated is False
    fabricated = check(resolved=False, issues=(GroundingIssue.QUOTE_MISMATCH,))
    assert fabricated.is_clean is False
    assert fabricated.is_fabricated is True
    weak = check(issues=(GroundingIssue.LOW_ENTITY_OVERLAP,))
    assert weak.is_fabricated is False
    assert weak.is_clean is False


def test_citation_check_exposes_its_span() -> None:
    assert check().span() == ("doc_0123456789ab", 10, 25)
