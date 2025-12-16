"""Per-citation verification against a small hand-built corpus."""

from __future__ import annotations

from hypoarena.corpus import Corpus, Document
from hypoarena.grounding import GroundingIssue, GroundingVerifier, VerifierConfig
from hypoarena.schema import Citation, Claim, PredictedRelation, Provenance, Scope

DOC = "doc_0123456789ab"
TEXT = (
    "Protein A increases cell growth in HeLa cells. "
    "We did not observe that protein A increases cell growth in HeLa cells. "
    "Growth increased 1.5 fold after adding protein A."
)
POSITIVE = "Protein A increases cell growth"
NEGATED_START = TEXT.index("We did not")
NUMBER_START = TEXT.index("Growth increased")


def corpus() -> Corpus:
    return Corpus([Document(DOC, "Synthetic findings", TEXT)])


def claim(statement: str = POSITIVE, **overrides: object) -> Claim:
    payload: dict[str, object] = {
        "claim_id": "clm_0123456789ab",
        "statement": statement,
        "subject": "protein A",
        "object": "cell growth",
        "relation": PredictedRelation.INCREASES,
        "scope": Scope(population="HeLa cells"),
        "provenance": Provenance(origin="synthetic", seed=1),
    }
    payload.update(overrides)
    return Claim(**payload)  # type: ignore[arg-type]


def cite(start: int, length: int, quote: str | None = None) -> Citation:
    return Citation(DOC, start, start + length, quote or TEXT[start : start + length])


def verifier(**overrides: object) -> GroundingVerifier:
    config = VerifierConfig(**overrides)  # type: ignore[arg-type]
    return GroundingVerifier(corpus(), config)


def test_a_faithful_citation_is_clean() -> None:
    start = TEXT.index(POSITIVE)
    check = verifier().check_citation(claim(), cite(start, len(POSITIVE)))
    assert check.resolved is True
    assert check.issues == ()
    assert check.entity_overlap > 0.5


def test_a_fabricated_quote_is_not_resolved() -> None:
    start = TEXT.index(POSITIVE)
    check = verifier().check_citation(
        claim(), cite(start, len(POSITIVE), quote="Protein A decreases cell growth")
    )
    assert check.resolved is False
    assert GroundingIssue.QUOTE_MISMATCH in check.issues
    assert check.is_fabricated is True
    assert "document says" in (check.detail or "")


def test_a_negated_quote_raises_a_polarity_conflict() -> None:
    length = len(
        "We did not observe that protein A increases cell growth in HeLa cells."
    )
    check = verifier().check_citation(claim(), cite(NEGATED_START, length))
    assert check.resolved is True
    assert check.is_fabricated is False
    assert GroundingIssue.POLARITY_CONFLICT in check.issues


def test_polarity_checks_can_be_disabled() -> None:
    length = len(
        "We did not observe that protein A increases cell growth in HeLa cells."
    )
    check = verifier(polarity_checks=False).check_citation(
        claim(), cite(NEGATED_START, length)
    )
    assert GroundingIssue.POLARITY_CONFLICT not in check.issues


def test_a_drifted_number_is_reported() -> None:
    numeric = claim(statement="Protein A increases cell growth by 2.5 fold")
    length = len("Growth increased 1.5 fold after adding protein A.")
    check = verifier().check_citation(numeric, cite(NUMBER_START, length))
    assert check.claimed_numbers == (2.5,)
    assert check.quoted_numbers == (1.5,)
    assert GroundingIssue.NUMERIC_MISMATCH in check.issues


def test_a_matching_number_is_not_reported() -> None:
    numeric = claim(statement="Protein A increases cell growth by 1.5 fold")
    length = len("Growth increased 1.5 fold after adding protein A.")
    check = verifier().check_citation(numeric, cite(NUMBER_START, length))
    assert GroundingIssue.NUMERIC_MISMATCH not in check.issues


def test_short_quotes_are_flagged_as_weak() -> None:
    check = verifier(min_quote_length=80).check_citation(
        claim(), cite(TEXT.index(POSITIVE), len(POSITIVE))
    )
    assert GroundingIssue.SHORT_QUOTE in check.issues
    assert check.is_fabricated is False


def test_low_entity_overlap_is_flagged() -> None:
    start = TEXT.index("Growth increased")
    unrelated = claim(subject="kinase K1", object="splice fidelity")
    length = len("Growth increased 1.5 fold after adding protein A.")
    check = verifier().check_citation(unrelated, cite(start, length))
    assert GroundingIssue.LOW_ENTITY_OVERLAP in check.issues
    assert check.entity_overlap == 0.0
