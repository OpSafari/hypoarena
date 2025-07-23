"""Adversarial sweeps over synthetic bundles.

Each tampering mirrors a realistic failure of a generated hypothesis: a quote
that was edited after the citation was made, a span that drifted when the
document changed, a reference to a document that is not in the corpus, a
statement whose polarity was flipped, and an effect size that was inflated. None
of them may be graded ``grounded``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hypoarena.grounding import GroundingFlag, GroundingVerifier
from hypoarena.schema import Citation
from hypoarena.synthetic import SyntheticBundle, SyntheticConfig, build_bundle


def bundle() -> SyntheticBundle:
    return build_bundle(SyntheticConfig(seed=131, chains=2, chain_length=3))


def corrupt_quote(citation: Citation) -> Citation:
    marker = "X" if citation.quote[0] != "X" else "Y"
    return replace(citation, quote=marker + citation.quote[1:])


def shift_span(citation: Citation) -> Citation:
    return replace(citation, start=citation.start + 1, end=citation.end + 1)


def test_tampering_with_the_quote_is_fabricated() -> None:
    produced = bundle()
    verifier = GroundingVerifier(produced.corpus)
    for claim in produced.claims[:5]:
        tampered = replace(
            claim, citations=tuple(corrupt_quote(item) for item in claim.citations)
        )
        report = verifier.verify(tampered)
        assert report.flag is GroundingFlag.FABRICATED
        assert report.is_grounded is False


def test_shifting_a_span_is_fabricated() -> None:
    produced = bundle()
    verifier = GroundingVerifier(produced.corpus)
    for claim in produced.claims[:5]:
        tampered = replace(
            claim, citations=tuple(shift_span(item) for item in claim.citations)
        )
        assert verifier.verify(tampered).flag is GroundingFlag.FABRICATED


def test_citing_an_absent_document_is_fabricated() -> None:
    produced = bundle()
    verifier = GroundingVerifier(produced.corpus)
    claim = produced.claims[0]
    absent = replace(claim.citations[0], document_id="doc_999999999999")
    tampered = replace(claim, citations=(absent,))
    report = verifier.verify(tampered)
    assert report.flag is GroundingFlag.FABRICATED
    assert report.score == 0.0


def test_flipping_the_statement_polarity_is_at_least_weak() -> None:
    produced = bundle()
    verifier = GroundingVerifier(produced.corpus)
    claim = produced.claims[0]
    negated = replace(claim, statement=f"We did not observe that {claim.statement}")
    report = verifier.verify(negated)
    assert report.flag in {GroundingFlag.WEAK, GroundingFlag.FABRICATED}
    assert report.is_grounded is False


def test_inflating_an_effect_size_is_at_least_weak() -> None:
    produced = bundle()
    verifier = GroundingVerifier(produced.corpus)
    claim = produced.claims[0]
    inflated = replace(claim, statement=f"{claim.statement} by 3.5 fold")
    report = verifier.verify(inflated)
    assert report.is_grounded is False
    assert report.flag is GroundingFlag.WEAK


def test_dropping_every_citation_is_ungrounded() -> None:
    produced = bundle()
    verifier = GroundingVerifier(produced.corpus)
    claim = produced.claims[0]
    assert verifier.verify(replace(claim, citations=())).flag is (
        GroundingFlag.UNGROUNDED
    )


@pytest.mark.parametrize("seed", [131, 977])
def test_no_tampered_claim_is_ever_grounded(seed: int) -> None:
    produced = build_bundle(SyntheticConfig(seed=seed, chains=2))
    verifier = GroundingVerifier(produced.corpus)
    tamperings = (
        lambda item: replace(item, quote="entirely invented finding text"),
        shift_span,
        lambda item: replace(item, document_id="doc_888888888888"),
    )
    for claim in produced.claims:
        for tamper in tamperings:
            edited = replace(
                claim, citations=tuple(tamper(item) for item in claim.citations)
            )
            report = verifier.verify(edited)
            assert report.flag is GroundingFlag.FABRICATED, (claim.claim_id, tamper)
