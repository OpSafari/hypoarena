"""Gold evidence: polarity, strength schedule and citation integrity."""

from __future__ import annotations

from hypoarena.schema import Evidence, EvidencePolarity
from hypoarena.synthetic import (
    CONTRADICTION_STRENGTH,
    SUPPORT_STRENGTHS,
    GeneratedCorpus,
    SyntheticConfig,
    generate,
    gold_evidence,
)


def bundle() -> GeneratedCorpus:
    return generate(SyntheticConfig(seed=41, chains=2, chain_length=3))


def evidence_for(generated: GeneratedCorpus) -> tuple[Evidence, ...]:
    return gold_evidence(generated, corpus_hash=generated.corpus.signature())


def test_one_evidence_item_per_finding() -> None:
    generated = bundle()
    produced = evidence_for(generated)
    assert len(produced) == len(generated.findings)
    assert len({item.evidence_id for item in produced}) == len(produced)


def test_negated_findings_become_refuting_evidence() -> None:
    generated = bundle()
    produced = evidence_for(generated)
    for item, finding in zip(produced, generated.findings, strict=True):
        if finding.is_negated:
            assert item.polarity is EvidencePolarity.REFUTE
            assert item.strength == CONTRADICTION_STRENGTH
        else:
            assert item.polarity is EvidencePolarity.SUPPORT
            assert item.strength in SUPPORT_STRENGTHS
    assert any(item.polarity is EvidencePolarity.REFUTE for item in produced)


def test_support_strengths_follow_the_position_schedule() -> None:
    generated = bundle()
    produced = evidence_for(generated)
    checked = 0
    for index, (item, finding) in enumerate(
        zip(produced, generated.findings, strict=True)
    ):
        if finding.is_negated:
            continue
        assert item.strength == SUPPORT_STRENGTHS[index % len(SUPPORT_STRENGTHS)]
        checked += 1
    assert checked == len(produced) - sum(
        1 for finding in generated.findings if finding.is_negated
    )


def test_evidence_citations_resolve_and_statements_match() -> None:
    generated = bundle()
    for item, finding in zip(evidence_for(generated), generated.findings, strict=True):
        assert item.citations == (finding.citation,)
        assert item.statement == finding.sentence
        assert generated.corpus.resolve(item.citations[0]) == item.statement
        assert item.method == "synthetic_finding"
        assert item.provenance.notes == finding.link.kind
