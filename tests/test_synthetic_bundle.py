"""Bundle assembly: gold records, derived graph and summary counts."""

from __future__ import annotations

from hypoarena.schema import ClaimRelation, EvidencePolarity
from hypoarena.synthetic import SyntheticBundle, SyntheticConfig, build_bundle


def bundle() -> SyntheticBundle:
    return build_bundle(SyntheticConfig(seed=51, chains=2, chain_length=3))


def test_bundle_exposes_a_consistent_corpus_hash() -> None:
    produced = bundle()
    assert produced.corpus_hash == produced.corpus.signature()
    assert all(
        claim.provenance.corpus_hash == produced.corpus_hash
        for claim in produced.claims
    )


def test_summary_counts_match_the_records() -> None:
    produced = bundle()
    summary = produced.summary()
    assert summary["documents"] == len(produced.corpus)
    assert summary["claims"] == len(produced.claims)
    assert summary["evidence"] == len(produced.evidence)
    assert summary["planted_links"] == len(produced.truth.true_links())
    assert summary["competing"] == len(produced.truth.competing)
    assert summary["paraphrase_pairs"] == produced.truth.paraphrase_pairs()


def test_claim_for_finds_planted_and_rival_claims() -> None:
    produced = bundle()
    for link in produced.truth.true_links():
        assert produced.claim_for(link) is not None
    for rival in produced.truth.competing:
        assert produced.claim_for(rival) is not None


def test_graph_wires_evidence_and_contradiction_edges() -> None:
    produced = bundle()
    graph = produced.graph()
    assert len(graph) == len(produced.claims)
    assert len(graph.evidence_items) == len(produced.evidence)
    assert graph.link_count == len(produced.evidence)
    assert len(graph.edges) == len(produced.truth.competing)
    assert all(edge.relation is ClaimRelation.CONTRADICTS for edge in graph.edges)


def test_negated_findings_are_attached_to_planted_claims() -> None:
    produced = bundle()
    graph = produced.graph()
    refuting = [
        item for item in produced.evidence if item.polarity is EvidencePolarity.REFUTE
    ]
    assert refuting
    for item in refuting:
        owners = graph.claims_for_evidence(item.evidence_id)
        assert len(owners) == 1
        assert graph.claim(owners[0]).provenance.notes == "planted"


def test_build_bundle_accepts_seed_overrides() -> None:
    produced = build_bundle(seed=7, chains=1, chain_length=2, distractor_documents=0)
    assert produced.config.chains == 1
    assert produced.summary()["distractors"] == 0
    assert len(produced.corpus) == produced.config.expected_documents()
