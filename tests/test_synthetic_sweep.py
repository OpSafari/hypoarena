"""Seeded sweeps over the generator.

These are the properties the rest of the toolkit takes for granted: a fixed seed
reproduces a corpus byte for byte, every gold citation resolves against the text
it quotes, planted clusters really do contain the paraphrases they claim, and the
derived graph stays valid.
"""

from __future__ import annotations

import pytest

from hypoarena.schema import EvidencePolarity
from hypoarena.serialize import corpus_from_text, corpus_to_text
from hypoarena.synthetic import SyntheticConfig, build_bundle

SEEDS = (1, 270106, 999983)


@pytest.mark.parametrize("seed", SEEDS)
def test_generation_is_reproducible(seed: int) -> None:
    config = SyntheticConfig(seed=seed, chains=2, chain_length=3)
    first = build_bundle(config)
    second = build_bundle(config)
    assert first.corpus.signature() == second.corpus.signature()
    assert [claim.claim_id for claim in first.claims] == [
        claim.claim_id for claim in second.claims
    ]
    assert [item.evidence_id for item in first.evidence] == [
        item.evidence_id for item in second.evidence
    ]


@pytest.mark.parametrize("seed", SEEDS)
def test_every_gold_citation_resolves(seed: int) -> None:
    bundle = build_bundle(SyntheticConfig(seed=seed))
    assert bundle.claims and bundle.evidence
    for claim in bundle.claims:
        for citation in claim.citations:
            assert bundle.corpus.resolve(citation) == citation.quote
    for item in bundle.evidence:
        assert bundle.corpus.resolve(item.citations[0]) == item.statement


@pytest.mark.parametrize("seed", SEEDS)
def test_corpus_serialization_is_byte_stable(seed: int) -> None:
    bundle = build_bundle(SyntheticConfig(seed=seed, chains=2))
    text = corpus_to_text(bundle.corpus)
    assert corpus_to_text(corpus_from_text(text)) == text


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_clusters_are_present_in_the_corpus(seed: int) -> None:
    bundle = build_bundle(SyntheticConfig(seed=seed, chains=2, paraphrases_per_link=3))
    for cluster in bundle.truth.clusters:
        assert cluster.size == 3
        for document_id in cluster.document_ids:
            document = bundle.corpus.document(document_id)
            assert cluster.statement.split()[0] in document.text


@pytest.mark.parametrize("seed", SEEDS)
def test_rivals_and_negations_are_distinct_from_the_truth(seed: int) -> None:
    bundle = build_bundle(SyntheticConfig(seed=seed, chains=2))
    planted_keys = {link.key() for link in bundle.truth.true_links()}
    assert all(rival.key() not in planted_keys for rival in bundle.truth.competing)
    refuting = [
        item for item in bundle.evidence if item.polarity is EvidencePolarity.REFUTE
    ]
    assert len(refuting) == len(bundle.truth.contradictions)
    for item in refuting:
        assert any(cue in item.statement.lower() for cue in ("not", "no "))


@pytest.mark.parametrize("seed", SEEDS)
def test_derived_graph_is_valid(seed: int) -> None:
    bundle = build_bundle(SyntheticConfig(seed=seed, chains=2))
    graph = bundle.graph()
    assert graph.validate() is None
    assert graph.link_count == len(bundle.evidence)


def test_distractors_share_no_vocabulary_with_planted_links() -> None:
    bundle = build_bundle(SyntheticConfig(seed=5, chains=2, distractor_documents=3))
    planted_words = {
        word
        for link in bundle.truth.true_links()
        for word in (link.subject, link.target)
    }
    for document_id in bundle.truth.distractor_ids:
        text = bundle.corpus.document(document_id).text
        assert not any(variable in text for variable in planted_words)
