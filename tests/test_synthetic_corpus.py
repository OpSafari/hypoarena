"""Whole-corpus generation: counts, determinism and structure."""

from __future__ import annotations

from hypoarena.synthetic import GeneratedCorpus, SyntheticConfig, build_corpus, generate


def test_generated_corpus_matches_the_expected_document_count() -> None:
    config = SyntheticConfig(seed=4, chains=2, chain_length=3, paraphrases_per_link=2)
    generated = generate(config)
    assert len(generated.corpus) == config.expected_documents()
    assert len(generated.corpus) == 2 * 2 * 2 + 2 + 2 + 4


def test_generation_is_deterministic_for_a_seed() -> None:
    config = SyntheticConfig(seed=8)
    first = generate(config)
    second = generate(config)
    assert first.corpus.signature() == second.corpus.signature()
    assert first.findings == second.findings
    assert first.chains == second.chains


def test_different_seeds_produce_different_corpora() -> None:
    assert generate(SyntheticConfig(seed=1)).corpus.signature() != (
        generate(SyntheticConfig(seed=2)).corpus.signature()
    )


def test_findings_cover_planted_competing_and_negated_links() -> None:
    generated = generate(SyntheticConfig(seed=6, chains=2, chain_length=3))
    kinds = {finding.link.kind for finding in generated.findings}
    assert kinds == {"planted", "competing", "contradiction"}
    planted = [item for item in generated.findings if item.link.kind == "planted"]
    assert len(planted) == 2 * 2 * generated.config.paraphrases_per_link


def test_findings_for_selects_by_link_and_kind() -> None:
    generated = generate(SyntheticConfig(seed=6, chains=1, chain_length=2))
    link = generated.chains[0].links[0]
    assert len(generated.findings_for(link)) == generated.config.paraphrases_per_link
    assert generated.findings_for(generated.contradictions[0]) != ()
    for finding in generated.findings_for(generated.contradictions[0]):
        assert finding.is_negated is True


def test_distractor_ids_are_part_of_the_corpus() -> None:
    generated = generate(SyntheticConfig(seed=6, distractor_documents=3))
    assert len(generated.distractor_ids) == 3
    for document_id in generated.distractor_ids:
        assert generated.corpus.has_document(document_id)
        assert generated.corpus.document(document_id).attribute("kind") == "distractor"


def test_build_corpus_helper_returns_a_corpus() -> None:
    corpus = build_corpus(12, chains=1, chain_length=2, distractor_documents=1)
    assert isinstance(corpus.signature(), str)
    assert (
        len(corpus)
        == SyntheticConfig(
            seed=12, chains=1, chain_length=2, distractor_documents=1
        ).expected_documents()
    )


def test_every_finding_citation_resolves_against_the_corpus() -> None:
    generated = generate(SyntheticConfig(seed=15, chains=2, chain_length=3))
    assert isinstance(generated, GeneratedCorpus)
    for finding in generated.findings:
        assert generated.corpus.resolve(finding.citation) == finding.sentence
