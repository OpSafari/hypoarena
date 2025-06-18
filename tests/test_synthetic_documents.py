"""Document assembly: exact spans, shuffling and distractor isolation."""

from __future__ import annotations

from random import Random

from hypoarena.schema import PredictedRelation
from hypoarena.synthetic import (
    PlacedFinding,
    PlantedLink,
    SyntheticConfig,
    assemble_document,
    distractor_documents,
    finding_documents,
    join_sentences,
)


def link(kind: str = "planted") -> PlantedLink:
    return PlantedLink(
        chain_id="chn_0123456789ab",
        subject="protein A",
        target="cell growth",
        relation=PredictedRelation.INCREASES,
        system="HeLa cells",
        kind=kind,
    )


def test_join_sentences_tracks_finding_offsets() -> None:
    text, spans = join_sentences([("first sentence.", False), ("finding here.", True)])
    assert text == "first sentence. finding here."
    assert spans == [(16, 29)]
    assert text[spans[0][0] : spans[0][1]] == "finding here."


def test_join_sentences_handles_multiple_findings() -> None:
    text, spans = join_sentences([("a.", True), ("b.", False), ("c.", True)])
    assert [text[start:end] for start, end in spans] == ["a.", "c."]


def test_assemble_document_keeps_spans_exact_after_shuffling() -> None:
    rng = Random("shuffle")
    findings = ["protein A increases cell growth in HeLa cells."]
    filler = ["Samples were collected in triplicate.", "Controls were run in parallel."]
    document, spans = assemble_document(
        rng, "doc_0123456789ab", "A study", findings, filler, "synthetic:v1"
    )
    assert len(spans) == 1
    start, end = spans[0]
    assert document.text[start:end] == findings[0]
    assert all(sentence in document.text for sentence in filler)


def test_finding_documents_cite_their_own_text() -> None:
    config = SyntheticConfig(seed=5, filler_sentences=2)
    placed = finding_documents(
        config, config.rng(), link(), ["protein A increases cell growth."], 0, 0
    )
    finding = placed[0]
    assert isinstance(finding, PlacedFinding)
    assert finding.citation.quote == finding.sentence
    start, end = finding.citation.start, finding.citation.end
    assert finding.document.text[start:end] == finding.sentence
    assert finding.document_id == finding.document.document_id
    assert finding.is_negated is False


def test_document_ids_are_deterministic_and_kind_scoped() -> None:
    config = SyntheticConfig(seed=5)
    first = finding_documents(config, config.rng(), link(), ["s one."], 0, 0)
    second = finding_documents(config, config.rng(), link(), ["s one."], 0, 0)
    assert first[0].document_id == second[0].document_id
    negated = finding_documents(
        config, config.rng(), link("contradiction"), ["s one."], 0, 0
    )
    assert negated[0].document_id != first[0].document_id
    assert negated[0].is_negated is True


def test_documents_carry_kind_and_chain_metadata() -> None:
    config = SyntheticConfig(seed=5)
    placed = finding_documents(config, config.rng(), link(), ["s one."], 0, 0)
    assert placed[0].document.attribute("kind") == "planted"
    assert placed[0].document.attribute("chain") == "chn_0123456789ab"
    assert placed[0].document.source == config.source_tag


def test_distractors_avoid_planted_vocabulary() -> None:
    config = SyntheticConfig(seed=9, distractor_documents=3)
    documents = distractor_documents(config, config.rng())
    assert len(documents) == 3
    for document in documents:
        assert document.attribute("kind") == "distractor"
        assert "protein A" not in document.text
        assert "cell growth" not in document.text


def test_distractor_count_can_be_zero() -> None:
    config = SyntheticConfig(seed=9, distractor_documents=0)
    assert distractor_documents(config, config.rng()) == ()
