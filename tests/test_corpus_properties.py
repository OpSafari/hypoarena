"""Seeded corpus sweeps.

Random documents are generated from an explicit ``random.Random(seed)``, so every
case is reproducible offline. The properties under test are the ones the
grounding verifier and the runner rely on: every sampled citation resolves,
serialization roundtrips are byte-identical, signatures ignore insertion order
and composition never loses documents.
"""

from __future__ import annotations

from random import Random

import pytest

from hypoarena.corpus import Corpus, Document, sample_spans
from hypoarena.ids import make_id
from hypoarena.serialize import corpus_from_text, corpus_to_text

SUBJECTS = ("protein A", "kinase K", "metabolite M", "gene B", "complex C")
VERBS = ("binds", "increases", "inhibits", "stabilizes", "colocalizes with")
OBJECTS = ("the promoter", "ribosome assembly", "membrane potential", "splice site")


def random_document(seed: int, index: int, sentences: int = 4) -> Document:
    rng = Random(f"{seed}:{index}")
    parts = []
    for step in range(sentences):
        subject = rng.choice(SUBJECTS)
        verb = rng.choice(VERBS)
        target = rng.choice(OBJECTS)
        parts.append(f"{subject} {verb} {target} in sample {step}.")
    return Document(
        make_id("doc", seed, index),
        f"Synthetic report {index}",
        " ".join(parts),
        "synthetic:v1",
        (("seed", str(seed)), ("index", str(index))),
    )


def random_corpus(seed: int, size: int = 8) -> Corpus:
    return Corpus([random_document(seed, index) for index in range(size)])


@pytest.mark.parametrize("seed", [1, 270106])
def test_every_sampled_citation_resolves(seed: int) -> None:
    corpus = random_corpus(seed)
    rng = Random(seed)
    total = 0
    for document in corpus:
        for citation in sample_spans(document, rng, 4):
            assert corpus.resolve(citation) == citation.quote
            total += 1
    assert total > 0


@pytest.mark.parametrize("seed", [1, 270106])
def test_serialization_roundtrip_is_byte_identical(seed: int) -> None:
    corpus = random_corpus(seed)
    text = corpus_to_text(corpus)
    assert corpus_to_text(corpus_from_text(text)) == text


def test_signature_ignores_insertion_order() -> None:
    forward = random_corpus(5)
    backward = Corpus(list(reversed(list(forward.documents))))
    assert forward.signature() == backward.signature()


def test_subcorpus_and_merge_preserve_documents() -> None:
    corpus = random_corpus(9, size=6)
    ids = corpus.document_ids
    left = corpus.subcorpus(ids[:3])
    right = corpus.subcorpus(ids[3:])
    assert len(left) == 3 and len(right) == 3
    assert left.merged(right).signature() == corpus.signature()


def test_truncation_is_detected_for_random_corpora() -> None:
    corpus = random_corpus(11, size=4)
    lines = corpus_to_text(corpus).splitlines(keepends=True)
    with pytest.raises(Exception) as info:  # noqa: B017, PT011 - category asserted below
        corpus_from_text("".join(lines[:-1]))
    assert info.value.__class__.__name__ == "SchemaError"
