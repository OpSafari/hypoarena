"""Context extraction from corpora and claim sets."""

from __future__ import annotations

import pytest

from hypoarena.corpus import Corpus, Document
from hypoarena.debate import CONTEXT_ELLIPSIS, claim_context, corpus_context
from hypoarena.errors import ValidationError
from hypoarena.synthetic import SyntheticConfig, build_bundle


def corpus() -> Corpus:
    return Corpus(
        [
            Document("doc_ffffffffffff", "Second", "Second finding. More detail here."),
            Document("doc_0123456789ab", "First", "First finding. More detail here."),
        ]
    )


def test_context_follows_identifier_order_not_insertion_order() -> None:
    lines = corpus_context(corpus())
    assert lines == ("First finding.", "Second finding.")


def test_limit_caps_the_number_of_lines() -> None:
    assert len(corpus_context(corpus(), limit=1)) == 1
    with pytest.raises(ValidationError, match="limit"):
        corpus_context(corpus(), limit=0)


def test_long_sentences_are_truncated_with_an_ellipsis() -> None:
    long_text = "An extremely long first sentence that keeps going and going."
    store = Corpus([Document("doc_0123456789ab", "Long", long_text)])
    line = corpus_context(store, max_chars=30)[0]
    assert len(line) == 30
    assert line.endswith(CONTEXT_ELLIPSIS)
    with pytest.raises(ValidationError, match="too small"):
        corpus_context(store, max_chars=10)


def test_context_works_on_a_generated_corpus() -> None:
    bundle = build_bundle(SyntheticConfig(seed=7, chains=1, chain_length=2))
    lines = corpus_context(bundle.corpus, limit=4)
    assert len(lines) == 4
    assert all(line.strip() for line in lines)
    assert corpus_context(bundle.corpus, limit=4) == lines


def test_claim_context_lists_statements() -> None:
    bundle = build_bundle(SyntheticConfig(seed=7, chains=1, chain_length=2))
    lines = claim_context(bundle.claims, limit=2)
    assert len(lines) == 2
    assert lines[0] == bundle.claims[0].statement
    assert claim_context([]) == ()
    with pytest.raises(ValidationError, match="limit"):
        claim_context(bundle.claims, limit=0)
