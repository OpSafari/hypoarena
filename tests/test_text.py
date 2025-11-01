"""Text normalization: whitespace, case, punctuation and token boundaries."""

from __future__ import annotations

from hypoarena.text import (
    STOPWORDS,
    content_tokens,
    fold_accents,
    normalize,
    normalize_whitespace,
    strip_punctuation,
    tokenize,
)


def test_whitespace_runs_collapse_to_single_spaces() -> None:
    assert normalize_whitespace("  a \t\n b  ") == "a b"


def test_normalize_whitespace_is_idempotent_and_keeps_inner_text() -> None:
    once = normalize_whitespace("gene x\tbinds  y")
    assert normalize_whitespace(once) == once
    assert once == "gene x binds y"


def test_empty_and_whitespace_only_inputs_become_empty() -> None:
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   \n\t ") == ""


def test_fold_accents_maps_to_ascii_equivalents() -> None:
    assert fold_accents("café naïve") == "cafe naive"


def test_fold_accents_keeps_characters_without_decomposition() -> None:
    assert fold_accents("蛋白 p53") == "蛋白 p53"


def test_fold_accents_is_idempotent() -> None:
    assert fold_accents(fold_accents("Ångström")) == fold_accents("Ångström")


def test_strip_punctuation_uses_spaces_so_words_do_not_merge() -> None:
    assert strip_punctuation("state-of-the-art") == "state of the art"
    assert strip_punctuation("a,b") == "a b"


def test_strip_punctuation_keeps_word_characters_and_spaces() -> None:
    assert strip_punctuation("gene_1 binds RNA") == "gene_1 binds RNA"


def test_normalize_lowercases_strips_and_collapses() -> None:
    assert normalize("  Café   NOT   found! ") == "cafe not found"


def test_normalize_can_preserve_case_and_punctuation() -> None:
    assert normalize("A.  B", lower=False, remove_punctuation=False) == "A. B"
    assert (
        normalize("A. B", collapse_whitespace=False, remove_punctuation=False) == "a. b"
    )


def test_normalize_is_idempotent_with_default_flags() -> None:
    once = normalize("Protein (p53) -- binds DNA; see Fig. 2!")
    assert normalize(once) == once


def test_normalize_without_folding_keeps_accents() -> None:
    assert normalize("café", fold=False) == "café"


def test_tokenize_splits_on_punctuation_and_whitespace() -> None:
    assert tokenize("Gene A binds gene-B!") == ["gene", "a", "binds", "gene", "b"]


def test_tokenize_keeps_alphanumeric_terms_together() -> None:
    assert tokenize("IL6 and p53 levels") == ["il6", "and", "p53", "levels"]


def test_tokenize_returns_empty_list_for_punctuation_only() -> None:
    assert tokenize("!!! --- ...") == []


def test_content_tokens_drops_stopwords_but_keeps_order() -> None:
    assert content_tokens("the protein binds to the receptor") == [
        "protein",
        "binds",
        "receptor",
    ]


def test_content_tokens_keeps_every_token_when_there_are_no_stopwords() -> None:
    text = "kinase phosphorylates substrate"
    assert content_tokens(text) == tokenize(text)


def test_stopword_list_is_lowercase_and_unique() -> None:
    assert all(word == word.casefold() for word in STOPWORDS)
    assert "the" in STOPWORDS and "protein" not in STOPWORDS
