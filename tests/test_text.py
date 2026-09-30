"""Text normalization: whitespace, case, punctuation and token boundaries."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.text import (
    NEGATION_CUES,
    STOPWORDS,
    char_ngrams,
    content_tokens,
    extract_numbers,
    fold_accents,
    has_negation,
    negation_flip,
    normalize,
    normalize_whitespace,
    sentence_split,
    strip_punctuation,
    tokenize,
    word_ngrams,
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
    assert content_tokens("the kinase binds the promoter") == [
        "kinase",
        "binds",
        "promoter",
    ]


def test_content_tokens_keeps_every_token_when_there_are_no_stopwords() -> None:
    text = "kinase phosphorylates substrate"
    assert content_tokens(text) == tokenize(text)


def test_stopword_list_is_lowercase_and_unique() -> None:
    assert all(word == word.casefold() for word in STOPWORDS)
    assert "the" in STOPWORDS and "protein" not in STOPWORDS


def test_char_ngrams_slide_over_the_compact_text() -> None:
    assert char_ngrams("abc", 2) == ["ab", "bc"]
    assert char_ngrams("a b", 2) == ["ab"]


def test_char_ngrams_handle_short_and_empty_inputs() -> None:
    assert char_ngrams("ab", 5) == ["ab"]
    assert char_ngrams("", 3) == []
    assert char_ngrams("!!!", 3) == []


def test_char_ngrams_unigram_matches_compact_characters() -> None:
    assert char_ngrams("a b", 1) == ["a", "b"]


def test_word_ngrams_join_tokens_with_single_spaces() -> None:
    assert word_ngrams(["a", "b", "c"], 2) == ["a b", "b c"]
    assert word_ngrams(["a", "b", "c"], 3) == ["a b c"]


def test_word_ngrams_handle_short_and_empty_inputs() -> None:
    assert word_ngrams(["a"], 2) == ["a"]
    assert word_ngrams([], 2) == []


@pytest.mark.parametrize("size", [0, -1])
def test_ngram_builders_reject_non_positive_sizes(size: int) -> None:
    with pytest.raises(ValidationError):
        char_ngrams("abc", size)
    with pytest.raises(ValidationError):
        word_ngrams(["a", "b"], size)


def test_extract_numbers_finds_integers_floats_and_signs() -> None:
    assert extract_numbers("levels rose by 12 and fell to -3.5 (+0.25)") == [
        12.0,
        -3.5,
        0.25,
    ]


def test_extract_numbers_handles_scientific_notation_and_percent() -> None:
    assert extract_numbers("Kd = 1.5e-9 M, 42% of cells") == [1.5e-9, 42.0]


def test_extract_numbers_ignores_numbers_glued_to_letters() -> None:
    assert extract_numbers("p53 and IL6 are proteins") == []


def test_extract_numbers_partially_reads_dotted_version_strings() -> None:
    assert extract_numbers("v1.2.3") == [2.3]


def test_extract_numbers_returns_empty_list_without_digits() -> None:
    assert extract_numbers("no numeric content here") == []


def test_sentence_split_separates_on_terminal_punctuation() -> None:
    text = "Protein A binds B. Binding is dose dependent! Does it matter?"
    assert sentence_split(text) == [
        "Protein A binds B.",
        "Binding is dose dependent!",
        "Does it matter?",
    ]


def test_sentence_split_preserves_case_and_inner_punctuation() -> None:
    assert sentence_split("Levels rise (2x). Then they fall.") == [
        "Levels rise (2x).",
        "Then they fall.",
    ]


def test_sentence_split_returns_empty_list_for_blank_input() -> None:
    assert sentence_split("   ") == []


def test_sentence_split_keeps_a_single_unterminated_sentence() -> None:
    assert sentence_split("no terminal punctuation here") == [
        "no terminal punctuation here"
    ]


def test_sentence_split_over_splits_abbreviations_by_design() -> None:
    assert sentence_split("See Fig. 2 for details.") == ["See Fig.", "2 for details."]


def test_strip_punctuation_keeps_contractions_as_single_tokens() -> None:
    assert strip_punctuation("doesn't") == "doesnt"
    assert (
        strip_punctuation("the assay doesn’t replicate") == "the assay doesnt replicate"
    )


def test_strip_punctuation_leaves_trailing_apostrophes_alone() -> None:
    assert strip_punctuation("the cells' response") == "the cells  response"


def test_tokenize_yields_one_token_per_contraction() -> None:
    assert tokenize("Binding doesn't occur") == ["binding", "doesnt", "occur"]


def test_has_negation_detects_cues_in_normalized_tokens() -> None:
    assert has_negation(tokenize("Binding does not occur.")) is True
    assert has_negation(tokenize("The assay doesn't replicate")) is True


def test_has_negation_is_false_for_positive_statements() -> None:
    assert has_negation(tokenize("Binding occurs in a dose dependent way")) is False
    assert has_negation([]) is False


def test_negation_flip_is_true_only_when_polarity_differs() -> None:
    assert negation_flip("A binds B", "A does not bind B") is True
    assert negation_flip("A binds B", "A binds B strongly") is False
    assert negation_flip("A fails to bind B", "A does not bind B") is False


def test_negation_cues_are_normalized_forms() -> None:
    assert "doesnt" in NEGATION_CUES
    assert "doesn't" not in NEGATION_CUES
    assert all(cue == normalize(cue) for cue in NEGATION_CUES)


def test_tokenize_emits_one_token_per_cjk_ideograph() -> None:
    assert tokenize("基因G3导致") == ["基", "因", "g3", "导", "致"]


def test_tokenize_returns_empty_for_cjk_punctuation_only() -> None:
    assert tokenize("，。！？") == []


def test_content_tokens_preserves_cjk_characters() -> None:
    assert content_tokens("基因G3导致") == ["基", "因", "g3", "导", "致"]


def test_word_shingles_no_longer_collapse_distinct_cjk_claims() -> None:
    from hypoarena import dedup

    left = "基因G3导致细胞凋亡率上升"
    right = "基因G3抑制肿瘤转移"
    assert dedup.jaccard_similarity(left, right, words=True) < 0.5


def test_word_shingles_order_cjk_paraphrase_above_unrelated() -> None:
    from hypoarena import dedup

    claim = "基因G3导致细胞凋亡率上升"
    paraphrase = "G3基因导致细胞的凋亡率上升"
    unrelated = "代谢物M1抑制肿瘤细胞的生长速度"
    close = dedup.jaccard_similarity(claim, paraphrase, n=2, words=True)
    apart = dedup.jaccard_similarity(claim, unrelated, n=2, words=True)
    assert close > apart
    assert close > 0.25
