"""Golden outputs for the text normalization chain.

The expectations below were derived by hand from the documented behavior and
then frozen. They exist so that a change in folding order, punctuation handling
or number extraction becomes a deliberate decision instead of a silent drift in
every downstream grounding and dedup measurement.
"""

from __future__ import annotations

import pytest

from hypoarena.text import (
    char_ngrams,
    content_tokens,
    extract_numbers,
    has_negation,
    normalize,
    sentence_split,
    tokenize,
    word_ngrams,
)

NORMALIZE_GOLDEN = [
    ("  Café   NOT   found! ", "cafe not found"),
    ("Protein (p53) -- binds DNA; see Fig. 2!", "protein p53 binds dna see fig 2"),
    ("ß-cell  Ångström", "ss cell angstrom"),
    ("already normalized", "already normalized"),
    ("", ""),
]

TOKENIZE_GOLDEN = [
    ("Gene A binds gene-B!", ["gene", "a", "binds", "gene", "b"]),
    ("IL6, p53 and TNF-α levels", ["il6", "p53", "and", "tnf", "levels"]),
    ("no tokens here: !!!", ["no", "tokens", "here"]),
]

CONTENT_TOKENS_GOLDEN = [
    ("the protein binds to the receptor", ["protein", "binds", "receptor"]),
    ("we do not observe any effect", ["observe", "effect"]),
]

NUMBER_GOLDEN = [
    ("Effect: +1.5 fold (p = 0.03), n=42", [1.5, 0.03, 42.0]),
    ("no numbers at all", []),
    ("p53 stays glued", []),
]

SENTENCE_GOLDEN = [
    (
        "A binds B. It does not bind C! Does D bind E?",
        ["A binds B.", "It does not bind C!", "Does D bind E?"],
    ),
    ("single sentence without terminator", ["single sentence without terminator"]),
]

NEGATION_GOLDEN = [
    ("A does not bind B", True),
    ("A binds B", False),
    ("The assay fails to replicate", True),
    ("Binding lacks specificity", True),
]


@pytest.mark.parametrize(("raw", "expected"), NORMALIZE_GOLDEN)
def test_normalize_golden(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), TOKENIZE_GOLDEN)
def test_tokenize_golden(raw: str, expected: list[str]) -> None:
    assert tokenize(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), CONTENT_TOKENS_GOLDEN)
def test_content_tokens_golden(raw: str, expected: list[str]) -> None:
    assert content_tokens(raw) == expected


def test_char_and_word_ngram_golden() -> None:
    assert char_ngrams("protein", 3) == ["pro", "rot", "ote", "tei", "ein"]
    assert char_ngrams("a b", 3) == ["ab"]
    assert word_ngrams(["a", "b", "c", "d"], 2) == ["a b", "b c", "c d"]


@pytest.mark.parametrize(("raw", "expected"), NUMBER_GOLDEN)
def test_extract_numbers_golden(raw: str, expected: list[float]) -> None:
    assert extract_numbers(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), SENTENCE_GOLDEN)
def test_sentence_split_golden(raw: str, expected: list[str]) -> None:
    assert sentence_split(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), NEGATION_GOLDEN)
def test_negation_golden(raw: str, expected: bool) -> None:
    assert has_negation(tokenize(raw)) is expected
