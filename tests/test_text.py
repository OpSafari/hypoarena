"""Text normalization: whitespace, case, punctuation and token boundaries."""

from __future__ import annotations

from hypoarena.text import fold_accents, normalize_whitespace


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
