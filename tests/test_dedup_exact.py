"""Exact duplicate detection on normalized text."""

from __future__ import annotations

from hypoarena.dedup import duplicate_rate, exact_duplicates, normalized_signature


def test_signatures_ignore_case_punctuation_and_whitespace() -> None:
    base = normalized_signature("Protein A increases cell growth")
    assert normalized_signature("protein a   increases CELL growth!") == base
    assert normalized_signature("Protein A decreases cell growth") != base


def test_identical_texts_are_grouped() -> None:
    texts = {
        "a": "Protein A increases cell growth",
        "b": "protein a increases cell growth.",
        "c": "Kinase K1 binds the promoter",
    }
    assert exact_duplicates(texts) == (("a", "b"),)


def test_distinct_texts_produce_no_groups() -> None:
    texts = {"a": "first statement", "b": "second statement"}
    assert exact_duplicates(texts) == ()
    assert duplicate_rate(texts) == 0.0


def test_groups_are_sorted_and_complete() -> None:
    texts = {
        "z": "same text",
        "y": "Same   text!",
        "x": "same text",
        "w": "other text",
    }
    assert exact_duplicates(texts) == (("x", "y", "z"),)
    assert duplicate_rate(texts) == 0.75


def test_empty_input_is_handled() -> None:
    assert exact_duplicates({}) == ()
    assert duplicate_rate({}) == 0.0


def test_grouping_does_not_depend_on_insertion_order() -> None:
    forward = {"a": "one", "b": "two", "c": "one"}
    backward = {"c": "one", "b": "two", "a": "one"}
    assert exact_duplicates(forward) == exact_duplicates(backward)
