"""Deterministic hashing helpers: canonical form, stability and error paths."""

from __future__ import annotations

import pytest

from hypoarena.errors import SchemaError, ValidationError
from hypoarena.ids import (
    DEFAULT_HASH_LENGTH,
    canonical_json,
    content_hash,
    hash_parts,
    short_hash,
    stable_hash,
)


def test_canonical_json_sorts_keys_and_drops_spaces() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'


def test_canonical_json_is_insensitive_to_insertion_order() -> None:
    first = canonical_json({"x": 1, "y": {"n": 2, "m": 3}})
    second = canonical_json({"y": {"m": 3, "n": 2}, "x": 1})
    assert first == second


def test_canonical_json_escapes_non_ascii_for_byte_stability() -> None:
    assert canonical_json({"k": "é"}) == '{"k":"\\u00e9"}'


def test_canonical_json_rejects_nan_and_unserializable_values() -> None:
    with pytest.raises(SchemaError):
        canonical_json(float("nan"))
    with pytest.raises(SchemaError):
        canonical_json({"k": object()})


def test_stable_hash_is_deterministic_and_length_bounded() -> None:
    digest = stable_hash("hypoarena")
    assert digest == stable_hash("hypoarena")
    assert len(digest) == DEFAULT_HASH_LENGTH
    assert len(stable_hash("hypoarena", length=64)) == 64


def test_stable_hash_separates_different_payloads() -> None:
    assert stable_hash("claim a") != stable_hash("claim b")


def test_stable_hash_rejects_bad_length_and_payload() -> None:
    with pytest.raises(ValidationError):
        stable_hash("x", length=7)
    with pytest.raises(ValidationError):
        stable_hash("x", length=65)
    with pytest.raises(SchemaError):
        stable_hash(b"bytes")  # type: ignore[arg-type]


def test_content_hash_ignores_key_order() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_content_hash_separates_values_and_types() -> None:
    assert content_hash({"a": 1}) != content_hash({"a": "1"})
    assert content_hash([1, 2]) != content_hash([2, 1])


def test_content_hash_accepts_a_custom_length() -> None:
    assert len(content_hash({"a": 1}, length=32)) == 32


def test_short_hash_defaults_to_eight_characters() -> None:
    assert len(short_hash("display")) == 8
    assert short_hash("display") == short_hash("display")


def test_hash_parts_keeps_part_boundaries() -> None:
    assert hash_parts("ab") != hash_parts("a", "b")
    assert hash_parts("a", "b") == hash_parts("a", "b")


def test_hash_parts_is_order_sensitive() -> None:
    assert hash_parts("a", "b") != hash_parts("b", "a")


def test_hash_parts_rejects_unserializable_parts() -> None:
    with pytest.raises(SchemaError):
        hash_parts({"ok": 1}, object())
