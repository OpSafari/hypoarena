"""Golden digests for the deterministic hashing helpers.

The literals below were produced by running the helpers once and reviewing the
output. They pin the digest algorithm (SHA-256 over canonical JSON), the default
truncation lengths and the identifier layout, so that any change to hashing —
which would silently invalidate every stored provenance hash — has to be made on
purpose.
"""

from __future__ import annotations

from hypoarena.ids import (
    canonical_json,
    content_hash,
    hash_parts,
    make_id,
    short_hash,
    stable_hash,
)


def test_canonical_json_golden() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'


def test_stable_hash_golden() -> None:
    assert stable_hash("hypoarena") == "f5cd71c4d9e3a16f"
    assert stable_hash("hypoarena", length=8) == "f5cd71c4"


def test_content_hash_golden() -> None:
    assert content_hash({"b": 1, "a": [2, 3]}) == "2aae3bfa906be395"
    nested = {
        "claim": "x increases y",
        "spans": [{"doc": "doc_1", "start": 3, "end": 9}],
    }
    assert content_hash(nested) == "2008cc08346227fe"


def test_hash_parts_golden_keeps_boundaries() -> None:
    assert hash_parts("a", "b") == "deedf2289a6b11c2"
    assert hash_parts("ab") == "5bb6b851df887c96"


def test_short_hash_golden() -> None:
    assert short_hash("display") == "dfbb889c"


def test_make_id_golden() -> None:
    assert make_id("clm", "x increases y", "scope=cell") == "clm_9b8bf4144eb5"
    assert make_id("evd", "doc_1", 12, 34, length=20) == "evd_eda2b0021c5932140b3a"
