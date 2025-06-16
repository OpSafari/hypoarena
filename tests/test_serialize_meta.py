"""Shared JSONL header helpers used by every document type."""

from __future__ import annotations

import json

import pytest

from hypoarena.errors import SchemaError
from hypoarena.serialize import check_document_meta, meta_line

COUNTS = {"documents": 2, "characters": 120}
SIGNATURE = "0123456789abcdef"


def test_meta_line_is_canonical_and_sorted() -> None:
    line = meta_line(COUNTS, SIGNATURE)
    payload = json.loads(line)
    assert line == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    assert payload == {
        "record": "meta",
        "schema_version": "1.0",
        "counts": {"characters": 120, "documents": 2},
        "signature": SIGNATURE,
    }


def test_check_accepts_a_matching_header() -> None:
    meta = json.loads(meta_line(COUNTS, SIGNATURE))
    assert (
        check_document_meta(
            meta,
            counts=COUNTS,
            signature=SIGNATURE,
            kind="corpus",
            count_keys=("documents", "characters"),
        )
        is None
    )


def test_check_ignores_a_missing_header() -> None:
    assert (
        check_document_meta(
            None, counts={}, signature="x", kind="corpus", count_keys=()
        )
        is None
    )


def test_check_reports_count_and_signature_drift() -> None:
    meta = json.loads(meta_line(COUNTS, SIGNATURE))
    with pytest.raises(SchemaError, match="counts disagree") as info:
        check_document_meta(
            meta,
            counts={"documents": 3, "characters": 120},
            signature=SIGNATURE,
            kind="corpus",
            count_keys=("documents", "characters"),
        )
    assert info.value.details["expected"]["documents"] == 2
    with pytest.raises(SchemaError, match="signature disagrees"):
        check_document_meta(
            meta,
            counts=COUNTS,
            signature="f" * 16,
            kind="corpus",
            count_keys=("documents", "characters"),
        )


def test_check_rejects_foreign_count_keys() -> None:
    meta = json.loads(meta_line({**COUNTS, "extra": 1}, SIGNATURE))
    with pytest.raises(SchemaError, match="unknown keys"):
        check_document_meta(
            meta,
            counts=COUNTS,
            signature=SIGNATURE,
            kind="corpus",
            count_keys=("documents", "characters"),
        )
