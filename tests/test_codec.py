"""Strict decoding helpers: accepted shapes and every rejection path."""

from __future__ import annotations

import pytest

from hypoarena.codec import (
    check_schema_version,
    reject_unknown_keys,
    require_keys,
    require_mapping,
    require_str,
)
from hypoarena.errors import SchemaError


def test_require_mapping_converts_and_copies() -> None:
    source = {"a": 1}
    result = require_mapping(source, field="claim")
    assert result == {"a": 1}
    result["b"] = 2
    assert "b" not in source


@pytest.mark.parametrize("value", [[], "text", 3, None, {1, 2}])
def test_require_mapping_rejects_non_objects(value: object) -> None:
    with pytest.raises(SchemaError) as info:
        require_mapping(value, field="claim")
    assert info.value.details["field"] == "claim"


def test_require_str_reads_present_strings() -> None:
    assert require_str({"name": "kinase"}, "name", field="scope") == "kinase"


def test_require_str_rejects_missing_blank_and_wrong_types() -> None:
    with pytest.raises(SchemaError, match="missing"):
        require_str({}, "name", field="scope")
    with pytest.raises(SchemaError, match="blank"):
        require_str({"name": "   "}, "name", field="scope")
    with pytest.raises(SchemaError, match="must be a string"):
        require_str({"name": 7}, "name", field="scope")


def test_require_str_can_allow_empty_values() -> None:
    assert require_str({"name": ""}, "name", field="scope", allow_empty=True) == ""


def test_require_keys_reports_every_missing_key() -> None:
    with pytest.raises(SchemaError) as info:
        require_keys({"a": 1}, ["a", "b", "c"], field="claim")
    assert info.value.details["missing"] == ["b", "c"]


def test_require_keys_accepts_a_superset() -> None:
    require_keys({"a": 1, "b": 2}, ["a"], field="claim")


def test_reject_unknown_keys_lists_offenders_sorted() -> None:
    with pytest.raises(SchemaError) as info:
        reject_unknown_keys({"a": 1, "zz": 2, "b": 3}, ["a", "b"], field="claim")
    assert info.value.details["unknown"] == ["zz"]


def test_reject_unknown_keys_accepts_a_subset() -> None:
    reject_unknown_keys({"a": 1}, ["a", "b"], field="claim")


def test_check_schema_version_accepts_and_rejects() -> None:
    payload = {"schema_version": "1.0"}
    assert check_schema_version(payload, field="claim", expected="1.0") == "1.0"
    with pytest.raises(SchemaError, match="unsupported schema version"):
        check_schema_version(payload, field="claim", expected="2.0")
    with pytest.raises(SchemaError, match="missing"):
        check_schema_version({}, field="claim", expected="1.0")
