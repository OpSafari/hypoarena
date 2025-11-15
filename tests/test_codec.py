"""Strict decoding helpers: accepted shapes and every rejection path."""

from __future__ import annotations

from enum import StrEnum

import pytest

from hypoarena.codec import (
    check_schema_version,
    dumps_line,
    loads_line,
    optional_float,
    optional_int,
    optional_str,
    present_value,
    reject_unknown_keys,
    require_bool,
    require_enum,
    require_float,
    require_int,
    require_keys,
    require_mapping,
    require_mapping_list,
    require_str,
    require_str_tuple,
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


def test_require_int_accepts_in_range_values() -> None:
    assert require_int({"start": 0}, "start", field="span") == 0
    assert (
        require_int({"start": 12}, "start", field="span", minimum=0, maximum=100) == 12
    )


def test_require_int_rejects_bools_floats_and_strings() -> None:
    for value in (True, 1.5, "3", None):
        with pytest.raises(SchemaError, match="must be an integer"):
            require_int({"start": value}, "start", field="span")


def test_require_int_reports_bound_violations() -> None:
    with pytest.raises(SchemaError) as low:
        require_int({"start": -1}, "start", field="span", minimum=0)
    assert low.value.details["minimum"] == 0
    with pytest.raises(SchemaError) as high:
        require_int({"end": 101}, "end", field="span", minimum=0, maximum=100)
    assert high.value.details["maximum"] == 100


def test_require_float_coerces_integers_and_rejects_nan() -> None:
    assert require_float({"strength": 1}, "strength", field="evidence") == 1.0
    with pytest.raises(SchemaError, match="must be finite"):
        require_float({"strength": float("nan")}, "strength", field="evidence")
    with pytest.raises(SchemaError, match="must be a number"):
        require_float({"strength": "0.5"}, "strength", field="evidence")


def test_require_float_enforces_inclusive_bounds() -> None:
    assert require_float({"s": 0.0}, "s", field="e", minimum=0.0, maximum=1.0) == 0.0
    assert require_float({"s": 1.0}, "s", field="e", minimum=0.0, maximum=1.0) == 1.0
    with pytest.raises(SchemaError):
        require_float({"s": 1.01}, "s", field="e", minimum=0.0, maximum=1.0)


def test_require_bool_rejects_truthy_substitutes() -> None:
    assert require_bool({"flag": False}, "flag", field="cfg") is False
    for value in (1, "true", None):
        with pytest.raises(SchemaError, match="must be a boolean"):
            require_bool({"flag": value}, "flag", field="cfg")


def test_present_value_names_the_missing_key() -> None:
    with pytest.raises(SchemaError) as info:
        present_value({}, "strength", field="evidence")
    assert info.value.details["key"] == "strength"


class Colour(StrEnum):
    RED = "red"
    BLUE = "blue"


def test_require_enum_accepts_values_and_members() -> None:
    assert require_enum({"c": "red"}, "c", Colour, field="cell") is Colour.RED
    assert require_enum({"c": Colour.BLUE}, "c", Colour, field="cell") is Colour.BLUE


def test_require_enum_rejects_unknown_values_and_lists_allowed() -> None:
    with pytest.raises(SchemaError) as info:
        require_enum({"c": "green"}, "c", Colour, field="cell")
    assert info.value.details["allowed"] == ["red", "blue"]


def test_require_enum_rejects_non_strings() -> None:
    with pytest.raises(SchemaError, match="must be a string"):
        require_enum({"c": 1}, "c", Colour, field="cell")


def test_require_str_tuple_returns_a_tuple_of_strings() -> None:
    assert require_str_tuple({"v": ["a", "b"]}, "v", field="claim") == ("a", "b")
    assert require_str_tuple({"v": []}, "v", field="claim") == ()


def test_require_str_tuple_rejects_scalars_and_blank_items() -> None:
    with pytest.raises(SchemaError, match="must be a list"):
        require_str_tuple({"v": "ab"}, "v", field="claim")
    with pytest.raises(SchemaError, match="blank"):
        require_str_tuple({"v": ["a", "  "]}, "v", field="claim")


def test_require_str_tuple_enforces_item_bounds() -> None:
    with pytest.raises(SchemaError, match="at least 1"):
        require_str_tuple({"v": []}, "v", field="claim", minimum_items=1)
    with pytest.raises(SchemaError, match="at most 2"):
        require_str_tuple({"v": ["a", "b", "c"]}, "v", field="claim", maximum_items=2)


def test_optional_str_treats_absent_and_null_as_none() -> None:
    assert optional_str({}, "note", field="claim") is None
    assert optional_str({"note": None}, "note", field="claim") is None
    assert optional_str({"note": "x"}, "note", field="claim") == "x"
    with pytest.raises(SchemaError, match="blank"):
        optional_str({"note": " "}, "note", field="claim")


def test_dumps_line_is_canonical_and_newline_terminated() -> None:
    assert dumps_line({"b": 1, "a": 2}) == '{"a":2,"b":1}\n'


def test_dumps_line_is_stable_across_key_insertion_order() -> None:
    assert dumps_line({"a": 1, "b": 2}) == dumps_line({"b": 2, "a": 1})


def test_dumps_line_rejects_non_serializable_values() -> None:
    with pytest.raises(SchemaError):
        dumps_line({"fn": lambda: None})


def test_loads_line_roundtrips_a_dumped_payload() -> None:
    payload = {"claim_id": "clm_0123456789ab", "strength": 0.5}
    assert loads_line(dumps_line(payload), field="claim") == payload


def test_loads_line_reports_line_numbers_and_reasons() -> None:
    with pytest.raises(SchemaError) as blank:
        loads_line("   \n", field="evidence", line_number=4)
    assert blank.value.details["line_number"] == 4
    with pytest.raises(SchemaError, match="not valid JSON"):
        loads_line("{oops", field="evidence", line_number=7)
    with pytest.raises(SchemaError, match="must be a JSON object"):
        loads_line("[1, 2]", field="evidence")


def test_optional_numerics_accept_null_and_absent_keys() -> None:
    for mapping in ({}, {"n": None}):
        assert optional_int(mapping, "n", field="f") is None
        assert optional_float(mapping, "n", field="f") is None


def test_optional_numerics_validate_when_present() -> None:
    assert optional_int({"n": 3}, "n", field="f") == 3
    assert optional_float({"n": 1}, "n", field="f") == 1.0
    with pytest.raises(SchemaError, match="below the minimum"):
        optional_int({"n": -1}, "n", field="f", minimum=0)
    with pytest.raises(SchemaError, match="must be an integer"):
        optional_int({"n": True}, "n", field="f")


def test_require_mapping_list_decodes_entries_with_positions() -> None:
    payload = {"items": [{"a": 1}, {"b": 2}]}
    assert require_mapping_list(payload, "items", field="claim") == [{"a": 1}, {"b": 2}]
    with pytest.raises(SchemaError) as info:
        require_mapping_list({"items": [{"a": 1}, 5]}, "items", field="claim")
    assert info.value.details["field"] == "claim.items[1]"


def test_require_mapping_list_rejects_scalars() -> None:
    with pytest.raises(SchemaError, match="must be a list"):
        require_mapping_list({"items": "nope"}, "items", field="claim")
    with pytest.raises(SchemaError, match="missing"):
        require_mapping_list({}, "items", field="claim")
