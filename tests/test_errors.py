"""Error hierarchy: messages, structured details and exit codes."""

from __future__ import annotations

import pytest

from hypoarena.errors import HypoArenaError, SchemaError, ValidationError


def test_base_error_keeps_message_and_details() -> None:
    error = HypoArenaError("boom", stage="generate", index=3)
    assert error.message == "boom"
    assert error.details == {"stage": "generate", "index": 3}


def test_str_renders_details_in_sorted_key_order() -> None:
    error = ValidationError("bad value", zeta=1, alpha=2)
    assert str(error) == "bad value (alpha=2, zeta=1)"


def test_str_without_details_is_the_bare_message() -> None:
    assert str(ValidationError("bad value")) == "bad value"


def test_validation_and_schema_errors_share_a_category() -> None:
    assert issubclass(ValidationError, HypoArenaError)
    assert issubclass(SchemaError, ValidationError)


def test_categories_have_distinct_exit_codes() -> None:
    assert HypoArenaError.exit_code != ValidationError.exit_code
    assert isinstance(SchemaError("x").exit_code, int)


def test_errors_can_be_raised_and_caught_by_base_class() -> None:
    with pytest.raises(HypoArenaError):
        raise SchemaError("unknown key", key="surprise")
