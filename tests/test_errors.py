"""Error hierarchy: messages, structured details and exit codes."""

from __future__ import annotations

import pytest

from hypoarena.errors import (
    DuplicateIdError,
    GraphInvariantError,
    HypoArenaError,
    SchemaError,
    UnknownReferenceError,
    ValidationError,
)


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


def test_duplicate_id_error_exposes_identifier_and_kind() -> None:
    error = DuplicateIdError("clm_01", "claim")
    assert error.identifier == "clm_01"
    assert error.kind == "claim"
    assert error.code == "duplicate_id"
    assert "identifier='clm_01'" in str(error)


def test_unknown_reference_error_records_the_optional_owner() -> None:
    bare = UnknownReferenceError("doc_9", "document")
    owned = UnknownReferenceError("doc_9", "document", owner="clm_01")
    assert bare.owner is None
    assert owned.owner == "clm_01"
    assert "owner" not in str(bare)
    assert "owner='clm_01'" in str(owned)


def test_graph_invariant_error_is_a_validation_error() -> None:
    assert issubclass(GraphInvariantError, ValidationError)
    assert GraphInvariantError("self loop", node="clm_01").exit_code == 2
