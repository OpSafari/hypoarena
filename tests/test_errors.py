"""Error hierarchy: messages, structured details and exit codes."""

from __future__ import annotations

import pytest

from hypoarena.errors import (
    ERROR_CODES,
    AdapterError,
    ArtifactError,
    ConfigError,
    CorpusError,
    DuplicateIdError,
    GraphInvariantError,
    HypoArenaError,
    ReplayExhaustedError,
    SchemaError,
    SecretLeakError,
    SpanNotFoundError,
    TransportError,
    UnknownReferenceError,
    ValidationError,
    error_for_code,
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


def test_corpus_errors_use_their_own_exit_code() -> None:
    assert SpanNotFoundError("missing span", span="s1").exit_code == 3
    assert issubclass(SpanNotFoundError, CorpusError)


def test_replay_exhausted_is_an_adapter_error() -> None:
    error = ReplayExhaustedError("no recorded response", agent="scripted")
    assert isinstance(error, AdapterError)
    assert error.exit_code == 4


def test_transport_error_keeps_status_and_attempt_count() -> None:
    error = TransportError("upstream unavailable", status=503, attempts=3)
    assert error.status == 503
    assert error.attempts == 3
    assert "status=503" in str(error)


def test_transport_error_defaults_are_inert() -> None:
    error = TransportError("connection reset")
    assert error.status is None
    assert error.attempts == 0


def test_config_and_artifact_errors_have_distinct_exit_codes() -> None:
    codes = {ConfigError("c").exit_code, ArtifactError("a").exit_code}
    assert codes == {5, 6}


def test_secret_leak_is_an_artifact_error() -> None:
    assert issubclass(SecretLeakError, ArtifactError)
    assert SecretLeakError("refused", field="api_key").code == "secret_leak"


def test_every_registered_code_maps_back_to_its_class() -> None:
    for code, cls in ERROR_CODES.items():
        assert cls.code == code
        assert error_for_code(code) is cls


def test_registry_codes_are_unique() -> None:
    classes = list(ERROR_CODES.values())
    assert len(classes) == len(set(classes))


def test_unknown_code_lookup_raises_a_structured_error() -> None:
    with pytest.raises(UnknownReferenceError) as info:
        error_for_code("not_a_real_code")
    assert info.value.kind == "error code"


def test_golden_registry_snapshot() -> None:
    assert sorted(ERROR_CODES) == [
        "adapter_error",
        "artifact_error",
        "config_error",
        "corpus_error",
        "duplicate_id",
        "graph_invariant",
        "hypoarena_error",
        "replay_exhausted",
        "schema_error",
        "secret_leak",
        "span_not_found",
        "transport_error",
        "unknown_reference",
        "validation_error",
    ]
