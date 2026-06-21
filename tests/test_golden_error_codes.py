"""Golden over the error-code registry: codes, classes and exit codes.

The CLI maps a raised error's ``exit_code`` to the process exit status, so the
code -> (class, exit_code) table is a public contract. Pinning it means a new
error category, a changed exit code or a renamed code is a deliberate change.
"""

from __future__ import annotations

from hypoarena.errors import ERROR_CODES, HypoArenaError, error_for_code

EXPECTED: dict[str, tuple[str, int]] = {
    "hypoarena_error": ("HypoArenaError", 1),
    "validation_error": ("ValidationError", 2),
    "schema_error": ("SchemaError", 2),
    "duplicate_id": ("DuplicateIdError", 2),
    "unknown_reference": ("UnknownReferenceError", 2),
    "graph_invariant": ("GraphInvariantError", 2),
    "corpus_error": ("CorpusError", 3),
    "span_not_found": ("SpanNotFoundError", 3),
    "adapter_error": ("AdapterError", 4),
    "replay_exhausted": ("ReplayExhaustedError", 4),
    "transport_error": ("TransportError", 4),
    "config_error": ("ConfigError", 5),
    "artifact_error": ("ArtifactError", 6),
    "secret_leak": ("SecretLeakError", 6),
}


def test_registry_matches_the_golden_table() -> None:
    actual = {code: (cls.__name__, cls.exit_code) for code, cls in ERROR_CODES.items()}
    assert actual == EXPECTED


def test_every_code_resolves_back_to_its_class() -> None:
    for code, cls in ERROR_CODES.items():
        assert error_for_code(code) is cls


def test_every_registered_class_is_a_hypoarena_error() -> None:
    assert all(issubclass(cls, HypoArenaError) for cls in ERROR_CODES.values())


def test_exit_codes_stay_within_the_documented_range() -> None:
    assert {cls.exit_code for cls in ERROR_CODES.values()} == {1, 2, 3, 4, 5, 6}


def test_each_class_code_attribute_agrees_with_its_registry_key() -> None:
    for code, cls in ERROR_CODES.items():
        assert cls.code == code
