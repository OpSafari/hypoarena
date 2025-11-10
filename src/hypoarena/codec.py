"""Strict helpers for converting JSON payloads into validated dataclasses.

Every ``from_dict`` in this package goes through these functions, so malformed
input fails with a :class:`~hypoarena.errors.SchemaError` that names the
offending field instead of raising an opaque ``TypeError`` deep inside a
constructor. Coercion is deliberately narrow: types are checked, not guessed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hypoarena.errors import SchemaError


def require_mapping(value: object, *, field: str) -> dict[str, Any]:
    """Return ``value`` as a plain dict or raise a schema error."""
    if not isinstance(value, Mapping):
        raise SchemaError(
            f"{field} must be a JSON object", field=field, got=type(value).__name__
        )
    return dict(value)


def require_str(
    mapping: Mapping[str, Any],
    key: str,
    *,
    field: str,
    allow_empty: bool = False,
) -> str:
    """Read ``key`` as a non-empty string unless ``allow_empty`` is set."""
    if key not in mapping:
        raise SchemaError(f"{field} is missing '{key}'", field=field, key=key)
    value = mapping[key]
    if not isinstance(value, str):
        raise SchemaError(
            f"{field}.{key} must be a string",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    if not allow_empty and not value.strip():
        raise SchemaError(f"{field}.{key} must not be blank", field=field, key=key)
    return value


def require_keys(
    mapping: Mapping[str, Any], required: Sequence[str], *, field: str
) -> None:
    """Raise when any of the ``required`` keys is absent from ``mapping``."""
    missing = [key for key in required if key not in mapping]
    if missing:
        raise SchemaError(
            f"{field} is missing required keys", field=field, missing=missing
        )


def reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: Sequence[str], *, field: str
) -> None:
    """Raise when ``mapping`` carries keys outside the ``allowed`` set.

    Rejecting unknown keys keeps old artifacts from being silently reinterpreted
    after a schema change: a new field must be introduced deliberately.
    """
    known = set(allowed)
    unknown = sorted(str(key) for key in mapping if key not in known)
    if unknown:
        raise SchemaError(
            f"{field} has unknown keys",
            field=field,
            unknown=unknown,
            allowed=sorted(known),
        )


def check_schema_version(
    mapping: Mapping[str, Any],
    *,
    field: str,
    expected: str,
    key: str = "schema_version",
) -> str:
    """Verify the payload's ``schema_version`` equals ``expected``."""
    found = require_str(mapping, key, field=field)
    if found != expected:
        raise SchemaError(
            f"{field} uses an unsupported schema version",
            field=field,
            found=found,
            expected=expected,
        )
    return found
