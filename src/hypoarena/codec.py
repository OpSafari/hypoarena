"""Strict helpers for converting JSON payloads into validated dataclasses.

Every ``from_dict`` in this package goes through these functions, so malformed
input fails with a :class:`~hypoarena.errors.SchemaError` that names the
offending field instead of raising an opaque ``TypeError`` deep inside a
constructor. Coercion is deliberately narrow: types are checked, not guessed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, TypeVar

from hypoarena.errors import SchemaError
from hypoarena.ids import canonical_json

EnumT = TypeVar("EnumT", bound=Enum)


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


def present_value(mapping: Mapping[str, Any], key: str, *, field: str) -> Any:
    """Return ``mapping[key]`` or raise a schema error naming the missing key."""
    if key not in mapping:
        raise SchemaError(f"{field} is missing '{key}'", field=field, key=key)
    return mapping[key]


def check_range(
    value: float,
    *,
    field: str,
    key: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate inclusive numeric bounds, reporting which side was violated."""
    if minimum is not None and value < minimum:
        raise SchemaError(
            f"{field}.{key} is below the minimum",
            field=field,
            key=key,
            value=value,
            minimum=minimum,
        )
    if maximum is not None and value > maximum:
        raise SchemaError(
            f"{field}.{key} is above the maximum",
            field=field,
            key=key,
            value=value,
            maximum=maximum,
        )
    return value


def require_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read ``key`` as an integer within optional inclusive bounds.

    Booleans are rejected: ``True`` is an ``int`` subclass in Python, and
    accepting it would let ``{"start": true}`` pass as offset 1.
    """
    value = present_value(mapping, key, field=field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(
            f"{field}.{key} must be an integer",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    check_range(value, field=field, key=key, minimum=minimum, maximum=maximum)
    return value


def require_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read ``key`` as a finite float within optional inclusive bounds."""
    value = present_value(mapping, key, field=field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(
            f"{field}.{key} must be a number",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError(
            f"{field}.{key} must be finite", field=field, key=key, value=value
        )
    check_range(number, field=field, key=key, minimum=minimum, maximum=maximum)
    return number


def require_bool(mapping: Mapping[str, Any], key: str, *, field: str) -> bool:
    """Read ``key`` as a real boolean; no truthiness coercion is performed."""
    value = present_value(mapping, key, field=field)
    if not isinstance(value, bool):
        raise SchemaError(
            f"{field}.{key} must be a boolean",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    return value


def require_enum(
    mapping: Mapping[str, Any],
    key: str,
    enum_type: type[EnumT],
    *,
    field: str,
) -> EnumT:
    """Read ``key`` as a member of ``enum_type``, listing valid values on error."""
    value = present_value(mapping, key, field=field)
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise SchemaError(
            f"{field}.{key} must be a string",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    try:
        return enum_type(value)
    except ValueError:
        allowed = [member.value for member in enum_type]
        raise SchemaError(
            f"{field}.{key} is not a valid {enum_type.__name__}",
            field=field,
            key=key,
            got=value,
            allowed=allowed,
        ) from None


def require_str_tuple(
    mapping: Mapping[str, Any],
    key: str,
    *,
    field: str,
    allow_empty: bool = False,
    minimum_items: int = 0,
    maximum_items: int | None = None,
) -> tuple[str, ...]:
    """Read ``key`` as a list of non-blank strings and return it as a tuple."""
    value = present_value(mapping, key, field=field)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(
            f"{field}.{key} must be a list",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    items = tuple(
        require_str({"v": item}, "v", field=f"{field}.{key}", allow_empty=allow_empty)
        for item in value
    )
    if len(items) < minimum_items:
        raise SchemaError(
            f"{field}.{key} needs at least {minimum_items} item(s)",
            field=field,
            key=key,
            count=len(items),
        )
    if maximum_items is not None and len(items) > maximum_items:
        raise SchemaError(
            f"{field}.{key} accepts at most {maximum_items} item(s)",
            field=field,
            key=key,
            count=len(items),
        )
    return items


def optional_str(
    mapping: Mapping[str, Any],
    key: str,
    *,
    field: str,
    allow_empty: bool = False,
) -> str | None:
    """Read ``key`` as a string, accepting ``None`` and absent keys."""
    if key not in mapping or mapping[key] is None:
        return None
    return require_str(mapping, key, field=field, allow_empty=allow_empty)


def dumps_line(value: object) -> str:
    """Serialize ``value`` as one canonical JSON line ending in a newline.

    Keys are sorted and separators are fixed, so equal payloads always produce
    byte-identical lines — the property that makes JSONL artifacts diffable and
    lets checkpointed runs be compared byte for byte.
    """
    return canonical_json(value) + "\n"


def loads_line(
    line: str, *, field: str, line_number: int | None = None
) -> dict[str, Any]:
    """Decode one JSONL line into a mapping, reporting the offending line."""
    stripped = line.strip()
    if not stripped:
        raise SchemaError(
            f"{field} line is empty", field=field, line_number=line_number
        )
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise SchemaError(
            f"{field} line is not valid JSON",
            field=field,
            line_number=line_number,
            reason=str(error),
        ) from None
    return require_mapping(payload, field=field)


def optional_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    """Read ``key`` as an integer, accepting ``None`` and absent keys."""
    if key not in mapping or mapping[key] is None:
        return None
    return require_int(mapping, key, field=field, minimum=minimum, maximum=maximum)


def optional_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Read ``key`` as a finite float, accepting ``None`` and absent keys."""
    if key not in mapping or mapping[key] is None:
        return None
    return require_float(mapping, key, field=field, minimum=minimum, maximum=maximum)


def require_mapping_list(
    mapping: Mapping[str, Any], key: str, *, field: str
) -> list[dict[str, Any]]:
    """Read ``key`` as a list of JSON objects, decoding each entry strictly."""
    value = present_value(mapping, key, field=field)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(
            f"{field}.{key} must be a list",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    return [
        require_mapping(item, field=f"{field}.{key}[{index}]")
        for index, item in enumerate(value)
    ]


def require_float_list(
    mapping: Mapping[str, Any], key: str, *, field: str
) -> tuple[float, ...]:
    """Read ``key`` as a list of finite numbers."""
    value = present_value(mapping, key, field=field)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(
            f"{field}.{key} must be a list",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    return tuple(
        require_float({"v": item}, "v", field=f"{field}.{key}[{index}]")
        for index, item in enumerate(value)
    )


def require_enum_list(
    mapping: Mapping[str, Any],
    key: str,
    enum_type: type[EnumT],
    *,
    field: str,
) -> tuple[EnumT, ...]:
    """Read ``key`` as a list of enum values."""
    value = present_value(mapping, key, field=field)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(
            f"{field}.{key} must be a list",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    return tuple(
        require_enum({"v": item}, "v", enum_type, field=f"{field}.{key}[{index}]")
        for index, item in enumerate(value)
    )
