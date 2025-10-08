"""Deterministic identifiers and content hashes.

Every helper is a pure function of its arguments: no clock, no random state and
no iteration order leaking into the result. Digests are therefore stable across
processes and machines, which is what lets golden tests pin serialized hashes.
"""

from __future__ import annotations

import hashlib
import json
import re

from hypoarena.errors import SchemaError, ValidationError

DEFAULT_HASH_LENGTH = 16
MIN_HASH_LENGTH = 8
MAX_HASH_LENGTH = 64
ID_PREFIX_PATTERN = re.compile(r"[a-z]{2,8}")
ID_PATTERN = re.compile(r"[a-z]{2,8}_[0-9a-f]{8,64}")
DEFAULT_ID_LENGTH = 12


def canonical_json(value: object) -> str:
    """Serialize ``value`` to JSON with sorted keys and no insignificant spaces.

    Non-JSON values (and NaN/inf floats, which have no portable representation)
    raise :class:`~hypoarena.errors.SchemaError` instead of silently producing a
    digest that could not be reproduced elsewhere.
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SchemaError(
            "value is not canonically serializable", reason=str(error)
        ) from error


def stable_hash(payload: str, *, length: int = DEFAULT_HASH_LENGTH) -> str:
    """Return the first ``length`` hex characters of the SHA-256 of ``payload``."""
    if not isinstance(payload, str):
        raise SchemaError("hash payload must be a string", got=type(payload).__name__)
    if not MIN_HASH_LENGTH <= length <= MAX_HASH_LENGTH:
        raise ValidationError(
            "hash length out of range",
            length=length,
            minimum=MIN_HASH_LENGTH,
            maximum=MAX_HASH_LENGTH,
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def content_hash(value: object, *, length: int = DEFAULT_HASH_LENGTH) -> str:
    """Hash any JSON-serializable value through its canonical form."""
    return stable_hash(canonical_json(value), length=length)


def short_hash(text: str, *, length: int = 8) -> str:
    """Return a short digest meant for display, not for collision resistance."""
    return stable_hash(text, length=length)


def hash_parts(*parts: object, length: int = DEFAULT_HASH_LENGTH) -> str:
    """Hash an ordered sequence of parts without merging their boundaries.

    Each part is canonicalized separately first, so ``hash_parts("ab")`` and
    ``hash_parts("a", "b")`` produce different digests.
    """
    return content_hash([canonical_json(part) for part in parts], length=length)


def is_valid_id(value: str) -> bool:
    """True when ``value`` looks like ``<prefix>_<hex digest>``."""
    return isinstance(value, str) and ID_PATTERN.fullmatch(value) is not None


def make_id(prefix: str, *parts: object, length: int = DEFAULT_ID_LENGTH) -> str:
    """Build a deterministic identifier from ``prefix`` and the given parts."""
    if not ID_PREFIX_PATTERN.fullmatch(prefix):
        raise ValidationError(
            "id prefix must be 2-8 lowercase ascii letters", prefix=prefix
        )
    identifier = f"{prefix}_{hash_parts(*parts, length=length)}"
    if not is_valid_id(identifier):  # pragma: no cover - defensive
        raise ValidationError("generated id is malformed", identifier=identifier)
    return identifier
