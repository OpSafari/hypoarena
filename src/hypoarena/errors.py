"""Exception hierarchy for hypoarena.

Every error carries a stable machine-readable ``code`` and a process ``exit_code``
so the CLI can report failures without string matching, and tests can assert on
categories rather than messages.
"""

from __future__ import annotations


class HypoArenaError(Exception):
    """Base class for every error raised by this package."""

    code: str = "hypoarena_error"
    exit_code: int = 1

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, object] = dict(details)

    def __str__(self) -> str:
        if not self.details:
            return self.message
        rendered = ", ".join(
            f"{key}={value!r}" for key, value in sorted(self.details.items())
        )
        return f"{self.message} ({rendered})"


class ValidationError(HypoArenaError):
    """A value or structure failed a schema or invariant check."""

    code = "validation_error"
    exit_code = 2


class SchemaError(ValidationError):
    """A serialized payload did not match the expected schema."""

    code = "schema_error"
