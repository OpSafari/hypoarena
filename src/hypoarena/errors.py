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


class DuplicateIdError(ValidationError):
    """An identifier was reused for a second node or record."""

    code = "duplicate_id"

    def __init__(self, identifier: str, kind: str) -> None:
        super().__init__(f"duplicate {kind} id", identifier=identifier, kind=kind)
        self.identifier = identifier
        self.kind = kind


class UnknownReferenceError(ValidationError):
    """An edge or citation pointed at an identifier that does not exist."""

    code = "unknown_reference"

    def __init__(self, identifier: str, kind: str, *, owner: str | None = None) -> None:
        details: dict[str, object] = {"identifier": identifier, "kind": kind}
        if owner is not None:
            details["owner"] = owner
        super().__init__(f"unknown {kind} reference", **details)
        self.identifier = identifier
        self.kind = kind
        self.owner = owner


class GraphInvariantError(ValidationError):
    """A graph operation would have broken a structural invariant."""

    code = "graph_invariant"


class CorpusError(HypoArenaError):
    """A corpus operation was asked for something the corpus cannot provide."""

    code = "corpus_error"
    exit_code = 3


class SpanNotFoundError(CorpusError):
    """A citation referenced a document span that is not in the corpus."""

    code = "span_not_found"


class AdapterError(HypoArenaError):
    """An agent adapter could not produce a usable response."""

    code = "adapter_error"
    exit_code = 4


class ReplayExhaustedError(AdapterError):
    """A replay adapter ran out of recorded responses."""

    code = "replay_exhausted"


class TransportError(AdapterError):
    """An HTTP adapter failed after exhausting its retry budget."""

    code = "transport_error"

    def __init__(
        self, message: str, *, status: int | None = None, attempts: int = 0
    ) -> None:
        super().__init__(message, status=status, attempts=attempts)
        self.status = status
        self.attempts = attempts
