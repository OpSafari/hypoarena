"""Conversion between schema records and JSON-ready dictionaries.

Serialization lives beside the schema rather than inside it: the dataclasses stay
free of codec imports, and the wire format has a single home that golden tests
can pin. Conventions:

* ``*_to_dict`` emits only JSON primitives, lists and dicts, keeping ``None``
  values explicit so payloads stay key-stable across versions;
* ``*_from_dict`` rejects unknown keys and wrong schema versions, then defers
  semantic validation to the dataclass constructors;
* only top-level records (claims, evidence) carry ``schema_version``.
"""

from __future__ import annotations

from typing import Any

from hypoarena.codec import (
    optional_int,
    optional_str,
    reject_unknown_keys,
    require_int,
    require_mapping,
    require_mapping_list,
    require_str,
    require_str_tuple,
)
from hypoarena.schema import Citation, Provenance, Scope

SCOPE_KEYS = ("population", "conditions")
CITATION_KEYS = ("document_id", "start", "end", "quote")
PROVENANCE_KEYS = (
    "origin",
    "agent_id",
    "generation",
    "parents",
    "seed",
    "corpus_hash",
    "notes",
)


def scope_to_dict(scope: Scope) -> dict[str, Any]:
    """Encode a scope as a JSON object."""
    return {"population": scope.population, "conditions": list(scope.conditions)}


def scope_from_dict(payload: object, *, field: str = "scope") -> Scope:
    """Decode a scope, rejecting unknown keys."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, SCOPE_KEYS, field=field)
    return Scope(
        population=require_str(mapping, "population", field=field),
        conditions=require_str_tuple(mapping, "conditions", field=field),
    )


def citation_to_dict(citation: Citation) -> dict[str, Any]:
    """Encode a citation span as a JSON object."""
    return {
        "document_id": citation.document_id,
        "start": citation.start,
        "end": citation.end,
        "quote": citation.quote,
    }


def citation_from_dict(payload: object, *, field: str = "citation") -> Citation:
    """Decode a citation span with non-negative, ordered offsets."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, CITATION_KEYS, field=field)
    return Citation(
        document_id=require_str(mapping, "document_id", field=field),
        start=require_int(mapping, "start", field=field, minimum=0),
        end=require_int(mapping, "end", field=field, minimum=1),
        quote=require_str(mapping, "quote", field=field),
    )


def provenance_to_dict(provenance: Provenance) -> dict[str, Any]:
    """Encode provenance, keeping unset optional fields as explicit nulls."""
    return {
        "origin": provenance.origin,
        "agent_id": provenance.agent_id,
        "generation": provenance.generation,
        "parents": list(provenance.parents),
        "seed": provenance.seed,
        "corpus_hash": provenance.corpus_hash,
        "notes": provenance.notes,
    }


def provenance_from_dict(payload: object, *, field: str = "provenance") -> Provenance:
    """Decode provenance; cross-field rules are enforced by the constructor."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, PROVENANCE_KEYS, field=field)
    return Provenance(
        origin=require_str(mapping, "origin", field=field),
        agent_id=optional_str(mapping, "agent_id", field=field),
        generation=require_int(mapping, "generation", field=field, minimum=0),
        parents=require_str_tuple(mapping, "parents", field=field),
        seed=optional_int(mapping, "seed", field=field),
        corpus_hash=optional_str(mapping, "corpus_hash", field=field),
        notes=optional_str(mapping, "notes", field=field),
    )


def citation_list_from_payload(
    mapping: dict[str, Any], *, field: str, key: str = "citations"
) -> tuple[Citation, ...]:
    """Decode the citation list of a top-level record."""
    return tuple(
        citation_from_dict(item, field=f"{field}.{key}[{index}]")
        for index, item in enumerate(require_mapping_list(mapping, key, field=field))
    )
