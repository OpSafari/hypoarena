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

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from hypoarena.codec import (
    check_schema_version,
    dumps_line,
    loads_line,
    optional_float,
    optional_int,
    optional_str,
    present_value,
    reject_unknown_keys,
    require_enum,
    require_float,
    require_int,
    require_mapping,
    require_mapping_list,
    require_str,
    require_str_tuple,
)
from hypoarena.corpus import Corpus, Document
from hypoarena.errors import ArtifactError, SchemaError
from hypoarena.graph import ClaimEdge, GraphStats, HypothesisGraph
from hypoarena.schema import (
    SCHEMA_VERSION,
    Citation,
    Claim,
    ClaimRelation,
    Evidence,
    EvidencePolarity,
    PredictedRelation,
    Provenance,
    Scope,
)

SCOPE_KEYS = ("population", "conditions")
CLAIM_KEYS = (
    "schema_version",
    "claim_id",
    "statement",
    "subject",
    "object",
    "relation",
    "scope",
    "citations",
    "mechanism",
    "provenance",
)
EVIDENCE_KEYS = (
    "schema_version",
    "evidence_id",
    "statement",
    "polarity",
    "strength",
    "citations",
    "method",
    "provenance",
    "effect_size",
    "sample_size",
)
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


def claim_to_dict(claim: Claim) -> dict[str, Any]:
    """Encode a claim, stamping the current schema version."""
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_id": claim.claim_id,
        "statement": claim.statement,
        "subject": claim.subject,
        "object": claim.object,
        "relation": claim.relation.value,
        "scope": scope_to_dict(claim.scope),
        "citations": [citation_to_dict(citation) for citation in claim.citations],
        "mechanism": claim.mechanism,
        "provenance": provenance_to_dict(claim.provenance),
    }


def claim_from_dict(payload: object, *, field: str = "claim") -> Claim:
    """Decode a claim, rejecting unknown keys and foreign schema versions."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, CLAIM_KEYS, field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    return Claim(
        claim_id=require_str(mapping, "claim_id", field=field),
        statement=require_str(mapping, "statement", field=field),
        subject=require_str(mapping, "subject", field=field),
        object=require_str(mapping, "object", field=field),
        relation=require_enum(mapping, "relation", PredictedRelation, field=field),
        scope=scope_from_dict(
            present_value(mapping, "scope", field=field), field=f"{field}.scope"
        ),
        citations=citation_list_from_payload(mapping, field=field),
        mechanism=optional_str(mapping, "mechanism", field=field),
        provenance=provenance_from_dict(
            present_value(mapping, "provenance", field=field),
            field=f"{field}.provenance",
        ),
    )


def claim_to_line(claim: Claim) -> str:
    """Encode a claim as one canonical JSONL line."""
    return dumps_line(claim_to_dict(claim))


def claim_from_line(line: str, *, line_number: int | None = None) -> Claim:
    """Decode one JSONL line into a claim."""
    return claim_from_dict(loads_line(line, field="claim", line_number=line_number))


def evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    """Encode an evidence item, stamping the current schema version."""
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": evidence.evidence_id,
        "statement": evidence.statement,
        "polarity": evidence.polarity.value,
        "strength": evidence.strength,
        "citations": [citation_to_dict(citation) for citation in evidence.citations],
        "method": evidence.method,
        "provenance": provenance_to_dict(evidence.provenance),
        "effect_size": evidence.effect_size,
        "sample_size": evidence.sample_size,
    }


def evidence_from_dict(payload: object, *, field: str = "evidence") -> Evidence:
    """Decode an evidence item, rejecting unknown keys and foreign versions."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, EVIDENCE_KEYS, field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    return Evidence(
        evidence_id=require_str(mapping, "evidence_id", field=field),
        statement=require_str(mapping, "statement", field=field),
        polarity=require_enum(mapping, "polarity", EvidencePolarity, field=field),
        strength=require_float(
            mapping, "strength", field=field, minimum=0.0, maximum=1.0
        ),
        citations=citation_list_from_payload(mapping, field=field),
        method=require_str(mapping, "method", field=field),
        provenance=provenance_from_dict(
            present_value(mapping, "provenance", field=field),
            field=f"{field}.provenance",
        ),
        effect_size=optional_float(mapping, "effect_size", field=field),
        sample_size=optional_int(mapping, "sample_size", field=field, minimum=1),
    )


def evidence_to_line(evidence: Evidence) -> str:
    """Encode an evidence item as one canonical JSONL line."""
    return dumps_line(evidence_to_dict(evidence))


def evidence_from_line(line: str, *, line_number: int | None = None) -> Evidence:
    """Decode one JSONL line into an evidence item."""
    return evidence_from_dict(
        loads_line(line, field="evidence", line_number=line_number)
    )


EDGE_KEYS = ("source", "target", "relation", "note")
GRAPH_KEYS = ("schema_version", "claims", "evidence", "links", "edges")
LINK_KEYS = ("claim_id", "evidence_id")


def edge_to_dict(edge: ClaimEdge) -> dict[str, Any]:
    """Encode a typed claim relation."""
    return {
        "source": edge.source,
        "target": edge.target,
        "relation": edge.relation.value,
        "note": edge.note,
    }


def edge_from_dict(payload: object, *, field: str = "edge") -> ClaimEdge:
    """Decode a typed claim relation."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, EDGE_KEYS, field=field)
    return ClaimEdge(
        source=require_str(mapping, "source", field=field),
        target=require_str(mapping, "target", field=field),
        relation=require_enum(mapping, "relation", ClaimRelation, field=field),
        note=optional_str(mapping, "note", field=field),
    )


def graph_to_dict(graph: HypothesisGraph) -> dict[str, Any]:
    """Encode a whole graph, using canonical ordering for every collection."""
    return {
        "schema_version": SCHEMA_VERSION,
        "claims": [claim_to_dict(claim) for claim in graph.claims],
        "evidence": [evidence_to_dict(item) for item in graph.evidence_items],
        "links": [
            {"claim_id": claim_id, "evidence_id": evidence_id}
            for claim_id, evidence_id in graph.link_pairs()
        ],
        "edges": [edge_to_dict(edge) for edge in graph.edges],
    }


def graph_from_dict(payload: object, *, field: str = "graph") -> HypothesisGraph:
    """Decode a graph, validating every nested record and the final structure."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, GRAPH_KEYS, field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    graph = HypothesisGraph()
    for index, item in enumerate(require_mapping_list(mapping, "claims", field=field)):
        graph.add_claim(claim_from_dict(item, field=f"{field}.claims[{index}]"))
    for index, item in enumerate(
        require_mapping_list(mapping, "evidence", field=field)
    ):
        graph.add_evidence(evidence_from_dict(item, field=f"{field}.evidence[{index}]"))
    for index, item in enumerate(require_mapping_list(mapping, "links", field=field)):
        link_field = f"{field}.links[{index}]"
        reject_unknown_keys(item, LINK_KEYS, field=link_field)
        graph.link_evidence(
            require_str(item, "claim_id", field=link_field),
            require_str(item, "evidence_id", field=link_field),
        )
    for index, item in enumerate(require_mapping_list(mapping, "edges", field=field)):
        edge = edge_from_dict(item, field=f"{field}.edges[{index}]")
        graph.add_edge(edge.source, edge.target, edge.relation, note=edge.note)
    graph.validate()
    return graph


RECORD_TYPES = ("meta", "claim", "evidence", "link", "edge")
RECORD_KEYS_BY_TYPE = "record"
META_KEYS = ("record", "schema_version", "counts", "signature")
COUNT_KEYS = ("claims", "evidence", "links", "edges")


def meta_line(counts: Mapping[str, int], signature: str) -> str:
    """Return the canonical header line shared by every JSONL document type.

    The header carries the schema version, per-record counts and a content
    signature, which together let a reader detect a truncated or tampered
    artifact instead of silently rebuilding a partial object.
    """
    return dumps_line(
        {
            "record": "meta",
            "schema_version": SCHEMA_VERSION,
            "counts": dict(counts),
            "signature": signature,
        }
    )


def check_document_meta(
    meta: Mapping[str, Any] | None,
    *,
    counts: Mapping[str, int],
    signature: str,
    kind: str,
    count_keys: Sequence[str],
) -> None:
    """Verify a header against the records actually read.

    ``kind`` names the document type in error messages; ``count_keys`` fixes
    which counts the header must carry. A missing header is not an error: JSONL
    fragments written without one are still readable.
    """
    if meta is None:
        return
    field = f"{kind}.meta"
    recorded = require_mapping(meta["counts"], field=f"{field}.counts")
    reject_unknown_keys(recorded, count_keys, field=f"{field}.counts")
    expected = {
        key: require_int(recorded, key, field=f"{field}.counts") for key in count_keys
    }
    actual = {key: counts[key] for key in count_keys}
    if actual != expected:
        raise SchemaError(
            f"{kind} document counts disagree with its header",
            expected=expected,
            actual=actual,
        )
    recorded_signature = require_str(meta, "signature", field=field)
    if recorded_signature != signature:
        raise SchemaError(
            f"{kind} document signature disagrees with its header",
            recorded=recorded_signature,
            computed=signature,
        )


def graph_counts(stats: GraphStats) -> dict[str, int]:
    """Return the counted record types of a graph, keyed by name."""
    return {
        "claims": stats.claims,
        "evidence": stats.evidence,
        "links": stats.links,
        "edges": stats.edges,
    }


def graph_meta_line(graph: HypothesisGraph) -> str:
    """Return the header line carrying counts and the graph signature."""
    stats = graph.stats()
    return meta_line(graph_counts(stats), graph.signature())


def graph_to_lines(graph: HypothesisGraph, *, include_meta: bool = True) -> list[str]:
    """Serialize a graph as an ordered list of JSONL lines.

    Order is meta, claims, evidence, links, edges — dependencies before the
    records that reference them, so a reader can rebuild the graph in a single
    forward pass without backtracking.
    """
    lines = [graph_meta_line(graph)] if include_meta else []
    lines.extend(
        dumps_line({"record": "claim", "claim": claim_to_dict(claim)})
        for claim in graph.claims
    )
    lines.extend(
        dumps_line({"record": "evidence", "evidence": evidence_to_dict(item)})
        for item in graph.evidence_items
    )
    lines.extend(
        dumps_line({"record": "link", "claim_id": claim_id, "evidence_id": evidence_id})
        for claim_id, evidence_id in graph.link_pairs()
    )
    lines.extend(
        dumps_line({"record": "edge", **edge_to_dict(edge)}) for edge in graph.edges
    )
    return lines


def graph_to_text(graph: HypothesisGraph, *, include_meta: bool = True) -> str:
    """Return the whole JSONL document as one string."""
    return "".join(graph_to_lines(graph, include_meta=include_meta))


def graph_from_lines(
    lines: Iterable[str], *, verify_meta: bool = True
) -> HypothesisGraph:
    """Rebuild a graph from JSONL lines, checking the header when present."""
    graph = HypothesisGraph()
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="graph", line_number=number)
        record = require_str(payload, "record", field=f"graph[{number}]")
        if record not in RECORD_TYPES:
            raise SchemaError(
                "unknown graph record type",
                field=f"graph[{number}]",
                line_number=number,
                got=record,
                allowed=list(RECORD_TYPES),
            )
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"graph[{number}]")
            check_schema_version(
                payload, field=f"graph[{number}]", expected=SCHEMA_VERSION
            )
            meta = payload
        elif record == "claim":
            graph.add_claim(
                claim_from_dict(payload["claim"], field=f"graph[{number}].claim")
            )
        elif record == "evidence":
            graph.add_evidence(
                evidence_from_dict(
                    payload["evidence"], field=f"graph[{number}].evidence"
                )
            )
        elif record == "link":
            reject_unknown_keys(
                payload, ("record", *LINK_KEYS), field=f"graph[{number}]"
            )
            graph.link_evidence(
                require_str(payload, "claim_id", field=f"graph[{number}]"),
                require_str(payload, "evidence_id", field=f"graph[{number}]"),
            )
        else:
            edge_payload = {
                key: value for key, value in payload.items() if key != "record"
            }
            edge = edge_from_dict(edge_payload, field=f"graph[{number}]")
            graph.add_edge(edge.source, edge.target, edge.relation, note=edge.note)
    graph.validate()
    if verify_meta:
        check_document_meta(
            meta,
            counts=graph_counts(graph.stats()),
            signature=graph.signature(),
            kind="graph",
            count_keys=COUNT_KEYS,
        )
    return graph


def graph_from_text(text: str, *, verify_meta: bool = True) -> HypothesisGraph:
    """Rebuild a graph from a JSONL document string."""
    return graph_from_lines(text.splitlines(), verify_meta=verify_meta)


def write_graph(
    graph: HypothesisGraph, path: Path | str, *, include_meta: bool = True
) -> int:
    """Write a JSONL graph document atomically and return the line count.

    The document is written to a sibling temporary file and renamed into place,
    so an interrupted run never leaves a half-written artifact behind. The same
    graph always produces the same bytes, which is what makes checkpointed runs
    comparable with straight runs.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = graph_to_lines(graph, include_meta=include_meta)
    temporary = target.with_name(f"{target.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return len(lines)


def read_graph(path: Path | str, *, verify_meta: bool = True) -> HypothesisGraph:
    """Read a JSONL graph document, reporting a missing file as an artifact error."""
    source = Path(path)
    if not source.is_file():
        raise ArtifactError("graph document not found", path=str(source))
    with source.open("r", encoding="utf-8") as handle:
        return graph_from_lines(handle, verify_meta=verify_meta)


def read_jsonl_records(
    path: Path | str, *, field: str = "record"
) -> Iterator[dict[str, Any]]:
    """Yield decoded objects from any JSONL artifact, skipping blank lines."""
    source = Path(path)
    if not source.is_file():
        raise ArtifactError("artifact not found", path=str(source))
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield loads_line(line, field=field, line_number=number)


def write_jsonl_records(records: Iterable[Mapping[str, Any]], path: Path | str) -> int:
    """Write mappings as canonical JSONL lines atomically; return the count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.partial")
    written = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(dumps_line(record))
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return written


DOCUMENT_KEYS = ("document_id", "title", "text", "source", "attributes")
CORPUS_KEYS = ("schema_version", "documents")


def pair_list_from_payload(
    mapping: Mapping[str, Any], key: str, *, field: str
) -> tuple[tuple[str, str], ...]:
    """Decode a list of ``[key, value]`` pairs, preserving order."""
    value = present_value(mapping, key, field=field)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(
            f"{field}.{key} must be a list",
            field=field,
            key=key,
            got=type(value).__name__,
        )
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        item_field = f"{field}.{key}[{index}]"
        if (
            isinstance(item, (str, bytes))
            or not isinstance(item, Sequence)
            or len(item) != 2
        ):
            raise SchemaError(
                f"{item_field} must be a [key, value] pair", field=item_field, got=item
            )
        pairs.append(
            (
                require_str({"v": item[0]}, "v", field=item_field),
                require_str({"v": item[1]}, "v", field=item_field, allow_empty=True),
            )
        )
    return tuple(pairs)


def document_to_dict(document: Document) -> dict[str, Any]:
    """Encode a corpus document."""
    return {
        "document_id": document.document_id,
        "title": document.title,
        "text": document.text,
        "source": document.source,
        "attributes": [[key, value] for key, value in document.attributes],
    }


def document_from_dict(payload: object, *, field: str = "document") -> Document:
    """Decode a corpus document, rejecting unknown keys."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, DOCUMENT_KEYS, field=field)
    return Document(
        document_id=require_str(mapping, "document_id", field=field),
        title=require_str(mapping, "title", field=field),
        text=require_str(mapping, "text", field=field),
        source=require_str(mapping, "source", field=field),
        attributes=pair_list_from_payload(mapping, "attributes", field=field),
    )


def corpus_to_dict(corpus: Corpus) -> dict[str, Any]:
    """Encode a whole corpus with its schema version."""
    return {
        "schema_version": SCHEMA_VERSION,
        "documents": [document_to_dict(document) for document in corpus.documents],
    }


def corpus_from_dict(payload: object, *, field: str = "corpus") -> Corpus:
    """Decode a corpus, reporting the position of any malformed document."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, CORPUS_KEYS, field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    return Corpus(
        [
            document_from_dict(item, field=f"{field}.documents[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "documents", field=field)
            )
        ]
    )


CORPUS_RECORD_TYPES = ("meta", "document")
CORPUS_COUNT_KEYS = ("documents", "characters")


def corpus_counts(corpus: Corpus) -> dict[str, int]:
    """Return the counted properties recorded in a corpus header."""
    stats = corpus.stats()
    return {"documents": stats.documents, "characters": stats.characters}


def corpus_to_lines(corpus: Corpus, *, include_meta: bool = True) -> list[str]:
    """Serialize a corpus as ordered JSONL lines (header first)."""
    lines = (
        [meta_line(corpus_counts(corpus), corpus.signature())] if include_meta else []
    )
    lines.extend(
        dumps_line({"record": "document", "document": document_to_dict(document)})
        for document in corpus.documents
    )
    return lines


def corpus_to_text(corpus: Corpus, *, include_meta: bool = True) -> str:
    """Return the whole corpus document as one string."""
    return "".join(corpus_to_lines(corpus, include_meta=include_meta))


def corpus_from_lines(lines: Iterable[str], *, verify_meta: bool = True) -> Corpus:
    """Rebuild a corpus from JSONL lines, checking the header when present."""
    corpus = Corpus()
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="corpus", line_number=number)
        record = require_str(payload, "record", field=f"corpus[{number}]")
        if record not in CORPUS_RECORD_TYPES:
            raise SchemaError(
                "unknown corpus record type",
                field=f"corpus[{number}]",
                line_number=number,
                got=record,
                allowed=list(CORPUS_RECORD_TYPES),
            )
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"corpus[{number}]")
            check_schema_version(
                payload, field=f"corpus[{number}]", expected=SCHEMA_VERSION
            )
            meta = payload
        else:
            reject_unknown_keys(
                payload, ("record", "document"), field=f"corpus[{number}]"
            )
            corpus.add_document(
                document_from_dict(
                    payload["document"], field=f"corpus[{number}].document"
                )
            )
    if verify_meta:
        check_document_meta(
            meta,
            counts=corpus_counts(corpus),
            signature=corpus.signature(),
            kind="corpus",
            count_keys=CORPUS_COUNT_KEYS,
        )
    return corpus


def corpus_from_text(text: str, *, verify_meta: bool = True) -> Corpus:
    """Rebuild a corpus from a JSONL document string."""
    return corpus_from_lines(text.splitlines(), verify_meta=verify_meta)


def write_corpus(corpus: Corpus, path: Path | str, *, include_meta: bool = True) -> int:
    """Write a corpus document atomically and return the line count."""
    return write_lines(corpus_to_lines(corpus, include_meta=include_meta), path)


def read_corpus(path: Path | str, *, verify_meta: bool = True) -> Corpus:
    """Read a corpus document, reporting a missing file as an artifact error."""
    return corpus_from_lines(iter_lines(path), verify_meta=verify_meta)


def iter_lines(path: Path | str) -> Iterator[str]:
    """Yield the lines of a JSONL artifact, raising when it is missing."""
    source = Path(path)
    if not source.is_file():
        raise ArtifactError("artifact not found", path=str(source))
    with source.open("r", encoding="utf-8") as handle:
        yield from handle


def write_lines(lines: Iterable[str], path: Path | str) -> int:
    """Write pre-rendered lines atomically; return how many were written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.partial")
    written = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line if line.endswith("\n") else line + "\n")
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return written
