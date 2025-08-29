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
from dataclasses import replace
from pathlib import Path
from typing import Any

from hypoarena.agents import ReplayAgent, ReplayEntry
from hypoarena.codec import (
    check_schema_version,
    dumps_line,
    loads_line,
    optional_float,
    optional_int,
    optional_str,
    present_value,
    reject_unknown_keys,
    require_bool,
    require_enum,
    require_enum_list,
    require_float,
    require_float_list,
    require_int,
    require_int_list,
    require_mapping,
    require_mapping_list,
    require_str,
    require_str_tuple,
)
from hypoarena.corpus import Corpus, Document
from hypoarena.debate import Critique, DebateResult, DebateTurn
from hypoarena.errors import ArtifactError, SchemaError
from hypoarena.graph import ClaimEdge, GraphStats, HypothesisGraph
from hypoarena.grounding import (
    CitationCheck,
    GroundingFlag,
    GroundingIssue,
    GroundingReport,
    summarize_reports,
)
from hypoarena.http_agent import HttpConfig
from hypoarena.ids import content_hash
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
from hypoarena.synthetic import (
    PlantedChain,
    PlantedCluster,
    PlantedLink,
    PlantedTruth,
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


PLANTED_LINK_KEYS = (
    "chain_id",
    "subject",
    "target",
    "relation",
    "system",
    "kind",
)
CHAIN_KEYS = ("chain_id", "variables", "system", "links")
CLUSTER_KEYS = ("link_key", "statement", "document_ids")
TRUTH_KEYS = (
    "schema_version",
    "chains",
    "competing",
    "contradictions",
    "clusters",
    "distractor_ids",
)
TRUTH_RECORD_TYPES = (
    "meta",
    "chain",
    "competing",
    "contradiction",
    "cluster",
    "distractors",
)
DISTRACTOR_RECORD_KEYS = ("record", "document_ids")
TRUTH_COUNT_KEYS = ("chains", "competing", "contradictions", "clusters")


def link_to_dict(link: PlantedLink) -> dict[str, Any]:
    """Encode one planted link."""
    return {
        "chain_id": link.chain_id,
        "subject": link.subject,
        "target": link.target,
        "relation": link.relation.value,
        "system": link.system,
        "kind": link.kind,
    }


def link_from_dict(payload: object, *, field: str = "link") -> PlantedLink:
    """Decode one planted link."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, PLANTED_LINK_KEYS, field=field)
    return PlantedLink(
        chain_id=require_str(mapping, "chain_id", field=field),
        subject=require_str(mapping, "subject", field=field),
        target=require_str(mapping, "target", field=field),
        relation=require_enum(mapping, "relation", PredictedRelation, field=field),
        system=require_str(mapping, "system", field=field),
        kind=require_str(mapping, "kind", field=field),
    )


def chain_to_dict(chain: PlantedChain) -> dict[str, Any]:
    """Encode a planted chain with its links."""
    return {
        "chain_id": chain.chain_id,
        "variables": list(chain.variables),
        "system": chain.system,
        "links": [link_to_dict(link) for link in chain.links],
    }


def chain_from_dict(payload: object, *, field: str = "chain") -> PlantedChain:
    """Decode a planted chain."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, CHAIN_KEYS, field=field)
    return PlantedChain(
        chain_id=require_str(mapping, "chain_id", field=field),
        variables=require_str_tuple(mapping, "variables", field=field, minimum_items=2),
        system=require_str(mapping, "system", field=field),
        links=tuple(
            link_from_dict(item, field=f"{field}.links[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "links", field=field)
            )
        ),
    )


def cluster_to_dict(cluster: PlantedCluster) -> dict[str, Any]:
    """Encode a paraphrase cluster."""
    return {
        "link_key": list(cluster.link_key),
        "statement": cluster.statement,
        "document_ids": list(cluster.document_ids),
    }


def cluster_from_dict(payload: object, *, field: str = "cluster") -> PlantedCluster:
    """Decode a paraphrase cluster."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, CLUSTER_KEYS, field=field)
    key = require_str_tuple(mapping, "link_key", field=field)
    if len(key) != 3:
        raise SchemaError(
            f"{field}.link_key must have three parts", field=field, count=len(key)
        )
    return PlantedCluster(
        link_key=(key[0], key[1], key[2]),
        statement=require_str(mapping, "statement", field=field),
        document_ids=require_str_tuple(mapping, "document_ids", field=field),
    )


def truth_to_dict(truth: PlantedTruth) -> dict[str, Any]:
    """Encode planted ground truth as a single JSON object."""
    return {
        "schema_version": SCHEMA_VERSION,
        "chains": [chain_to_dict(chain) for chain in truth.chains],
        "competing": [link_to_dict(link) for link in truth.competing],
        "contradictions": [link_to_dict(link) for link in truth.contradictions],
        "clusters": [cluster_to_dict(cluster) for cluster in truth.clusters],
        "distractor_ids": list(truth.distractor_ids),
    }


def truth_from_dict(payload: object, *, field: str = "truth") -> PlantedTruth:
    """Decode planted ground truth."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, TRUTH_KEYS, field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    return PlantedTruth(
        chains=tuple(
            chain_from_dict(item, field=f"{field}.chains[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "chains", field=field)
            )
        ),
        competing=tuple(
            link_from_dict(item, field=f"{field}.competing[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "competing", field=field)
            )
        ),
        contradictions=tuple(
            link_from_dict(item, field=f"{field}.contradictions[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "contradictions", field=field)
            )
        ),
        clusters=tuple(
            cluster_from_dict(item, field=f"{field}.clusters[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "clusters", field=field)
            )
        ),
        distractor_ids=require_str_tuple(mapping, "distractor_ids", field=field),
    )


def truth_signature(truth: PlantedTruth) -> str:
    """Return a digest over the encoded ground truth."""
    return content_hash(truth_to_dict(truth))


def truth_counts(truth: PlantedTruth) -> dict[str, int]:
    """Return the record counts stored in a truth document header."""
    return {
        "chains": len(truth.chains),
        "competing": len(truth.competing),
        "contradictions": len(truth.contradictions),
        "clusters": len(truth.clusters),
    }


def truth_to_lines(truth: PlantedTruth, *, include_meta: bool = True) -> list[str]:
    """Serialize planted truth as ordered JSONL lines."""
    lines = (
        [meta_line(truth_counts(truth), truth_signature(truth))] if include_meta else []
    )
    lines.extend(
        dumps_line({"record": "chain", "chain": chain_to_dict(chain)})
        for chain in truth.chains
    )
    lines.extend(
        dumps_line({"record": "competing", "link": link_to_dict(link)})
        for link in truth.competing
    )
    lines.extend(
        dumps_line({"record": "contradiction", "link": link_to_dict(link)})
        for link in truth.contradictions
    )
    lines.extend(
        dumps_line({"record": "cluster", "cluster": cluster_to_dict(cluster)})
        for cluster in truth.clusters
    )
    lines.append(
        dumps_line(
            {"record": "distractors", "document_ids": list(truth.distractor_ids)}
        )
    )
    return lines


def truth_from_lines(lines: Iterable[str], *, verify_meta: bool = True) -> PlantedTruth:
    """Rebuild planted truth from JSONL lines, checking the header by default."""
    chains: list[PlantedChain] = []
    competing: list[PlantedLink] = []
    contradictions: list[PlantedLink] = []
    clusters: list[PlantedCluster] = []
    distractor_ids: tuple[str, ...] = ()
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="truth", line_number=number)
        record = require_str(payload, "record", field=f"truth[{number}]")
        if record not in TRUTH_RECORD_TYPES:
            raise SchemaError(
                "unknown truth record type",
                field=f"truth[{number}]",
                line_number=number,
                got=record,
                allowed=list(TRUTH_RECORD_TYPES),
            )
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"truth[{number}]")
            check_schema_version(
                payload, field=f"truth[{number}]", expected=SCHEMA_VERSION
            )
            meta = payload
        elif record == "chain":
            chains.append(chain_from_dict(payload["chain"], field=f"truth[{number}]"))
        elif record == "cluster":
            clusters.append(
                cluster_from_dict(payload["cluster"], field=f"truth[{number}]")
            )
        elif record == "distractors":
            reject_unknown_keys(
                payload, DISTRACTOR_RECORD_KEYS, field=f"truth[{number}]"
            )
            distractor_ids = require_str_tuple(
                payload, "document_ids", field=f"truth[{number}]"
            )
        else:
            link = link_from_dict(payload["link"], field=f"truth[{number}]")
            if record == "competing":
                competing.append(link)
            else:
                contradictions.append(link)
    truth = PlantedTruth(
        chains=tuple(chains),
        competing=tuple(competing),
        contradictions=tuple(contradictions),
        clusters=tuple(clusters),
        distractor_ids=distractor_ids,
    )
    if verify_meta:
        check_document_meta(
            meta,
            counts=truth_counts(truth),
            signature=truth_signature(truth),
            kind="truth",
            count_keys=TRUTH_COUNT_KEYS,
        )
    return truth


CITATION_CHECK_KEYS = (
    "document_id",
    "start",
    "end",
    "quote",
    "resolved",
    "issues",
    "entity_overlap",
    "claimed_numbers",
    "quoted_numbers",
    "detail",
)
GROUNDING_REPORT_KEYS = (
    "schema_version",
    "claim_id",
    "statement",
    "flag",
    "score",
    "issues",
    "checks",
)
GROUNDING_RECORD_TYPES = ("meta", "report")
GROUNDING_COUNT_KEYS = (
    "reports",
    "grounded",
    "weakly_grounded",
    "ungrounded",
    "fabricated",
)


def citation_check_to_dict(check: CitationCheck) -> dict[str, Any]:
    """Encode one citation check, flattening the citation into the payload."""
    return {
        "document_id": check.citation.document_id,
        "start": check.citation.start,
        "end": check.citation.end,
        "quote": check.citation.quote,
        "resolved": check.resolved,
        "issues": [issue.value for issue in check.issues],
        "entity_overlap": check.entity_overlap,
        "claimed_numbers": list(check.claimed_numbers),
        "quoted_numbers": list(check.quoted_numbers),
        "detail": check.detail,
    }


def citation_check_from_dict(payload: object, *, field: str = "check") -> CitationCheck:
    """Decode one citation check."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, CITATION_CHECK_KEYS, field=field)
    return CitationCheck(
        citation=Citation(
            require_str(mapping, "document_id", field=field),
            require_int(mapping, "start", field=field, minimum=0),
            require_int(mapping, "end", field=field, minimum=1),
            require_str(mapping, "quote", field=field),
        ),
        resolved=require_bool(mapping, "resolved", field=field),
        issues=require_enum_list(mapping, "issues", GroundingIssue, field=field),
        entity_overlap=require_float(mapping, "entity_overlap", field=field),
        claimed_numbers=require_float_list(mapping, "claimed_numbers", field=field),
        quoted_numbers=require_float_list(mapping, "quoted_numbers", field=field),
        detail=optional_str(mapping, "detail", field=field),
    )


def grounding_report_to_dict(report: GroundingReport) -> dict[str, Any]:
    """Encode a grounding report with all of its citation checks."""
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_id": report.claim_id,
        "statement": report.statement,
        "flag": report.flag.value,
        "score": report.score,
        "issues": [issue.value for issue in report.issues],
        "checks": [citation_check_to_dict(check) for check in report.checks],
    }


def grounding_report_from_dict(
    payload: object, *, field: str = "report"
) -> GroundingReport:
    """Decode a grounding report."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, GROUNDING_REPORT_KEYS, field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    return GroundingReport(
        claim_id=require_str(mapping, "claim_id", field=field),
        statement=require_str(mapping, "statement", field=field),
        flag=require_enum(mapping, "flag", GroundingFlag, field=field),
        score=require_float(mapping, "score", field=field, minimum=0.0, maximum=1.0),
        issues=require_enum_list(mapping, "issues", GroundingIssue, field=field),
        checks=tuple(
            citation_check_from_dict(item, field=f"{field}.checks[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "checks", field=field)
            )
        ),
    )


def grounding_signature(reports: Sequence[GroundingReport]) -> str:
    """Return a digest over a sequence of grounding reports."""
    return content_hash([grounding_report_to_dict(report) for report in reports])


def grounding_counts(reports: Sequence[GroundingReport]) -> dict[str, int]:
    """Return the per-flag counts stored in a grounding document header."""
    summary = summarize_reports(reports)
    return {
        "reports": summary.total,
        "grounded": summary.grounded,
        "weakly_grounded": summary.weakly_grounded,
        "ungrounded": summary.ungrounded,
        "fabricated": summary.fabricated,
    }


def grounding_reports_to_lines(
    reports: Sequence[GroundingReport], *, include_meta: bool = True
) -> list[str]:
    """Serialize grounding reports as ordered JSONL lines."""
    lines = (
        [meta_line(grounding_counts(reports), grounding_signature(reports))]
        if include_meta
        else []
    )
    lines.extend(
        dumps_line({"record": "report", "report": grounding_report_to_dict(report)})
        for report in reports
    )
    return lines


def grounding_reports_from_lines(
    lines: Iterable[str], *, verify_meta: bool = True
) -> tuple[GroundingReport, ...]:
    """Rebuild grounding reports from JSONL lines, checking the header."""
    reports: list[GroundingReport] = []
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="grounding", line_number=number)
        record = require_str(payload, "record", field=f"grounding[{number}]")
        if record not in GROUNDING_RECORD_TYPES:
            raise SchemaError(
                "unknown grounding record type",
                field=f"grounding[{number}]",
                line_number=number,
                got=record,
                allowed=list(GROUNDING_RECORD_TYPES),
            )
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"grounding[{number}]")
            check_schema_version(
                payload, field=f"grounding[{number}]", expected=SCHEMA_VERSION
            )
            meta = payload
        else:
            reject_unknown_keys(
                payload, ("record", "report"), field=f"grounding[{number}]"
            )
            reports.append(
                grounding_report_from_dict(
                    payload["report"], field=f"grounding[{number}].report"
                )
            )
    if verify_meta:
        check_document_meta(
            meta,
            counts=grounding_counts(reports),
            signature=grounding_signature(reports),
            kind="grounding",
            count_keys=GROUNDING_COUNT_KEYS,
        )
    return tuple(reports)


REPLAY_ENTRY_KEYS = (
    "task",
    "prompt",
    "text",
    "agent",
    "context",
    "prompt_tokens",
    "completion_tokens",
    "model",
)
REPLAY_LINE_KEYS = ("record", "entry")
REPLAY_COUNT_KEYS = ("entries",)


def replay_entry_to_dict(entry: ReplayEntry) -> dict[str, Any]:
    """Encode one replay fixture entry."""
    return {
        "task": entry.task,
        "prompt": entry.prompt,
        "text": entry.text,
        "agent": entry.agent,
        "context": list(entry.context),
        "prompt_tokens": entry.prompt_tokens,
        "completion_tokens": entry.completion_tokens,
        "model": entry.model,
    }


def replay_entry_from_dict(payload: object, *, field: str = "entry") -> ReplayEntry:
    """Decode one replay fixture entry."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, REPLAY_ENTRY_KEYS, field=field)
    return ReplayEntry(
        task=require_str(mapping, "task", field=field),
        prompt=require_str(mapping, "prompt", field=field),
        text=require_str(mapping, "text", field=field),
        agent=require_str(mapping, "agent", field=field),
        context=require_str_tuple(mapping, "context", field=field),
        prompt_tokens=optional_int(mapping, "prompt_tokens", field=field, minimum=0),
        completion_tokens=optional_int(
            mapping, "completion_tokens", field=field, minimum=0
        ),
        model=require_str(mapping, "model", field=field),
    )


def replay_entries_to_lines(entries: Sequence[ReplayEntry]) -> list[str]:
    """Serialize replay entries as JSONL lines with a counting header."""
    return [
        meta_line(
            {"entries": len(entries)},
            content_hash([replay_entry_to_dict(entry) for entry in entries]),
        ),
        *(
            dumps_line({"record": "replay", "entry": replay_entry_to_dict(entry)})
            for entry in entries
        ),
    ]


def replay_entries_from_lines(lines: Iterable[str]) -> tuple[ReplayEntry, ...]:
    """Load replay entries, verifying the header count and digest."""
    entries: list[ReplayEntry] = []
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="replay", line_number=number)
        record = require_str(payload, "record", field=f"replay[{number}]")
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"replay[{number}]")
            meta = payload
            continue
        if record != "replay":
            raise SchemaError(
                "unknown replay record type",
                field=f"replay[{number}]",
                line_number=number,
                got=record,
                allowed=["meta", "replay"],
            )
        reject_unknown_keys(payload, REPLAY_LINE_KEYS, field=f"replay[{number}]")
        entries.append(
            replay_entry_from_dict(payload["entry"], field=f"replay[{number}].entry")
        )
    check_document_meta(
        meta,
        counts={"entries": len(entries)},
        signature=content_hash([replay_entry_to_dict(entry) for entry in entries]),
        kind="replay",
        count_keys=REPLAY_COUNT_KEYS,
    )
    return tuple(entries)


def read_replay_entries(path: Path | str) -> tuple[ReplayEntry, ...]:
    """Read a replay fixture from disk."""
    return replay_entries_from_lines(iter_lines(path))


def write_replay_entries(entries: Sequence[ReplayEntry], path: Path | str) -> int:
    """Write a replay fixture atomically; return the line count."""
    return write_lines(replay_entries_to_lines(entries), path)


def replay_agent_from_lines(
    name: str, lines: Iterable[str], *, mode: str = "sequence"
) -> ReplayAgent:
    """Build a replay agent straight from a fixture document."""
    return ReplayAgent(name, replay_entries_from_lines(lines), mode=mode)


HTTP_CONFIG_KEYS = (
    "base_url",
    "model",
    "api_key",
    "timeout",
    "max_retries",
    "retry_backoff",
    "retry_statuses",
    "allow_remote",
)


def http_config_to_dict(config: HttpConfig) -> dict[str, Any]:
    """Encode an adapter configuration for run metadata.

    The encoded form is the *redacted* view: a key is replaced by a mask, so an
    artifact written from it can be shared without leaking a credential.
    """
    return dict(config.redacted())


def http_config_from_dict(payload: object, *, field: str = "http") -> HttpConfig:
    """Decode an adapter configuration, ignoring any stored credential.

    ``api_key`` is never read back: a configuration restored from an artifact
    always starts without one, and callers must supply a key explicitly if they
    intend to make authenticated calls.
    """
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, HTTP_CONFIG_KEYS, field=field)
    return HttpConfig(
        base_url=require_str(mapping, "base_url", field=field),
        model=require_str(mapping, "model", field=field),
        api_key=None,
        timeout=require_float(mapping, "timeout", field=field, minimum=0.000001),
        max_retries=require_int(mapping, "max_retries", field=field, minimum=0),
        retry_backoff=require_float(mapping, "retry_backoff", field=field, minimum=0.0),
        retry_statuses=tuple(
            int(value)
            for value in require_int_list(mapping, "retry_statuses", field=field)
        ),
        allow_remote=require_bool(mapping, "allow_remote", field=field),
    )


CRITIQUE_KEYS = ("agent", "text", "request_id")
TURN_KEYS = ("round_index", "statement", "critiques", "revised")
RESULT_KEYS = (
    "schema_version",
    "proposal",
    "final_statement",
    "rounds_run",
    "converged",
    "agents",
    "context",
    "config_fingerprint",
    "usage",
    "turns",
)
RESULT_LINE_KEYS = tuple(key for key in RESULT_KEYS if key != "turns")
DEBATE_RECORD_TYPES = ("meta", "turn", "result")
DEBATE_COUNT_KEYS = ("turns",)


def critique_to_dict(critique: Critique) -> dict[str, Any]:
    """Encode one critique."""
    return {
        "agent": critique.agent,
        "text": critique.text,
        "request_id": critique.request_id,
    }


def critique_from_dict(payload: object, *, field: str = "critique") -> Critique:
    """Decode one critique."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, CRITIQUE_KEYS, field=field)
    return Critique(
        agent=require_str(mapping, "agent", field=field),
        text=require_str(mapping, "text", field=field),
        request_id=require_str(mapping, "request_id", field=field),
    )


def turn_to_dict(turn: DebateTurn) -> dict[str, Any]:
    """Encode one debate round."""
    return {
        "round_index": turn.round_index,
        "statement": turn.statement,
        "critiques": [critique_to_dict(item) for item in turn.critiques],
        "revised": turn.revised,
    }


def turn_from_dict(payload: object, *, field: str = "turn") -> DebateTurn:
    """Decode one debate round."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, TURN_KEYS, field=field)
    return DebateTurn(
        round_index=require_int(mapping, "round_index", field=field, minimum=0),
        statement=require_str(mapping, "statement", field=field),
        critiques=tuple(
            critique_from_dict(item, field=f"{field}.critiques[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "critiques", field=field)
            )
        ),
        revised=require_str(mapping, "revised", field=field, allow_empty=True),
    )


def debate_result_to_dict(result: DebateResult) -> dict[str, Any]:
    """Encode a full debate result, including its turns."""
    payload = debate_summary_to_dict(result)
    payload["turns"] = [turn_to_dict(turn) for turn in result.turns]
    return payload


def debate_summary_to_dict(result: DebateResult) -> dict[str, Any]:
    """Encode a debate result without its turns (the JSONL result line)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "proposal": result.proposal,
        "final_statement": result.final_statement,
        "rounds_run": result.rounds_run,
        "converged": result.converged,
        "agents": list(result.agents),
        "context": list(result.context),
        "config_fingerprint": result.config_fingerprint,
        "usage": result.usage,
    }


def debate_result_from_dict(payload: object, *, field: str = "debate") -> DebateResult:
    """Decode a full debate result."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, RESULT_KEYS, field=field)
    summary = debate_summary_from_mapping(mapping, field=field)
    turns = tuple(
        turn_from_dict(item, field=f"{field}.turns[{index}]")
        for index, item in enumerate(
            require_mapping_list(mapping, "turns", field=field)
        )
    )
    return replace(summary, turns=turns)


def debate_summary_from_mapping(
    mapping: Mapping[str, Any], *, field: str, allowed: Sequence[str] = RESULT_KEYS
) -> DebateResult:
    """Decode the shared part of a debate payload."""
    reject_unknown_keys(mapping, (*allowed, "turns"), field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    usage = require_mapping(
        present_value(mapping, "usage", field=field), field=f"{field}.usage"
    )
    return DebateResult(
        proposal=require_str(mapping, "proposal", field=field),
        final_statement=require_str(mapping, "final_statement", field=field),
        turns=(),
        converged=require_bool(mapping, "converged", field=field),
        rounds_run=require_int(mapping, "rounds_run", field=field, minimum=0),
        agents=require_str_tuple(mapping, "agents", field=field),
        usage=dict(usage),
        context=require_str_tuple(mapping, "context", field=field),
        config_fingerprint=require_str(mapping, "config_fingerprint", field=field),
    )


def debate_signature(result: DebateResult) -> str:
    """Return a digest over the encoded transcript."""
    return content_hash(debate_result_to_dict(result))


def debate_to_lines(result: DebateResult, *, include_meta: bool = True) -> list[str]:
    """Serialize a debate as ordered JSONL lines: header, turns, result."""
    lines = (
        [meta_line({"turns": len(result.turns)}, debate_signature(result))]
        if include_meta
        else []
    )
    lines.extend(
        dumps_line({"record": "turn", "turn": turn_to_dict(turn)})
        for turn in result.turns
    )
    lines.append(
        dumps_line({"record": "result", "result": debate_summary_to_dict(result)})
    )
    return lines


def debate_from_lines(
    lines: Iterable[str], *, verify_meta: bool = True
) -> DebateResult:
    """Rebuild a debate result from JSONL lines."""
    turns: list[DebateTurn] = []
    result: DebateResult | None = None
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="debate", line_number=number)
        record = require_str(payload, "record", field=f"debate[{number}]")
        if record not in DEBATE_RECORD_TYPES:
            raise SchemaError(
                "unknown debate record type",
                field=f"debate[{number}]",
                line_number=number,
                got=record,
                allowed=list(DEBATE_RECORD_TYPES),
            )
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"debate[{number}]")
            check_schema_version(
                payload, field=f"debate[{number}]", expected=SCHEMA_VERSION
            )
            meta = payload
        elif record == "turn":
            reject_unknown_keys(payload, ("record", "turn"), field=f"debate[{number}]")
            turns.append(turn_from_dict(payload["turn"], field=f"debate[{number}]"))
        else:
            reject_unknown_keys(
                payload, ("record", "result"), field=f"debate[{number}]"
            )
            result = debate_summary_from_mapping(
                payload["result"],
                field=f"debate[{number}].result",
                allowed=RESULT_LINE_KEYS,
            )
    if result is None:
        raise SchemaError("debate document has no result line", field="debate")
    rebuilt = replace(result, turns=tuple(turns))
    if verify_meta:
        check_document_meta(
            meta,
            counts={"turns": len(turns)},
            signature=debate_signature(rebuilt),
            kind="debate",
            count_keys=DEBATE_COUNT_KEYS,
        )
    return rebuilt
