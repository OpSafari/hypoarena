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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from hypoarena.agents import ReplayAgent, ReplayEntry
from hypoarena.belief import (
    BeliefConfig,
    BeliefState,
    ContradictionPolicy,
    LikelihoodModel,
)
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
from hypoarena.config import RunConfig
from hypoarena.corpus import Corpus, Document
from hypoarena.cost import CostEntry, CostLedger
from hypoarena.debate import Critique, DebateConfig, DebateResult, DebateTurn
from hypoarena.dedup import DedupConfig, DedupReport, DuplicateCluster
from hypoarena.errors import ArtifactError, SchemaError
from hypoarena.evolve import EvolutionConfig, EvolutionRecord, EvolutionStep, Rejection
from hypoarena.graph import ClaimEdge, GraphStats, HypothesisGraph
from hypoarena.grounding import (
    CitationCheck,
    GroundingFlag,
    GroundingIssue,
    GroundingReport,
    VerifierConfig,
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
from hypoarena.stages import RunSummary, StageResult
from hypoarena.synthetic import (
    PlantedChain,
    PlantedCluster,
    PlantedLink,
    PlantedTruth,
    SyntheticConfig,
)
from hypoarena.tournament import (
    EloModel,
    MatchResult,
    Rating,
    RubricScore,
    RubricWeights,
    TournamentConfig,
    TournamentResult,
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


RUBRIC_KEYS = ("novelty", "testability", "grounding", "consistency")
RATING_KEYS = ("subject", "elo", "played", "wins", "losses", "draws")
MATCH_KEYS = (
    "left",
    "right",
    "left_score",
    "right_score",
    "left_total",
    "right_total",
    "outcome",
    "judge",
    "round_index",
    "match_index",
    "seed",
)
ELO_MODEL_KEYS = ("initial", "k_factor", "k_decay", "k_floor", "scale", "draw_margin")
TOURNAMENT_KEYS = (
    "schema_version",
    "seed",
    "repeats",
    "model",
    "weights",
    "ratings",
    "matches",
)
TOURNAMENT_RECORD_TYPES = ("meta", "config", "rating", "match")
TOURNAMENT_COUNT_KEYS = ("ratings", "matches")


def rubric_score_to_dict(score: RubricScore) -> dict[str, Any]:
    """Encode a rubric score."""
    return score.as_dict()


def rubric_score_from_dict(payload: object, *, field: str = "score") -> RubricScore:
    """Decode a rubric score, enforcing the ``[0, 1]`` bounds."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, RUBRIC_KEYS, field=field)
    return RubricScore(
        **{
            name: require_float(mapping, name, field=field, minimum=0.0, maximum=1.0)
            for name in RUBRIC_KEYS
        }
    )


def rating_to_dict(rating: Rating) -> dict[str, Any]:
    """Encode a rating exactly (no rounding) so replays stay identical."""
    return {
        "subject": rating.subject,
        "elo": rating.elo,
        "played": rating.played,
        "wins": rating.wins,
        "losses": rating.losses,
        "draws": rating.draws,
    }


def rating_from_dict(payload: object, *, field: str = "rating") -> Rating:
    """Decode a rating; the tally invariant is checked by the constructor."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, RATING_KEYS, field=field)
    return Rating(
        subject=require_str(mapping, "subject", field=field),
        elo=require_float(mapping, "elo", field=field),
        played=require_int(mapping, "played", field=field, minimum=0),
        wins=require_int(mapping, "wins", field=field, minimum=0),
        losses=require_int(mapping, "losses", field=field, minimum=0),
        draws=require_int(mapping, "draws", field=field, minimum=0),
    )


def match_to_dict(match: MatchResult) -> dict[str, Any]:
    """Encode one match of the audit trail."""
    return {
        "left": match.left,
        "right": match.right,
        "left_score": rubric_score_to_dict(match.left_score),
        "right_score": rubric_score_to_dict(match.right_score),
        "left_total": match.left_total,
        "right_total": match.right_total,
        "outcome": match.outcome,
        "judge": match.judge,
        "round_index": match.round_index,
        "match_index": match.match_index,
        "seed": match.seed,
    }


def match_from_dict(payload: object, *, field: str = "match") -> MatchResult:
    """Decode one match of the audit trail."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, MATCH_KEYS, field=field)
    return MatchResult(
        left=require_str(mapping, "left", field=field),
        right=require_str(mapping, "right", field=field),
        left_score=rubric_score_from_dict(
            present_value(mapping, "left_score", field=field),
            field=f"{field}.left_score",
        ),
        right_score=rubric_score_from_dict(
            present_value(mapping, "right_score", field=field),
            field=f"{field}.right_score",
        ),
        left_total=require_float(
            mapping, "left_total", field=field, minimum=0.0, maximum=1.0
        ),
        right_total=require_float(
            mapping, "right_total", field=field, minimum=0.0, maximum=1.0
        ),
        outcome=require_float(
            mapping, "outcome", field=field, minimum=0.0, maximum=1.0
        ),
        judge=require_str(mapping, "judge", field=field),
        round_index=require_int(mapping, "round_index", field=field, minimum=0),
        match_index=require_int(mapping, "match_index", field=field, minimum=0),
        seed=require_int(mapping, "seed", field=field),
    )


def elo_model_to_dict(model: EloModel) -> dict[str, Any]:
    """Encode the rating model settings."""
    return {
        "initial": model.initial,
        "k_factor": model.k_factor,
        "k_decay": model.k_decay,
        "k_floor": model.k_floor,
        "scale": model.scale,
        "draw_margin": model.draw_margin,
    }


def elo_model_from_dict(payload: object, *, field: str = "model") -> EloModel:
    """Decode the rating model settings."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, ELO_MODEL_KEYS, field=field)
    return EloModel(
        initial=require_float(mapping, "initial", field=field),
        k_factor=require_float(mapping, "k_factor", field=field),
        k_decay=require_float(mapping, "k_decay", field=field),
        k_floor=require_float(mapping, "k_floor", field=field),
        scale=require_float(mapping, "scale", field=field),
        draw_margin=require_float(mapping, "draw_margin", field=field),
    )


def tournament_to_dict(result: TournamentResult) -> dict[str, Any]:
    """Encode standings, configuration and the full audit trail."""
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": result.config.seed,
        "repeats": result.config.repeats,
        "model": elo_model_to_dict(result.config.model),
        "weights": result.config.weights.as_dict(),
        "ratings": [rating_to_dict(rating) for rating in result.ratings],
        "matches": [match_to_dict(match) for match in result.matches],
    }


def tournament_from_dict(
    payload: object, *, field: str = "tournament"
) -> TournamentResult:
    """Decode a tournament result."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, TOURNAMENT_KEYS, field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    model = elo_model_from_dict(
        present_value(mapping, "model", field=field), field=f"{field}.model"
    )
    weights_payload = require_mapping(
        present_value(mapping, "weights", field=field), field=f"{field}.weights"
    )
    reject_unknown_keys(weights_payload, RUBRIC_KEYS, field=f"{field}.weights")
    weights = RubricWeights(
        **{
            name: require_float(
                weights_payload, name, field=f"{field}.weights", minimum=0.0
            )
            for name in RUBRIC_KEYS
        }
    )
    config = TournamentConfig(
        seed=require_int(mapping, "seed", field=field),
        repeats=require_int(mapping, "repeats", field=field, minimum=1),
        model=model,
        weights=weights,
    )
    ratings = tuple(
        rating_from_dict(item, field=f"{field}.ratings[{index}]")
        for index, item in enumerate(
            require_mapping_list(mapping, "ratings", field=field)
        )
    )
    matches = tuple(
        match_from_dict(item, field=f"{field}.matches[{index}]")
        for index, item in enumerate(
            require_mapping_list(mapping, "matches", field=field)
        )
    )
    subjects = tuple(sorted({subject for match in matches for subject in match.pair()}))
    if not subjects:
        subjects = tuple(rating.subject for rating in ratings)
    return TournamentResult(
        subjects=subjects, ratings=ratings, matches=matches, config=config
    )


def tournament_signature(result: TournamentResult) -> str:
    """Return a digest over the encoded tournament."""
    return content_hash(tournament_to_dict(result))


def tournament_to_lines(
    result: TournamentResult, *, include_meta: bool = True
) -> list[str]:
    """Serialize a tournament as ordered JSONL lines: header, config, records."""
    lines = (
        [
            meta_line(
                {
                    "ratings": len(result.ratings),
                    "matches": len(result.matches),
                },
                tournament_signature(result),
            )
        ]
        if include_meta
        else []
    )
    lines.append(
        dumps_line(
            {
                "record": "config",
                "seed": result.config.seed,
                "repeats": result.config.repeats,
                "model": elo_model_to_dict(result.config.model),
                "weights": result.config.weights.as_dict(),
            }
        )
    )
    lines.extend(
        dumps_line({"record": "rating", "rating": rating_to_dict(rating)})
        for rating in result.ratings
    )
    lines.extend(
        dumps_line({"record": "match", "match": match_to_dict(match)})
        for match in result.matches
    )
    return lines


def tournament_from_lines(
    lines: Iterable[str], *, verify_meta: bool = True
) -> TournamentResult:
    """Rebuild a tournament result from JSONL lines."""
    config_payload: dict[str, Any] | None = None
    ratings: list[Rating] = []
    matches: list[MatchResult] = []
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="tournament", line_number=number)
        record = require_str(payload, "record", field=f"tournament[{number}]")
        if record not in TOURNAMENT_RECORD_TYPES:
            raise SchemaError(
                "unknown tournament record type",
                field=f"tournament[{number}]",
                line_number=number,
                got=record,
                allowed=list(TOURNAMENT_RECORD_TYPES),
            )
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"tournament[{number}]")
            check_schema_version(
                payload, field=f"tournament[{number}]", expected=SCHEMA_VERSION
            )
            meta = payload
        elif record == "config":
            reject_unknown_keys(
                payload,
                ("record", "seed", "repeats", "model", "weights"),
                field=f"tournament[{number}]",
            )
            config_payload = payload
        elif record == "rating":
            ratings.append(
                rating_from_dict(payload["rating"], field=f"tournament[{number}]")
            )
        else:
            matches.append(
                match_from_dict(payload["match"], field=f"tournament[{number}]")
            )
    if config_payload is None:
        raise SchemaError("tournament document has no config line", field="tournament")
    config = TournamentConfig(
        seed=require_int(config_payload, "seed", field="tournament.config"),
        repeats=require_int(
            config_payload, "repeats", field="tournament.config", minimum=1
        ),
        model=elo_model_from_dict(
            config_payload["model"], field="tournament.config.model"
        ),
        weights=RubricWeights(
            **{name: float(config_payload["weights"][name]) for name in RUBRIC_KEYS}
        ),
    )
    subjects = tuple(sorted({subject for match in matches for subject in match.pair()}))
    if not subjects:
        subjects = tuple(rating.subject for rating in ratings)
    result = TournamentResult(
        subjects=subjects,
        ratings=tuple(ratings),
        matches=tuple(matches),
        config=config,
    )
    if verify_meta:
        check_document_meta(
            meta,
            counts={"ratings": len(ratings), "matches": len(matches)},
            signature=tournament_signature(result),
            kind="tournament",
            count_keys=TOURNAMENT_COUNT_KEYS,
        )
    return result


SYNTHETIC_CONFIG_KEYS = (
    "seed",
    "chains",
    "chain_length",
    "paraphrases_per_link",
    "competitors_per_chain",
    "contradictions_per_chain",
    "distractor_documents",
    "filler_sentences",
    "source_tag",
)
DEDUP_CONFIG_KEYS = (
    "method",
    "threshold",
    "ngram_size",
    "word_ngram_size",
    "num_perm",
    "bands",
    "min_document_frequency",
    "use_lsh",
    "shingle_unit",
    "content_only",
    "seed",
)
SIMILARITY_KEYS = ("left", "right", "score")
DUPLICATE_CLUSTER_KEYS = (
    "representative",
    "method",
    "members",
    "similarities",
)
DEDUP_REPORT_KEYS = ("schema_version", "total", "config", "clusters")
DEDUP_RECORD_TYPES = ("meta", "report")
DEDUP_COUNT_KEYS = ("clusters", "members")


def dedup_config_to_dict(config: DedupConfig) -> dict[str, Any]:
    """Encode a dedup configuration."""
    return asdict(config)


def dedup_config_from_dict(payload: object, *, field: str = "config") -> DedupConfig:
    """Decode a dedup configuration; the constructor validates every value."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, DEDUP_CONFIG_KEYS, field=field)
    return DedupConfig(
        method=require_str(mapping, "method", field=field),
        threshold=require_float(
            mapping, "threshold", field=field, minimum=0.0, maximum=1.0
        ),
        ngram_size=require_int(mapping, "ngram_size", field=field, minimum=1),
        word_ngram_size=require_int(mapping, "word_ngram_size", field=field, minimum=1),
        num_perm=require_int(mapping, "num_perm", field=field, minimum=1),
        bands=require_int(mapping, "bands", field=field, minimum=1),
        min_document_frequency=require_int(
            mapping, "min_document_frequency", field=field, minimum=1
        ),
        use_lsh=require_bool(mapping, "use_lsh", field=field),
        shingle_unit=require_str(mapping, "shingle_unit", field=field),
        content_only=require_bool(mapping, "content_only", field=field),
        seed=require_int(mapping, "seed", field=field),
    )


def similarity_to_dict(left: str, right: str, score: float) -> dict[str, Any]:
    """Encode one verified pair similarity."""
    return {"left": left, "right": right, "score": score}


def duplicate_cluster_to_dict(cluster: DuplicateCluster) -> dict[str, Any]:
    """Encode one duplicate cluster."""
    return {
        "representative": cluster.representative,
        "method": cluster.method,
        "members": list(cluster.members),
        "similarities": [
            similarity_to_dict(left, right, score)
            for left, right, score in cluster.similarities
        ],
    }


def duplicate_cluster_from_dict(
    payload: object, *, field: str = "cluster"
) -> DuplicateCluster:
    """Decode one duplicate cluster."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, DUPLICATE_CLUSTER_KEYS, field=field)
    similarities = []
    for index, item in enumerate(
        require_mapping_list(mapping, "similarities", field=field)
    ):
        entry_field = f"{field}.similarities[{index}]"
        reject_unknown_keys(item, SIMILARITY_KEYS, field=entry_field)
        similarities.append(
            (
                require_str(item, "left", field=entry_field),
                require_str(item, "right", field=entry_field),
                require_float(
                    item, "score", field=entry_field, minimum=0.0, maximum=1.0
                ),
            )
        )
    return DuplicateCluster(
        members=require_str_tuple(mapping, "members", field=field, minimum_items=2),
        representative=require_str(mapping, "representative", field=field),
        method=require_str(mapping, "method", field=field),
        similarities=tuple(similarities),
    )


def dedup_report_to_dict(report: DedupReport) -> dict[str, Any]:
    """Encode a dedup report with its configuration and clusters."""
    return {
        "schema_version": SCHEMA_VERSION,
        "total": report.total,
        "config": dedup_config_to_dict(report.config),
        "clusters": [duplicate_cluster_to_dict(cluster) for cluster in report.clusters],
    }


def dedup_report_from_dict(payload: object, *, field: str = "dedup") -> DedupReport:
    """Decode a dedup report."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, DEDUP_REPORT_KEYS, field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    return DedupReport(
        clusters=tuple(
            duplicate_cluster_from_dict(item, field=f"{field}.clusters[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "clusters", field=field)
            )
        ),
        config=dedup_config_from_dict(
            present_value(mapping, "config", field=field), field=f"{field}.config"
        ),
        total=require_int(mapping, "total", field=field, minimum=0),
    )


def dedup_signature(report: DedupReport) -> str:
    """Return a digest over the encoded report."""
    return content_hash(dedup_report_to_dict(report))


def dedup_counts(report: DedupReport) -> dict[str, int]:
    """Return the counts stored in a dedup document header."""
    return {"clusters": len(report.clusters), "members": report.duplicated}


def dedup_report_to_lines(
    report: DedupReport, *, include_meta: bool = True
) -> list[str]:
    """Serialize a dedup report as JSONL lines."""
    lines = (
        [meta_line(dedup_counts(report), dedup_signature(report))]
        if include_meta
        else []
    )
    lines.append(
        dumps_line({"record": "report", "report": dedup_report_to_dict(report)})
    )
    return lines


def dedup_report_from_lines(
    lines: Iterable[str], *, verify_meta: bool = True
) -> DedupReport:
    """Rebuild a dedup report from JSONL lines."""
    report: DedupReport | None = None
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="dedup", line_number=number)
        record = require_str(payload, "record", field=f"dedup[{number}]")
        if record not in DEDUP_RECORD_TYPES:
            raise SchemaError(
                "unknown dedup record type",
                field=f"dedup[{number}]",
                line_number=number,
                got=record,
                allowed=list(DEDUP_RECORD_TYPES),
            )
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"dedup[{number}]")
            check_schema_version(
                payload, field=f"dedup[{number}]", expected=SCHEMA_VERSION
            )
            meta = payload
        else:
            reject_unknown_keys(payload, ("record", "report"), field=f"dedup[{number}]")
            report = dedup_report_from_dict(
                payload["report"], field=f"dedup[{number}].report"
            )
    if report is None:
        raise SchemaError("dedup document has no report line", field="dedup")
    if verify_meta:
        check_document_meta(
            meta,
            counts=dedup_counts(report),
            signature=dedup_signature(report),
            kind="dedup",
            count_keys=DEDUP_COUNT_KEYS,
        )
    return report


EVOLUTION_RECORD_KEYS = ("operator", "parents", "child", "rationale")
REJECTION_KEYS = (
    "operator",
    "parents",
    "reason",
    "statement",
    "nearest",
    "similarity",
)
EVOLUTION_STEP_KEYS = ("schema_version", "generation", "accepted", "rejected")
EVOLUTION_RECORD_TYPES = ("meta", "step")
EVOLUTION_COUNT_KEYS = ("steps", "accepted", "rejected")


def rejection_to_dict(rejection: Rejection) -> dict[str, Any]:
    """Encode one refused candidate."""
    return {
        "operator": rejection.operator,
        "parents": list(rejection.parents),
        "reason": rejection.reason,
        "statement": rejection.statement,
        "nearest": rejection.nearest,
        "similarity": rejection.similarity,
    }


def rejection_from_dict(payload: object, *, field: str = "rejection") -> Rejection:
    """Decode one refused candidate."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, REJECTION_KEYS, field=field)
    return Rejection(
        operator=require_str(mapping, "operator", field=field),
        parents=require_str_tuple(mapping, "parents", field=field),
        reason=require_str(mapping, "reason", field=field),
        statement=require_str(mapping, "statement", field=field, allow_empty=True),
        nearest=optional_str(mapping, "nearest", field=field),
        similarity=require_float(
            mapping, "similarity", field=field, minimum=0.0, maximum=1.0
        ),
    )


def evolution_record_to_dict(record: EvolutionRecord) -> dict[str, Any]:
    """Encode one accepted evolution, embedding the child claim."""
    return {
        "operator": record.operator,
        "parents": list(record.parents),
        "child": claim_to_dict(record.child),
        "rationale": record.rationale,
    }


def evolution_record_from_dict(
    payload: object, *, field: str = "record"
) -> EvolutionRecord:
    """Decode one accepted evolution."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, EVOLUTION_RECORD_KEYS, field=field)
    return EvolutionRecord(
        operator=require_str(mapping, "operator", field=field),
        parents=require_str_tuple(mapping, "parents", field=field, minimum_items=1),
        child=claim_from_dict(
            present_value(mapping, "child", field=field), field=f"{field}.child"
        ),
        rationale=require_str(mapping, "rationale", field=field),
    )


def evolution_step_to_dict(step: EvolutionStep) -> dict[str, Any]:
    """Encode one generation."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generation": step.generation,
        "accepted": [evolution_record_to_dict(item) for item in step.accepted],
        "rejected": [rejection_to_dict(item) for item in step.rejected],
    }


def evolution_step_from_dict(payload: object, *, field: str = "step") -> EvolutionStep:
    """Decode one generation."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, EVOLUTION_STEP_KEYS, field=field)
    check_schema_version(mapping, field=field, expected=SCHEMA_VERSION)
    return EvolutionStep(
        generation=require_int(mapping, "generation", field=field, minimum=0),
        accepted=tuple(
            evolution_record_from_dict(item, field=f"{field}.accepted[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "accepted", field=field)
            )
        ),
        rejected=tuple(
            rejection_from_dict(item, field=f"{field}.rejected[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "rejected", field=field)
            )
        ),
    )


def evolution_signature(steps: Sequence[EvolutionStep]) -> str:
    """Return a digest over a sequence of generations."""
    return content_hash([evolution_step_to_dict(step) for step in steps])


def evolution_counts(steps: Sequence[EvolutionStep]) -> dict[str, int]:
    """Return the counts stored in an evolution document header."""
    return {
        "steps": len(steps),
        "accepted": sum(len(step.accepted) for step in steps),
        "rejected": sum(len(step.rejected) for step in steps),
    }


def evolution_to_lines(
    steps: Sequence[EvolutionStep], *, include_meta: bool = True
) -> list[str]:
    """Serialize generations as ordered JSONL lines."""
    lines = (
        [meta_line(evolution_counts(steps), evolution_signature(steps))]
        if include_meta
        else []
    )
    lines.extend(
        dumps_line({"record": "step", "step": evolution_step_to_dict(step)})
        for step in steps
    )
    return lines


def evolution_from_lines(
    lines: Iterable[str], *, verify_meta: bool = True
) -> tuple[EvolutionStep, ...]:
    """Rebuild generations from JSONL lines."""
    steps: list[EvolutionStep] = []
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="evolution", line_number=number)
        record = require_str(payload, "record", field=f"evolution[{number}]")
        if record not in EVOLUTION_RECORD_TYPES:
            raise SchemaError(
                "unknown evolution record type",
                field=f"evolution[{number}]",
                line_number=number,
                got=record,
                allowed=list(EVOLUTION_RECORD_TYPES),
            )
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"evolution[{number}]")
            check_schema_version(
                payload, field=f"evolution[{number}]", expected=SCHEMA_VERSION
            )
            meta = payload
        else:
            reject_unknown_keys(
                payload, ("record", "step"), field=f"evolution[{number}]"
            )
            steps.append(
                evolution_step_from_dict(payload["step"], field=f"evolution[{number}]")
            )
    if verify_meta:
        check_document_meta(
            meta,
            counts=evolution_counts(steps),
            signature=evolution_signature(steps),
            kind="evolution",
            count_keys=EVOLUTION_COUNT_KEYS,
        )
    return tuple(steps)


LIKELIHOOD_KEYS = ("support_ratio", "refute_ratio", "neutral_ratio")
BELIEF_CONFIG_KEYS = (
    "prior",
    "likelihood",
    "contradiction_policy",
    "downweight_factor",
    "contradiction_threshold",
)
BELIEF_STATE_KEYS = (
    "claim_id",
    "prior",
    "posterior",
    "likelihood_ratio",
    "supporting",
    "refuting",
    "neutral",
)
BELIEF_RECORD_TYPES = ("meta", "config", "state")
BELIEF_COUNT_KEYS = ("states",)


def likelihood_model_to_dict(model: LikelihoodModel) -> dict[str, Any]:
    """Encode the likelihood model."""
    return asdict(model)


def likelihood_model_from_dict(
    payload: object, *, field: str = "likelihood"
) -> LikelihoodModel:
    """Decode the likelihood model."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, LIKELIHOOD_KEYS, field=field)
    return LikelihoodModel(
        support_ratio=require_float(mapping, "support_ratio", field=field),
        refute_ratio=require_float(mapping, "refute_ratio", field=field),
        neutral_ratio=require_float(mapping, "neutral_ratio", field=field),
    )


def belief_config_to_dict(config: BeliefConfig) -> dict[str, Any]:
    """Encode a belief configuration."""
    return {
        "prior": config.prior,
        "likelihood": likelihood_model_to_dict(config.likelihood),
        "contradiction_policy": config.contradiction_policy.value,
        "downweight_factor": config.downweight_factor,
        "contradiction_threshold": config.contradiction_threshold,
    }


def belief_config_from_dict(
    payload: object, *, field: str = "belief_config"
) -> BeliefConfig:
    """Decode a belief configuration."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, BELIEF_CONFIG_KEYS, field=field)
    return BeliefConfig(
        prior=require_float(mapping, "prior", field=field, minimum=0.0, maximum=1.0),
        likelihood=likelihood_model_from_dict(
            present_value(mapping, "likelihood", field=field),
            field=f"{field}.likelihood",
        ),
        contradiction_policy=require_enum(
            mapping, "contradiction_policy", ContradictionPolicy, field=field
        ),
        downweight_factor=require_float(
            mapping, "downweight_factor", field=field, minimum=0.0, maximum=1.0
        ),
        contradiction_threshold=require_int(
            mapping, "contradiction_threshold", field=field, minimum=1
        ),
    )


def belief_state_to_dict(state: BeliefState) -> dict[str, Any]:
    """Encode a belief state exactly, so replays reproduce it bit for bit."""
    return {
        "claim_id": state.claim_id,
        "prior": state.prior,
        "posterior": state.posterior,
        "likelihood_ratio": state.likelihood_ratio,
        "supporting": state.supporting,
        "refuting": state.refuting,
        "neutral": state.neutral,
    }


def belief_state_from_dict(payload: object, *, field: str = "state") -> BeliefState:
    """Decode a belief state."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, BELIEF_STATE_KEYS, field=field)
    return BeliefState(
        claim_id=require_str(mapping, "claim_id", field=field),
        prior=require_float(mapping, "prior", field=field, minimum=0.0, maximum=1.0),
        posterior=require_float(
            mapping, "posterior", field=field, minimum=0.0, maximum=1.0
        ),
        likelihood_ratio=require_float(
            mapping, "likelihood_ratio", field=field, minimum=0.0
        ),
        supporting=require_int(mapping, "supporting", field=field, minimum=0),
        refuting=require_int(mapping, "refuting", field=field, minimum=0),
        neutral=require_int(mapping, "neutral", field=field, minimum=0),
    )


def belief_signature(states: Sequence[BeliefState], config: BeliefConfig) -> str:
    """Return a digest over the encoded states and their configuration."""
    return content_hash(
        {
            "config": belief_config_to_dict(config),
            "states": [belief_state_to_dict(state) for state in states],
        }
    )


def belief_to_lines(
    states: Sequence[BeliefState],
    config: BeliefConfig | None = None,
    *,
    include_meta: bool = True,
) -> list[str]:
    """Serialize belief states as ordered JSONL lines: header, config, states."""
    settings = config or BeliefConfig()
    lines = (
        [meta_line({"states": len(states)}, belief_signature(states, settings))]
        if include_meta
        else []
    )
    lines.append(
        dumps_line({"record": "config", "config": belief_config_to_dict(settings)})
    )
    lines.extend(
        dumps_line({"record": "state", "state": belief_state_to_dict(state)})
        for state in states
    )
    return lines


def belief_from_lines(
    lines: Iterable[str], *, verify_meta: bool = True
) -> tuple[tuple[BeliefState, ...], BeliefConfig]:
    """Rebuild belief states and their configuration from JSONL lines."""
    states: list[BeliefState] = []
    config: BeliefConfig | None = None
    meta: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = loads_line(line, field="belief", line_number=number)
        record = require_str(payload, "record", field=f"belief[{number}]")
        if record not in BELIEF_RECORD_TYPES:
            raise SchemaError(
                "unknown belief record type",
                field=f"belief[{number}]",
                line_number=number,
                got=record,
                allowed=list(BELIEF_RECORD_TYPES),
            )
        if record == "meta":
            reject_unknown_keys(payload, META_KEYS, field=f"belief[{number}]")
            check_schema_version(
                payload, field=f"belief[{number}]", expected=SCHEMA_VERSION
            )
            meta = payload
        elif record == "config":
            reject_unknown_keys(
                payload, ("record", "config"), field=f"belief[{number}]"
            )
            config = belief_config_from_dict(
                payload["config"], field=f"belief[{number}].config"
            )
        else:
            reject_unknown_keys(payload, ("record", "state"), field=f"belief[{number}]")
            states.append(
                belief_state_from_dict(payload["state"], field=f"belief[{number}]")
            )
    settings = config or BeliefConfig()
    if verify_meta:
        check_document_meta(
            meta,
            counts={"states": len(states)},
            signature=belief_signature(states, settings),
            kind="belief",
            count_keys=BELIEF_COUNT_KEYS,
        )
    return tuple(states), settings


RUN_CONFIG_KEYS = (
    "seed",
    "stages",
    "run_id",
    "corpus",
    "grounding",
    "dedup",
    "debate",
    "tournament",
    "belief",
    "evolution",
)
GROUNDING_CONFIG_KEYS = (
    "min_entity_overlap",
    "min_quote_length",
    "numeric_tolerance",
    "polarity_checks",
    "numeric_checks",
)
TOURNAMENT_CONFIG_KEYS = ("seed", "repeats", "model", "weights")
DEBATE_CONFIG_KEYS = (
    "rounds",
    "critics",
    "stop_on_unchanged",
    "proposal_prompt",
    "critique_prompt",
)
EVOLUTION_CONFIG_KEYS = (
    "seed",
    "operators",
    "per_operator",
    "generations",
    "novelty_threshold",
)


def verifier_config_to_dict(config: VerifierConfig) -> dict[str, Any]:
    """Encode grounding verifier thresholds."""
    return asdict(config)


def verifier_config_from_dict(
    payload: object, *, field: str = "grounding"
) -> VerifierConfig:
    """Decode grounding verifier thresholds."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, GROUNDING_CONFIG_KEYS, field=field)
    return VerifierConfig(
        min_entity_overlap=require_float(
            mapping, "min_entity_overlap", field=field, minimum=0.0, maximum=1.0
        ),
        min_quote_length=require_int(
            mapping, "min_quote_length", field=field, minimum=1
        ),
        numeric_tolerance=require_float(
            mapping, "numeric_tolerance", field=field, minimum=0.0
        ),
        polarity_checks=require_bool(mapping, "polarity_checks", field=field),
        numeric_checks=require_bool(mapping, "numeric_checks", field=field),
    )


def debate_config_to_dict(config: DebateConfig) -> dict[str, Any]:
    """Encode debate loop settings."""
    return asdict(config)


def debate_config_from_dict(payload: object, *, field: str = "debate") -> DebateConfig:
    """Decode debate loop settings."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, DEBATE_CONFIG_KEYS, field=field)
    return DebateConfig(
        rounds=require_int(mapping, "rounds", field=field, minimum=1),
        critics=require_int(mapping, "critics", field=field, minimum=1),
        stop_on_unchanged=require_bool(mapping, "stop_on_unchanged", field=field),
        proposal_prompt=require_str(mapping, "proposal_prompt", field=field),
        critique_prompt=require_str(mapping, "critique_prompt", field=field),
    )


def tournament_config_to_dict(config: TournamentConfig) -> dict[str, Any]:
    """Encode tournament settings."""
    return {
        "seed": config.seed,
        "repeats": config.repeats,
        "model": elo_model_to_dict(config.model),
        "weights": config.weights.as_dict(),
    }


def tournament_config_from_dict(
    payload: object, *, field: str = "tournament"
) -> TournamentConfig:
    """Decode tournament settings."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, TOURNAMENT_CONFIG_KEYS, field=field)
    weights_payload = require_mapping(
        present_value(mapping, "weights", field=field), field=f"{field}.weights"
    )
    reject_unknown_keys(weights_payload, RUBRIC_KEYS, field=f"{field}.weights")
    return TournamentConfig(
        seed=require_int(mapping, "seed", field=field),
        repeats=require_int(mapping, "repeats", field=field, minimum=1),
        model=elo_model_from_dict(
            present_value(mapping, "model", field=field), field=f"{field}.model"
        ),
        weights=RubricWeights(
            **{
                name: require_float(
                    weights_payload, name, field=f"{field}.weights", minimum=0.0
                )
                for name in RUBRIC_KEYS
            }
        ),
    )


def evolution_config_to_dict(config: EvolutionConfig) -> dict[str, Any]:
    """Encode evolution settings."""
    return asdict(config)


def evolution_config_from_dict(
    payload: object, *, field: str = "evolution"
) -> EvolutionConfig:
    """Decode evolution settings."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, EVOLUTION_CONFIG_KEYS, field=field)
    return EvolutionConfig(
        seed=require_int(mapping, "seed", field=field),
        operators=require_str_tuple(mapping, "operators", field=field, minimum_items=1),
        per_operator=require_int(mapping, "per_operator", field=field, minimum=1),
        generations=require_int(mapping, "generations", field=field, minimum=1),
        novelty_threshold=require_float(
            mapping, "novelty_threshold", field=field, minimum=0.0, maximum=1.0
        ),
    )


def run_config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Encode a whole run configuration."""
    return {
        "seed": config.seed,
        "stages": list(config.stages),
        "run_id": config.run_id,
        "corpus": asdict(config.corpus),
        "grounding": verifier_config_to_dict(config.grounding),
        "dedup": dedup_config_to_dict(config.dedup),
        "debate": debate_config_to_dict(config.debate),
        "tournament": tournament_config_to_dict(config.tournament),
        "belief": belief_config_to_dict(config.belief),
        "evolution": evolution_config_to_dict(config.evolution),
    }


def run_config_from_dict(payload: object, *, field: str = "config") -> RunConfig:
    """Decode a whole run configuration."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, RUN_CONFIG_KEYS, field=field)
    corpus_payload = require_mapping(
        present_value(mapping, "corpus", field=field), field=f"{field}.corpus"
    )
    reject_unknown_keys(corpus_payload, SYNTHETIC_CONFIG_KEYS, field=f"{field}.corpus")
    return RunConfig(
        seed=require_int(mapping, "seed", field=field),
        stages=require_str_tuple(mapping, "stages", field=field, minimum_items=1),
        run_id=require_str(mapping, "run_id", field=field),
        corpus=SyntheticConfig(
            seed=require_int(corpus_payload, "seed", field=f"{field}.corpus"),
            chains=require_int(
                corpus_payload, "chains", field=f"{field}.corpus", minimum=1
            ),
            chain_length=require_int(
                corpus_payload, "chain_length", field=f"{field}.corpus", minimum=2
            ),
            paraphrases_per_link=require_int(
                corpus_payload,
                "paraphrases_per_link",
                field=f"{field}.corpus",
                minimum=1,
            ),
            competitors_per_chain=require_int(
                corpus_payload,
                "competitors_per_chain",
                field=f"{field}.corpus",
                minimum=0,
            ),
            contradictions_per_chain=require_int(
                corpus_payload,
                "contradictions_per_chain",
                field=f"{field}.corpus",
                minimum=0,
            ),
            distractor_documents=require_int(
                corpus_payload,
                "distractor_documents",
                field=f"{field}.corpus",
                minimum=0,
            ),
            filler_sentences=require_int(
                corpus_payload, "filler_sentences", field=f"{field}.corpus", minimum=0
            ),
            source_tag=require_str(
                corpus_payload, "source_tag", field=f"{field}.corpus"
            ),
        ),
        grounding=verifier_config_from_dict(
            present_value(mapping, "grounding", field=field), field=f"{field}.grounding"
        ),
        dedup=dedup_config_from_dict(
            present_value(mapping, "dedup", field=field), field=f"{field}.dedup"
        ),
        debate=debate_config_from_dict(
            present_value(mapping, "debate", field=field), field=f"{field}.debate"
        ),
        tournament=tournament_config_from_dict(
            present_value(mapping, "tournament", field=field),
            field=f"{field}.tournament",
        ),
        belief=belief_config_from_dict(
            present_value(mapping, "belief", field=field), field=f"{field}.belief"
        ),
        evolution=evolution_config_from_dict(
            present_value(mapping, "evolution", field=field), field=f"{field}.evolution"
        ),
    )


COST_ENTRY_KEYS = (
    "stage",
    "agent",
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
COST_LEDGER_KEYS = ("entries", "totals", "by_stage")


def cost_entry_to_dict(entry: CostEntry) -> dict[str, Any]:
    """Encode one ledger entry."""
    return dict(entry.as_dict())


def cost_entry_from_dict(payload: object, *, field: str = "entry") -> CostEntry:
    """Decode one ledger entry."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, COST_ENTRY_KEYS, field=field)
    return CostEntry(
        stage=require_str(mapping, "stage", field=field),
        agent=require_str(mapping, "agent", field=field),
        calls=require_int(mapping, "calls", field=field, minimum=0),
        prompt_tokens=require_int(mapping, "prompt_tokens", field=field, minimum=0),
        completion_tokens=require_int(
            mapping, "completion_tokens", field=field, minimum=0
        ),
    )


def cost_ledger_to_dict(ledger: CostLedger) -> dict[str, Any]:
    """Encode a whole ledger; totals are recomputed and checked on decode."""
    return {
        "entries": [cost_entry_to_dict(entry) for entry in ledger.entries],
        "totals": ledger.totals(),
        "by_stage": ledger.by_stage(),
    }


def cost_ledger_from_dict(payload: object, *, field: str = "ledger") -> CostLedger:
    """Decode a ledger, verifying the stored totals against its entries."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, COST_LEDGER_KEYS, field=field)
    ledger = CostLedger(
        entries=[
            cost_entry_from_dict(item, field=f"{field}.entries[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "entries", field=field)
            )
        ]
    )
    stored = require_mapping(
        present_value(mapping, "totals", field=field), field=f"{field}.totals"
    )
    if dict(stored) != ledger.totals():
        raise SchemaError(
            "ledger totals disagree with its entries",
            field=field,
            stored=dict(stored),
            computed=ledger.totals(),
        )
    return ledger


STAGE_RESULT_KEYS = ("stage", "records", "artifacts", "skipped")
RUN_SUMMARY_KEYS = (
    "run_id",
    "config_fingerprint",
    "completed",
    "stages",
    "executed",
    "skipped",
)


def stage_result_from_dict(payload: object, *, field: str = "stage") -> StageResult:
    """Decode one stage result."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, STAGE_RESULT_KEYS, field=field)
    return StageResult(
        stage=require_str(mapping, "stage", field=field),
        records=require_int(mapping, "records", field=field, minimum=0),
        artifacts=require_str_tuple(mapping, "artifacts", field=field),
        skipped=require_bool(mapping, "skipped", field=field),
    )


def run_summary_from_dict(payload: object, *, field: str = "summary") -> RunSummary:
    """Decode a run summary written by the pipeline."""
    mapping = require_mapping(payload, field=field)
    reject_unknown_keys(mapping, RUN_SUMMARY_KEYS, field=field)
    return RunSummary(
        run_id=require_str(mapping, "run_id", field=field),
        config_fingerprint=require_str(mapping, "config_fingerprint", field=field),
        completed=require_bool(mapping, "completed", field=field),
        stages=tuple(
            stage_result_from_dict(item, field=f"{field}.stages[{index}]")
            for index, item in enumerate(
                require_mapping_list(mapping, "stages", field=field)
            )
        ),
    )
