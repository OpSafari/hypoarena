"""Corpus containers: documents and the span resolution built on them.

A corpus is the ground truth for grounding checks: claims cite
:class:`~hypoarena.schema.Citation` spans, and this module is what turns a
citation back into the exact characters the document contains. Documents are
immutable, so a corpus hash recorded in provenance stays meaningful.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from random import Random

from hypoarena.errors import (
    CorpusError,
    DuplicateIdError,
    SpanNotFoundError,
    UnknownReferenceError,
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
    is_valid_id,
)
from hypoarena.schema import (
    Citation,
)

DOCUMENT_ID_PREFIX = "doc"


@dataclass(frozen=True)
class Document:
    """One paper-like record: an identifier, a title and its full text.

    ``source`` names where the text came from (for bundled corpora this is a
    synthetic generator tag such as ``synthetic:v1``), and ``attributes`` holds
    flat, JSON-safe metadata such as the planted chain a document belongs to.
    """

    document_id: str
    title: str
    text: str
    source: str = "unspecified"
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not is_valid_id(self.document_id) or not self.document_id.startswith(
            f"{DOCUMENT_ID_PREFIX}_"
        ):
            raise ValidationError(
                "document id is malformed",
                document_id=self.document_id,
                expected_prefix=f"{DOCUMENT_ID_PREFIX}_",
            )
        if not self.title.strip():
            raise ValidationError(
                "document title must not be blank", document_id=self.document_id
            )
        if not self.text.strip():
            raise ValidationError(
                "document text must not be blank", document_id=self.document_id
            )
        for key, _ in self.attributes:
            if not key.strip():
                raise ValidationError(
                    "document attribute keys must not be blank",
                    document_id=self.document_id,
                )
        keys = [key for key, _ in self.attributes]
        if len(set(keys)) != len(keys):
            raise ValidationError(
                "document attributes contain duplicate keys",
                document_id=self.document_id,
                keys=sorted({key for key in keys if keys.count(key) > 1}),
            )

    @property
    def length(self) -> int:
        """Number of characters in the document text."""
        return len(self.text)

    def span_text(self, start: int, end: int) -> str:
        """Return ``text[start:end]``, rejecting offsets outside the document."""
        if start < 0 or end > len(self.text) or end <= start:
            raise ValidationError(
                "span is outside the document",
                document_id=self.document_id,
                start=start,
                end=end,
                length=len(self.text),
            )
        return self.text[start:end]

    def attribute(self, key: str) -> str | None:
        """Return a metadata value by key, or ``None`` when absent."""
        for name, value in self.attributes:
            if name == key:
                return value
        return None

    def attribute_map(self) -> dict[str, str]:
        """Return the metadata as a plain dictionary."""
        return dict(self.attributes)

    def find_all(self, needle: str) -> list[int]:
        """Return every start offset where ``needle`` occurs, in order."""
        if not needle:
            return []
        offsets = []
        start = self.text.find(needle)
        while start != -1:
            offsets.append(start)
            start = self.text.find(needle, start + 1)
        return offsets

    def locate(self, quote: str) -> tuple[int, int] | None:
        """Return the half-open span of the first occurrence, or ``None``."""
        offsets = self.find_all(quote)
        if not offsets:
            return None
        return (offsets[0], offsets[0] + len(quote))

    def contains_span(self, citation: Citation) -> bool:
        """True when the citation's offsets hold exactly the quoted text."""
        if citation.document_id != self.document_id:
            return False
        if citation.end > len(self.text) or citation.start < 0:
            return False
        return self.text[citation.start : citation.end] == citation.quote

    def citation_spans(self, quote: str) -> list[tuple[int, int]]:
        """Return every span in which ``quote`` occurs verbatim."""
        return [(start, start + len(quote)) for start in self.find_all(quote)]


class Corpus:
    """An immutable-document collection keyed by identifier.

    Documents cannot be replaced in place: revising text would invalidate every
    citation that points into it, so updates happen by building a new corpus.
    Iteration is sorted by identifier for deterministic output.
    """

    def __init__(self, documents: Iterable[Document] = ()) -> None:
        self._documents: dict[str, Document] = {}
        for document in documents:
            self.add_document(document)

    def add_document(self, document: Document) -> Document:
        """Insert ``document`` and return it."""
        if not isinstance(document, Document):
            raise ValidationError(
                "corpus entries must be Document objects",
                got=type(document).__name__,
            )
        if document.document_id in self._documents:
            raise DuplicateIdError(document.document_id, "document")
        self._documents[document.document_id] = document
        return document

    def has_document(self, document_id: str) -> bool:
        """True when the identifier is present."""
        return document_id in self._documents

    def document(self, document_id: str) -> Document:
        """Return the document or raise ``UnknownReferenceError``."""
        try:
            return self._documents[document_id]
        except KeyError:
            raise UnknownReferenceError(document_id, "document") from None

    @property
    def document_ids(self) -> tuple[str, ...]:
        """All identifiers in sorted order."""
        return tuple(sorted(self._documents))

    @property
    def documents(self) -> tuple[Document, ...]:
        """All documents ordered by identifier."""
        return tuple(self._documents[item] for item in self.document_ids)

    def text_of(self, document_id: str) -> str:
        """Return the full text of one document."""
        return self.document(document_id).text

    def __len__(self) -> int:
        return len(self._documents)

    def __contains__(self, document_id: object) -> bool:
        return isinstance(document_id, str) and document_id in self._documents

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def stats(self) -> CorpusStats:
        """Summarize corpus shape; an empty corpus reports zeros."""
        lengths = [document.length for document in self._documents.values()]
        sources: dict[str, int] = {}
        for document in self._documents.values():
            sources[document.source] = sources.get(document.source, 0) + 1
        characters = sum(lengths)
        return CorpusStats(
            documents=len(lengths),
            characters=characters,
            shortest=min(lengths, default=0),
            longest=max(lengths, default=0),
            mean_length=round(characters / len(lengths), 3) if lengths else 0.0,
            sources=tuple(sorted(sources.items())),
        )

    def signature(self) -> str:
        """Return a content digest over every document, in sorted order.

        This is the value recorded as ``corpus_hash`` in provenance: two corpora
        with identical content hash identically, so a claim's provenance can be
        checked against the corpus it was grounded in.
        """
        return content_hash(
            [
                [
                    document.document_id,
                    content_hash(
                        {
                            "title": document.title,
                            "text": document.text,
                            "source": document.source,
                            "attributes": [list(pair) for pair in document.attributes],
                        }
                    ),
                ]
                for document in self.documents
            ]
        )

    def resolve(self, citation: Citation) -> str:
        """Return the exact text a citation points at.

        This is the ground truth used by grounding verification: a citation is
        only resolvable when the document exists, the offsets are inside it and
        the characters at those offsets equal the quoted text. A citation that
        quotes something else — the classic fabricated reference — raises
        :class:`~hypoarena.errors.SpanNotFoundError` with both strings attached.
        """
        document = self.document(citation.document_id)
        if citation.start < 0 or citation.end > document.length:
            raise SpanNotFoundError(
                "citation offsets are outside the document",
                document_id=citation.document_id,
                start=citation.start,
                end=citation.end,
                length=document.length,
            )
        found = document.text[citation.start : citation.end]
        if found != citation.quote:
            raise SpanNotFoundError(
                "citation quote does not match the document text",
                document_id=citation.document_id,
                start=citation.start,
                end=citation.end,
                quoted=citation.quote,
                found=found,
            )
        return found

    def verify_quote(self, citation: Citation) -> bool:
        """Non-raising variant of :meth:`resolve` for bulk checks."""
        try:
            self.resolve(citation)
        except (SpanNotFoundError, UnknownReferenceError):
            return False
        return True

    def find_quote(self, document_id: str, quote: str) -> tuple[int, int] | None:
        """Return the first span of ``quote`` in one document, if present."""
        return self.document(document_id).locate(quote)

    def search_quote(self, quote: str) -> list[Citation]:
        """Return every citation in the corpus whose text equals ``quote``.

        Results are ordered by document identifier then offset, so the same query
        always yields the same list.
        """
        found: list[Citation] = []
        for document in self.documents:
            for start, end in document.citation_spans(quote):
                found.append(Citation(document.document_id, start, end, quote))
        return found

    def subcorpus(self, document_ids: Iterable[str]) -> Corpus:
        """Return a new corpus holding exactly the requested documents."""
        wanted = set(document_ids)
        for document_id in wanted:
            self.document(document_id)
        return Corpus([self._documents[item] for item in sorted(wanted)])

    def merged(self, other: Corpus, *, on_conflict: str = "error") -> Corpus:
        """Return the union of two corpora.

        Identical documents (same identifier, same content) are shared. When the
        same identifier carries different content, ``on_conflict="error"`` raises
        :class:`~hypoarena.errors.CorpusError` — the default, because silently
        choosing a side would invalidate citations — while
        ``on_conflict="prefer_self"`` keeps this corpus's version.
        """
        if on_conflict not in ("error", "prefer_self"):
            raise ValidationError(
                "unknown conflict policy",
                on_conflict=on_conflict,
                allowed=["error", "prefer_self"],
            )
        result = Corpus(self.documents)
        for document in other.documents:
            if not result.has_document(document.document_id):
                result.add_document(document)
                continue
            if result.document(document.document_id) == document:
                continue
            if on_conflict == "error":
                raise CorpusError(
                    "conflicting document content during merge",
                    document_id=document.document_id,
                )
        return result

    def documents_for_attribute(self, key: str, value: str) -> tuple[Document, ...]:
        """Return documents whose metadata carries ``key == value``, sorted."""
        return tuple(
            document for document in self.documents if document.attribute(key) == value
        )


@dataclass(frozen=True)
class CorpusStats:
    """Shape summary of a corpus, used in reports and run metadata."""

    documents: int
    characters: int
    shortest: int
    longest: int
    mean_length: float
    sources: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view with sources as an object."""
        return {
            "documents": self.documents,
            "characters": self.characters,
            "shortest": self.shortest,
            "longest": self.longest,
            "mean_length": self.mean_length,
            "sources": dict(self.sources),
        }


WORD_SPAN_PATTERN = re.compile(r"\S+")
DEFAULT_MIN_SPAN_LENGTH = 12
DEFAULT_MAX_SPAN_LENGTH = 80


def word_spans(document: Document) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` offsets of every whitespace-delimited token."""
    return [
        (match.start(), match.end())
        for match in WORD_SPAN_PATTERN.finditer(document.text)
    ]


def sample_spans(
    document: Document,
    rng: Random,
    count: int,
    *,
    min_length: int = DEFAULT_MIN_SPAN_LENGTH,
    max_length: int = DEFAULT_MAX_SPAN_LENGTH,
) -> list[Citation]:
    """Sample up to ``count`` non-overlapping citations from one document.

    Spans start and end on word boundaries so quotes never cut a token in half,
    and they are returned sorted by offset. All randomness comes from ``rng``, so
    a fixed seed always reproduces the same spans — the property the synthetic
    literature factory and every grounded test depend on.

    Documents that are too short to satisfy ``min_length`` yield a single span
    covering the whole text (or nothing at all when even that is shorter than the
    minimum), which keeps callers free of special cases.
    """
    if count < 0:
        raise ValidationError("span count must be >= 0", count=count)
    if min_length < 1:
        raise ValidationError("minimum span length must be >= 1", min_length=min_length)
    if max_length < min_length:
        raise ValidationError(
            "maximum span length must be >= minimum",
            min_length=min_length,
            max_length=max_length,
        )
    spans = word_spans(document)
    if not spans:
        return []
    if document.length < min_length:
        return (
            [Citation(document.document_id, 0, document.length, document.text)]
            if document.length >= min_length
            else []
        )
    chosen: list[tuple[int, int]] = []
    attempts = 0
    limit = max(20, count * 20)
    while len(chosen) < count and attempts < limit:
        attempts += 1
        start_index = rng.randrange(len(spans))
        start = spans[start_index][0]
        end = start
        for index in range(start_index, len(spans)):
            candidate = spans[index][1]
            if candidate - start > max_length:
                break
            end = candidate
            if end - start >= min_length:
                break
        if end - start < min_length:
            continue
        if any(
            start < existing_end and existing_start < end
            for existing_start, existing_end in chosen
        ):
            continue
        chosen.append((start, end))
    return [
        Citation(document.document_id, start, end, document.text[start:end])
        for start, end in sorted(chosen)
    ]
