"""Corpus containers: documents and the span resolution built on them.

A corpus is the ground truth for grounding checks: claims cite
:class:`~hypoarena.schema.Citation` spans, and this module is what turns a
citation back into the exact characters the document contains. Documents are
immutable, so a corpus hash recorded in provenance stays meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
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
