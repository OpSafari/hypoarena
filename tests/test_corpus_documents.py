"""Document records: validation, span access and attribute lookup."""

from __future__ import annotations

import pytest

from hypoarena.corpus import Document
from hypoarena.errors import ValidationError

TEXT = "Protein A binds the promoter of gene B and increases its expression."


def make_document(**overrides: object) -> Document:
    payload: dict[str, object] = {
        "document_id": "doc_0123456789ab",
        "title": "Binding and expression",
        "text": TEXT,
    }
    payload.update(overrides)
    return Document(**payload)  # type: ignore[arg-type]


def test_document_exposes_length_and_span_text() -> None:
    document = make_document()
    assert document.length == len(TEXT)
    assert document.span_text(0, 9) == "Protein A"
    assert document.span_text(document.length - 1, document.length) == "."


def test_span_text_rejects_out_of_range_and_empty_spans() -> None:
    document = make_document()
    with pytest.raises(ValidationError, match="outside the document"):
        document.span_text(-1, 5)
    with pytest.raises(ValidationError, match="outside the document"):
        document.span_text(0, document.length + 1)
    with pytest.raises(ValidationError, match="outside the document"):
        document.span_text(5, 5)


def test_document_rejects_malformed_ids_and_blank_fields() -> None:
    with pytest.raises(ValidationError, match="malformed"):
        make_document(document_id="paper_0123456789ab")
    with pytest.raises(ValidationError, match="title"):
        make_document(title="  ")
    with pytest.raises(ValidationError, match="text"):
        make_document(text="")


def test_attributes_are_readable_and_validated() -> None:
    document = make_document(attributes=(("chain", "c1"), ("role", "support")))
    assert document.attribute("chain") == "c1"
    assert document.attribute("missing") is None
    assert document.attribute_map() == {"chain": "c1", "role": "support"}
    with pytest.raises(ValidationError, match="blank"):
        make_document(attributes=((" ", "x"),))
    with pytest.raises(ValidationError, match="duplicate keys"):
        make_document(attributes=(("chain", "c1"), ("chain", "c2")))


def test_documents_compare_structurally() -> None:
    assert make_document() == make_document()
    assert make_document() != make_document(source="synthetic:v1")
