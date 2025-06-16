"""Corpus serialization: golden payloads, roundtrips and rejection paths."""

from __future__ import annotations

import pytest

from hypoarena.corpus import Corpus, Document
from hypoarena.errors import DuplicateIdError, SchemaError, ValidationError
from hypoarena.serialize import (
    corpus_from_dict,
    corpus_to_dict,
    document_from_dict,
    document_to_dict,
    pair_list_from_payload,
)


def sample() -> Document:
    return Document(
        "doc_0123456789ab",
        "Binding study",
        "Protein A binds the promoter of gene B.",
        "synthetic:v1",
        (("chain", "c1"), ("role", "support")),
    )


def test_document_to_dict_is_golden() -> None:
    assert document_to_dict(sample()) == {
        "document_id": "doc_0123456789ab",
        "title": "Binding study",
        "text": "Protein A binds the promoter of gene B.",
        "source": "synthetic:v1",
        "attributes": [["chain", "c1"], ["role", "support"]],
    }


def test_document_roundtrip_preserves_attribute_order() -> None:
    document = sample()
    restored = document_from_dict(document_to_dict(document))
    assert restored == document
    assert restored.attributes == (("chain", "c1"), ("role", "support"))


def test_document_from_dict_rejects_unknown_keys_and_blank_text() -> None:
    payload = document_to_dict(sample())
    payload["license"] = "MIT"
    with pytest.raises(SchemaError, match="unknown keys"):
        document_from_dict(payload)
    payload = document_to_dict(sample())
    payload["text"] = "   "
    with pytest.raises(ValidationError, match="text"):
        document_from_dict(payload)


def test_pair_list_decoding_rejects_malformed_entries() -> None:
    assert pair_list_from_payload({"a": [["k", "v"]]}, "a", field="doc") == (
        ("k", "v"),
    )
    assert pair_list_from_payload({"a": []}, "a", field="doc") == ()
    for bad in ([["k"]], [["k", "v", "w"]], "kv", [1, 2]):
        with pytest.raises(SchemaError):
            pair_list_from_payload({"a": bad}, "a", field="doc")


def test_corpus_to_dict_is_golden() -> None:
    payload = corpus_to_dict(Corpus([sample()]))
    assert sorted(payload) == ["documents", "schema_version"]
    assert payload["schema_version"] == "1.0"
    assert payload["documents"][0]["document_id"] == "doc_0123456789ab"


def test_corpus_roundtrip_keeps_signature() -> None:
    corpus = Corpus(
        [sample(), Document("doc_ffffffffffff", "Other", "No effect seen.")]
    )
    restored = corpus_from_dict(corpus_to_dict(corpus))
    assert restored.signature() == corpus.signature()
    assert restored.document_ids == corpus.document_ids


def test_corpus_from_dict_reports_positions_and_duplicates() -> None:
    payload = corpus_to_dict(Corpus([sample()]))
    payload["documents"][0]["title"] = " "
    with pytest.raises(ValidationError, match="title"):
        corpus_from_dict(payload)
    payload = corpus_to_dict(Corpus([sample()]))
    payload["documents"].append(payload["documents"][0])
    with pytest.raises(DuplicateIdError):
        corpus_from_dict(payload)
    payload = corpus_to_dict(Corpus([sample()]))
    payload["schema_version"] = "2.0"
    with pytest.raises(SchemaError, match="unsupported schema version"):
        corpus_from_dict(payload)
