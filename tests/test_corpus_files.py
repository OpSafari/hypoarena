"""Corpus file I/O: atomic writes, byte stability and missing-file errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.corpus import Corpus, Document
from hypoarena.errors import ArtifactError
from hypoarena.serialize import (
    corpus_to_text,
    read_corpus,
    write_corpus,
    write_lines,
)


def sample_corpus() -> Corpus:
    return Corpus(
        [Document("doc_0123456789ab", "Binding study", "Protein A binds gene B.")]
    )


def test_write_then_read_preserves_content(tmp_path: Path) -> None:
    target = tmp_path / "corpus.jsonl"
    assert write_corpus(sample_corpus(), target) == 2
    assert read_corpus(target).signature() == sample_corpus().signature()


def test_written_bytes_match_the_in_memory_document(tmp_path: Path) -> None:
    target = tmp_path / "corpus.jsonl"
    write_corpus(sample_corpus(), target)
    assert target.read_text(encoding="utf-8") == corpus_to_text(sample_corpus())


def test_writes_are_atomic_and_leave_no_partial_files(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "corpus.jsonl"
    write_corpus(sample_corpus(), target)
    assert list(tmp_path.rglob("*.partial")) == []


def test_missing_files_raise_artifact_errors(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError) as info:
        read_corpus(tmp_path / "absent.jsonl")
    assert "absent.jsonl" in str(info.value)


def test_write_lines_normalizes_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "plain.jsonl"
    assert write_lines(["a", "b\n"], target) == 2
    assert target.read_text(encoding="utf-8") == "a\nb\n"
