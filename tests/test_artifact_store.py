"""Artifact store: naming rules, atomic writes and checkpoint markers."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.artifacts import ArtifactStore, check_artifact_name, check_stage_name
from hypoarena.errors import ArtifactError, ValidationError


def store(tmp_path: Path, run_id: str = "run") -> ArtifactStore:
    return ArtifactStore(tmp_path, run_id)


def test_paths_are_scoped_to_the_run_directory(tmp_path: Path) -> None:
    produced = store(tmp_path)
    assert produced.root == tmp_path / "run"
    assert produced.path("corpus.jsonl") == tmp_path / "run" / "corpus.jsonl"


def test_run_ids_must_be_a_single_segment(tmp_path: Path) -> None:
    for bad in ("", "  ", "a/b", "..", "a\\b"):
        with pytest.raises(ValidationError, match="single path segment"):
            ArtifactStore(tmp_path, bad)


def test_artifact_names_are_validated() -> None:
    assert check_artifact_name("corpus.jsonl") == "corpus.jsonl"
    assert check_artifact_name("run.json") == "run.json"
    for bad in ("../escape.jsonl", "a/b.jsonl", "noext", "", ".jsonl"):
        with pytest.raises(ValidationError, match="flat file names"):
            check_artifact_name(bad)


def test_stage_names_are_validated() -> None:
    assert check_stage_name("corpus") == "corpus"
    with pytest.raises(ValidationError, match="unknown pipeline stage"):
        check_stage_name("publish")


def test_writing_and_reading_lines_roundtrips(tmp_path: Path) -> None:
    produced = store(tmp_path)
    assert produced.write_lines("corpus.jsonl", ['{"a": 1}\n', '{"b": 2}\n']) == 2
    assert produced.read_lines("corpus.jsonl") == ['{"a": 1}\n', '{"b": 2}\n']
    assert produced.exists("corpus.jsonl") is True
    assert produced.listing() == ("corpus.jsonl",)


def test_json_artifacts_hold_exactly_one_object(tmp_path: Path) -> None:
    produced = store(tmp_path)
    produced.write_json("summary.json", {"stage": "corpus", "records": 2})
    assert produced.read_json("summary.json") == {"stage": "corpus", "records": 2}
    produced.write_lines("bad.json", ['{"a": 1}\n', '{"b": 2}\n'])
    with pytest.raises(ArtifactError, match="exactly one line"):
        produced.read_json("bad.json")


def test_missing_artifacts_raise(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        store(tmp_path).read_lines("absent.jsonl")


def test_no_partial_files_are_left_behind(tmp_path: Path) -> None:
    produced = store(tmp_path)
    produced.write_lines("corpus.jsonl", ['{"a": 1}\n'])
    assert list(tmp_path.rglob("*.partial")) == []


def test_checkpoints_mark_and_clear_stages(tmp_path: Path) -> None:
    produced = store(tmp_path)
    assert produced.stage_done("corpus") is False
    produced.mark_stage("corpus")
    produced.mark_stage("verify")
    assert produced.stage_done("corpus") is True
    assert produced.completed_stages() == ("corpus", "verify")
    assert produced.clear_stage("corpus") is True
    assert produced.clear_stage("corpus") is False
    assert produced.completed_stages() == ("verify",)
    assert produced.clear_checkpoints() == 1
    assert produced.completed_stages() == ()


def test_an_empty_store_lists_nothing(tmp_path: Path) -> None:
    assert store(tmp_path).listing() == ()
