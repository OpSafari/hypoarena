"""Text artifacts: name validation, atomic write and exact roundtrip."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena.artifacts import ArtifactStore, check_artifact_name
from hypoarena.errors import ArtifactError, ValidationError


@pytest.mark.parametrize(
    "name", ["report.md", "report.html", "corpus.jsonl", "run.json", "a-b_c.md"]
)
def test_text_and_json_artifact_names_are_accepted(name: str) -> None:
    assert check_artifact_name(name) == name


@pytest.mark.parametrize(
    "name", ["noext", "a/b.md", "../x.html", "", ".md", "rep ort.md"]
)
def test_unsafe_artifact_names_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        check_artifact_name(name)


def test_write_text_roundtrips_exactly(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run")
    payload = "# Title\n\n- café 文 🎉\n<table>&amp;</table>\n"
    store.write_text("report.md", payload)
    assert store.read_text("report.md") == payload


def test_write_text_leaves_no_partial_file(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run")
    store.write_text("report.html", "<html></html>")
    leftovers = list((tmp_path / "run").glob("*.partial"))
    assert leftovers == []
    assert store.exists("report.html")


def test_read_text_raises_when_missing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run")
    with pytest.raises(ArtifactError):
        store.read_text("absent.md")


def test_write_text_is_listed_alongside_json_artifacts(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run")
    store.write_json("run.json", {"a": 1})
    store.write_text("report.md", "# hi")
    assert store.listing() == ("report.md", "run.json")
