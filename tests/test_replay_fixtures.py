"""Replay fixtures on disk: loading, roundtrips and integrity checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypoarena.agents import ReplayAgent, ReplayEntry, replay_transcript
from hypoarena.errors import ArtifactError, ReplayExhaustedError, SchemaError
from hypoarena.serialize import (
    read_replay_entries,
    replay_entries_from_lines,
    replay_entries_to_lines,
    write_replay_entries,
)

FIXTURE = Path(__file__).parent / "fixtures" / "replay_propose.jsonl"
PROMPT = "Propose one hypothesis supported by the context."
CONTEXT = ("kinase K1 phosphorylates protein A",)


def test_the_bundled_fixture_drives_a_full_transcript() -> None:
    entries = read_replay_entries(FIXTURE)
    assert len(entries) == 3
    agent = ReplayAgent("fixture", entries, mode="sequence")
    turns = replay_transcript(agent, PROMPT, CONTEXT)
    assert [turn.task for turn in turns] == ["propose", "critique", "revise"]
    assert turns[-1].response.text.endswith("in the assayed population")


def test_the_fixture_header_counts_entries() -> None:
    first = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    meta = json.loads(first)
    assert meta["record"] == "meta"
    assert meta["counts"] == {"entries": 3}
    assert len(meta["signature"]) == 16
    with pytest.raises(SchemaError, match="counts disagree"):
        replay_entries_from_lines([first])


def test_fixture_entries_roundtrip_through_lines() -> None:
    entries = read_replay_entries(FIXTURE)
    assert replay_entries_from_lines(replay_entries_to_lines(entries)) == entries


def test_entry_validation_survives_decoding() -> None:
    entries = read_replay_entries(FIXTURE)
    assert all(entry.task in ("propose", "critique", "revise") for entry in entries)
    line = (
        '{"record":"replay","entry":{"task":"propose","prompt":"p",'
        '"text":"t","surprise":1}}'
    )
    with pytest.raises(SchemaError, match="unknown keys"):
        replay_entries_from_lines([line])


def test_truncated_fixtures_are_detected() -> None:
    lines = replay_entries_to_lines(read_replay_entries(FIXTURE))
    with pytest.raises(SchemaError, match="counts disagree"):
        replay_entries_from_lines(lines[:-1])


def test_writing_and_reading_a_fixture_is_lossless(tmp_path: Path) -> None:
    entries = (
        ReplayEntry("propose", "prompt one", "reply one"),
        ReplayEntry("critique", "reply one", "critique one"),
    )
    target = tmp_path / "fixture.jsonl"
    assert write_replay_entries(entries, target) == 3
    assert read_replay_entries(target) == entries


def test_missing_fixtures_raise_artifact_errors(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        read_replay_entries(tmp_path / "absent.jsonl")


def test_keyed_mode_ignores_fixture_order() -> None:
    entries = read_replay_entries(FIXTURE)
    agent = ReplayAgent("fixture", entries, mode="keyed")
    assert agent.propose(PROMPT, CONTEXT).text.startswith("kinase K1 increases")
    with pytest.raises(ReplayExhaustedError):
        agent.propose("an unrecorded prompt")
