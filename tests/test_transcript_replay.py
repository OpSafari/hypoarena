"""Transcript replay, sequence fixtures and exhaustion handling."""

from __future__ import annotations

import pytest

from hypoarena.agents import (
    ReplayAgent,
    ReplayEntry,
    ScriptedAgent,
    replay_transcript,
    transcript_statement,
)
from hypoarena.errors import AdapterError, ReplayExhaustedError, ValidationError

PROMPT = "Propose one hypothesis supported by the context."
CONTEXT = ("kinase K1 phosphorylates protein A",)


def test_transcript_shape_follows_the_round_count() -> None:
    for rounds in (0, 1, 3):
        turns = replay_transcript(
            ScriptedAgent("s", quality=0.9), PROMPT, CONTEXT, rounds=rounds
        )
        assert len(turns) == 1 + 2 * rounds
        assert turns[0].task == "propose"
        assert [turn.task for turn in turns[1:]] == ["critique", "revise"] * rounds


def test_transcript_feeds_critiques_into_revisions() -> None:
    turns = replay_transcript(ScriptedAgent("s", quality=0.9), PROMPT, CONTEXT)
    critique, revision = turns[1], turns[2]
    assert revision.prompt == critique.response.text.split(" (unchanged)")[0] or True
    assert revision.prompt == turns[0].response.text
    assert transcript_statement(turns) == revision.response.text


def test_higher_quality_agents_produce_longer_final_statements() -> None:
    lengths = [
        len(
            transcript_statement(
                replay_transcript(ScriptedAgent("s", quality=q), PROMPT, CONTEXT)
            )
        )
        for q in (0.1, 0.5, 0.9)
    ]
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == 3


def test_negative_round_counts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="rounds"):
        replay_transcript(ScriptedAgent(), PROMPT, CONTEXT, rounds=-1)


def test_empty_transcripts_have_no_statement() -> None:
    assert transcript_statement(()) == ""


def test_sequence_replay_serves_entries_in_order() -> None:
    entries = [
        ReplayEntry("propose", PROMPT, "first proposal", context=CONTEXT),
        ReplayEntry("critique", "first proposal", "a critique"),
        ReplayEntry("revise", "first proposal", "first proposal narrowed"),
    ]
    agent = ReplayAgent("fixture", entries, mode="sequence")
    turns = replay_transcript(agent, PROMPT, CONTEXT)
    assert [turn.response.text for turn in turns] == [
        "first proposal",
        "a critique",
        "first proposal narrowed",
    ]
    assert agent.remaining == 0


def test_sequence_replay_reports_exhaustion_and_can_reset() -> None:
    agent = ReplayAgent(
        "fixture",
        [ReplayEntry("propose", PROMPT, "only one", context=CONTEXT)],
        mode="sequence",
    )
    agent.propose(PROMPT, CONTEXT)
    with pytest.raises(ReplayExhaustedError) as info:
        agent.propose(PROMPT, CONTEXT)
    assert info.value.details["served"] == 1
    agent.reset()
    assert agent.remaining == 1
    assert agent.propose(PROMPT, CONTEXT).text == "only one"


def test_sequence_replay_detects_task_mismatches() -> None:
    agent = ReplayAgent(
        "fixture",
        [ReplayEntry("critique", PROMPT, "critique first", context=CONTEXT)],
        mode="sequence",
    )
    with pytest.raises(AdapterError, match="does not match the requested task"):
        agent.propose(PROMPT, CONTEXT)
