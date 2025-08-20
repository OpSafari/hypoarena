"""Capture-and-replay: a recording reproduces the original transcript."""

from __future__ import annotations

from hypoarena.agents import (
    RecordingAgent,
    ReplayEntry,
    ScriptedAgent,
    replay_transcript,
    transcript_statement,
)

PROMPT = "Propose one hypothesis supported by the context."
CONTEXT = ("kinase K1 phosphorylates protein A",)


def test_recording_captures_every_exchange_in_order() -> None:
    recorder = RecordingAgent(ScriptedAgent("inner", quality=0.9))
    replay_transcript(recorder, PROMPT, CONTEXT)
    entries = recorder.fixture()
    assert [entry.task for entry in entries] == ["propose", "critique", "revise"]
    assert all(isinstance(entry, ReplayEntry) for entry in entries)


def test_recorded_entries_keep_token_counts_and_the_inner_name() -> None:
    inner = ScriptedAgent("inner", quality=0.9)
    recorder = RecordingAgent(inner)
    replay_transcript(recorder, PROMPT, CONTEXT)
    entries = recorder.fixture()
    assert all(entry.agent == "inner" for entry in entries)
    assert sum(entry.prompt_tokens for entry in entries) == (
        recorder.usage.prompt_tokens
    )
    assert sum(entry.completion_tokens for entry in entries) == (
        recorder.usage.completion_tokens
    )
    # recording forwards to respond(), so the inner adapter keeps its own books
    assert inner.usage.calls == 0
    assert all(entry.completion_tokens > 0 for entry in entries)


def test_a_replayed_recording_reproduces_the_transcript() -> None:
    recorder = RecordingAgent(ScriptedAgent("inner", quality=0.9))
    original = replay_transcript(recorder, PROMPT, CONTEXT)
    replayed = replay_transcript(recorder.replay_agent(), PROMPT, CONTEXT)
    assert [turn.response.text for turn in replayed] == [
        turn.response.text for turn in original
    ]
    assert transcript_statement(replayed) == transcript_statement(original)


def test_the_recorder_accounts_for_its_own_usage() -> None:
    recorder = RecordingAgent(ScriptedAgent("inner", quality=0.5))
    replay_transcript(recorder, PROMPT, CONTEXT)
    assert recorder.usage.calls == 3
    assert recorder.usage.by_task == {"critique": 1, "propose": 1, "revise": 1}


def test_keyed_recordings_answer_repeated_requests() -> None:
    recorder = RecordingAgent(ScriptedAgent("inner", quality=0.5))
    replay_transcript(recorder, PROMPT, CONTEXT)
    agent = recorder.replay_agent(mode="keyed")
    assert agent.propose(PROMPT, CONTEXT).text == recorder.fixture()[0].text
    assert agent.propose(PROMPT, CONTEXT).text == recorder.fixture()[0].text
