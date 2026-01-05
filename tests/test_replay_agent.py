"""Replay adapter: keyed lookup, fallbacks and error reporting."""

from __future__ import annotations

import pytest

from hypoarena.agents import ReplayAgent, ReplayEntry
from hypoarena.errors import AdapterError, ReplayExhaustedError, ValidationError

PROMPT = "Propose one hypothesis supported by the context."
CONTEXT = ("Protein A increases cell growth in HeLa cells.",)


def entry(**overrides: object) -> ReplayEntry:
    payload: dict[str, object] = {
        "task": "propose",
        "prompt": PROMPT,
        "text": "protein A increases cell growth",
        "context": CONTEXT,
    }
    payload.update(overrides)
    return ReplayEntry(**payload)  # type: ignore[arg-type]


def test_entries_validate_task_text_and_counters() -> None:
    assert entry().key() == ("propose", PROMPT, CONTEXT)
    with pytest.raises(ValidationError, match="unknown replay task"):
        entry(task="summarize")
    with pytest.raises(ValidationError, match="blank"):
        entry(text="   ")
    with pytest.raises(ValidationError, match="prompt_tokens"):
        entry(prompt_tokens=-1)


def test_keyed_replay_returns_the_recorded_text() -> None:
    agent = ReplayAgent("fixture", [entry()])
    response = agent.propose(PROMPT, CONTEXT)
    assert response.text == "protein A increases cell growth"
    assert response.agent == "fixture"
    assert response.model == "replay"


def test_keyed_replay_requires_an_exact_context_match() -> None:
    agent = ReplayAgent("fixture", [entry()])
    with pytest.raises(ReplayExhaustedError) as info:
        agent.propose(PROMPT, ())
    assert info.value.details["task"] == "propose"
    with pytest.raises(ReplayExhaustedError):
        agent.propose("a different prompt", CONTEXT)


def test_fallback_answers_unknown_requests() -> None:
    agent = ReplayAgent("fixture", [entry()], fallback="nothing recorded")
    response = agent.propose("unrecorded prompt")
    assert response.text == "nothing recorded"
    assert response.model == "replay-fallback"


def test_token_counts_default_to_word_counts_and_can_be_pinned() -> None:
    derived = ReplayAgent("a", [entry()]).propose(PROMPT, CONTEXT)
    assert derived.prompt_tokens == len(PROMPT.split()) + len(CONTEXT[0].split())
    assert derived.completion_tokens == 5
    pinned = ReplayAgent("a", [entry(prompt_tokens=11, completion_tokens=2)])
    response = pinned.propose(PROMPT, CONTEXT)
    assert (response.prompt_tokens, response.completion_tokens) == (11, 2)


def test_duplicate_keys_are_rejected_in_keyed_mode() -> None:
    with pytest.raises(ValidationError, match="duplicate replay entry"):
        ReplayAgent("fixture", [entry(), entry(text="another reply")])


def test_unknown_modes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown replay mode"):
        ReplayAgent("fixture", [], mode="shuffled")


def test_replayed_usage_is_recorded_like_any_other_adapter() -> None:
    agent = ReplayAgent("fixture", [entry()])
    agent.propose(PROMPT, CONTEXT)
    agent.propose(PROMPT, CONTEXT)
    assert agent.usage.by_task == {"propose": 2}
    assert agent.remaining == 1


def test_unsupported_tasks_are_reported_not_guessed() -> None:
    agent = ReplayAgent("fixture", [entry()])
    with pytest.raises(AdapterError):
        agent.run("judge", PROMPT, CONTEXT)
