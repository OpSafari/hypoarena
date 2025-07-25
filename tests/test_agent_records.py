"""Agent request/response records and usage accounting."""

from __future__ import annotations

import pytest

from hypoarena.agents import (
    AGENT_TASKS,
    AgentRequest,
    AgentResponse,
    Usage,
    count_words,
)
from hypoarena.errors import ValidationError


def request(**overrides: object) -> AgentRequest:
    payload: dict[str, object] = {
        "task": "propose",
        "prompt": "Propose a hypothesis about the corpus.",
        "context": ("Protein A increases cell growth.",),
    }
    payload.update(overrides)
    return AgentRequest(**payload)  # type: ignore[arg-type]


def response(**overrides: object) -> AgentResponse:
    payload: dict[str, object] = {
        "request_id": "req_0123456789ab",
        "text": "protein A increases cell growth",
        "agent": "scripted",
        "prompt_tokens": 5,
        "completion_tokens": 3,
    }
    payload.update(overrides)
    return AgentResponse(**payload)  # type: ignore[arg-type]


def test_tasks_are_golden() -> None:
    assert AGENT_TASKS == ("propose", "critique", "revise", "judge")


def test_request_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValidationError, match="unknown agent task"):
        request(task="summarize")
    with pytest.raises(ValidationError, match="prompt"):
        request(prompt="   ")
    with pytest.raises(ValidationError, match="temperature"):
        request(temperature=-0.1)
    with pytest.raises(ValidationError, match="max_tokens"):
        request(max_tokens=0)


def test_request_fingerprint_is_content_based() -> None:
    assert request().fingerprint() == request().fingerprint()
    assert request().fingerprint() != request(prompt="Different prompt").fingerprint()
    assert request().fingerprint() != request(context=()).fingerprint()


def test_with_id_is_stable_and_idempotent() -> None:
    identified = request().with_id()
    assert identified.request_id.startswith("req_")
    assert identified.with_id() is identified
    assert request(prompt="other").with_id().request_id != identified.request_id


def test_response_totals_tokens_and_validates_counters() -> None:
    assert response().total_tokens == 8
    with pytest.raises(ValidationError, match="token counts"):
        response(prompt_tokens=-1)
    with pytest.raises(ValidationError, match="name their agent"):
        response(agent=" ")


def test_usage_accumulates_per_task() -> None:
    usage = Usage()
    usage.record("propose", response())
    usage.record("propose", response(completion_tokens=4))
    usage.record("critique", response(prompt_tokens=2))
    assert usage.as_dict() == {
        "calls": 3,
        "prompt_tokens": 12,
        "completion_tokens": 10,
        "total_tokens": 22,
        "by_task": {"critique": 1, "propose": 2},
    }


def test_usage_starts_empty() -> None:
    assert Usage().as_dict() == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_task": {},
    }


def test_count_words_is_a_whitespace_counter() -> None:
    assert count_words("") == 0
    assert count_words("  three word count  ") == 3
