"""Retry policy: retryable statuses, budgets, backoff and error reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from hypoarena.errors import TransportError, ValidationError
from hypoarena.http_agent import HttpAgent, HttpConfig, retry_budget, retry_delay


def body(text: str = "ok") -> str:
    return json.dumps(
        {
            "model": "mock-model",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
    )


class QueuedTransport:
    def __init__(self, replies: list[tuple[int, str]]) -> None:
        self.replies = replies
        self.attempts = 0

    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, str]:
        self.attempts += 1
        status, reply = self.replies.pop(0)
        return status, reply


def test_retryable_statuses_are_retried_until_success() -> None:
    transport = QueuedTransport([(503, ""), (429, ""), (200, body("recovered"))])
    agent = HttpAgent(HttpConfig(max_retries=3), transport=transport)
    assert agent.propose("propose").text == "recovered"
    assert transport.attempts == 3


def test_the_retry_budget_bounds_the_attempts() -> None:
    transport = QueuedTransport([(500, "")] * 5)
    agent = HttpAgent(HttpConfig(max_retries=2), transport=transport)
    with pytest.raises(TransportError) as info:
        agent.propose("propose")
    assert info.value.status == 500
    assert info.value.attempts == 3
    assert transport.attempts == 3
    assert retry_budget(HttpConfig(max_retries=2)) == 3


def test_non_retryable_statuses_fail_immediately() -> None:
    transport = QueuedTransport([(404, "missing"), (200, body())])
    agent = HttpAgent(HttpConfig(max_retries=3), transport=transport)
    with pytest.raises(TransportError, match="HTTP 404"):
        agent.propose("propose")
    assert transport.attempts == 1


def test_retry_statuses_are_configurable() -> None:
    transport = QueuedTransport([(418, ""), (200, body())])
    agent = HttpAgent(
        HttpConfig(max_retries=1, retry_statuses=(418,)), transport=transport
    )
    assert agent.propose("propose").text == "ok"
    assert transport.attempts == 2


def test_backoff_is_linear_and_validated() -> None:
    config = HttpConfig(retry_backoff=0.25)
    assert retry_delay(config, 1) == 0.25
    assert retry_delay(config, 3) == 0.75
    assert retry_delay(HttpConfig(), 4) == 0.0
    with pytest.raises(ValidationError, match="attempt"):
        retry_delay(config, 0)


def test_successful_retries_are_not_counted_as_failures() -> None:
    transport = QueuedTransport([(503, ""), (200, body("done"))])
    agent = HttpAgent(HttpConfig(max_retries=2), transport=transport)
    agent.propose("propose")
    assert agent.usage.calls == 1
