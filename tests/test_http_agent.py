"""Adapter behaviour against a scripted transport (no sockets involved)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from hypoarena.errors import TransportError
from hypoarena.http_agent import HttpAgent, HttpConfig


def body(text: str = "a hypothesis", prompt_tokens: int = 9) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "model": "mock-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 3},
        }
    )


class ScriptedTransport:
    """Returns queued (status, body) pairs and records what was sent."""

    def __init__(self, replies: list[tuple[int, str]]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, str]:
        self.calls.append(
            {
                "url": url,
                "payload": dict(payload),
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        if not self.replies:
            raise AssertionError("transport called more often than scripted")
        return self.replies.pop(0)


def test_a_successful_call_returns_the_upstream_text() -> None:
    transport = ScriptedTransport([(200, body("kinase K1 increases protein A"))])
    agent = HttpAgent(HttpConfig(), transport=transport)
    response = agent.propose("propose", ("kinase K1 phosphorylates protein A",))
    assert response.text == "kinase K1 increases protein A"
    assert response.agent == "http"
    assert response.model == "mock-model"
    assert response.prompt_tokens == 9
    assert response.completion_tokens == 3


def test_the_payload_carries_model_messages_and_context() -> None:
    transport = ScriptedTransport([(200, body())])
    agent = HttpAgent(
        HttpConfig(model="demo-model"), transport=transport, system_prompt="be brief"
    )
    agent.propose("propose a hypothesis", ("line one", "line two"))
    sent = transport.calls[0]["payload"]
    assert sent["model"] == "demo-model"
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["messages"][1]["content"] == "propose a hypothesis\nline one\nline two"
    assert transport.calls[0]["url"].endswith("/chat/completions")
    assert transport.calls[0]["timeout"] == 5.0


def test_request_options_are_forwarded() -> None:
    from hypoarena.agents import AgentRequest

    transport = ScriptedTransport([(200, body())])
    agent = HttpAgent(HttpConfig(), transport=transport)
    request = AgentRequest("propose", "p", (), temperature=0.4, max_tokens=32).with_id()
    agent.respond(request)
    sent = transport.calls[0]["payload"]
    assert sent["temperature"] == 0.4
    assert sent["max_tokens"] == 32


def test_error_statuses_become_transport_errors() -> None:
    transport = ScriptedTransport([(404, "not found")])
    agent = HttpAgent(HttpConfig(max_retries=0), transport=transport)
    with pytest.raises(TransportError) as info:
        agent.propose("propose")
    assert info.value.status == 404
    assert info.value.attempts == 1


def test_usage_is_recorded_for_successful_calls_only() -> None:
    transport = ScriptedTransport([(404, "nope")])
    agent = HttpAgent(HttpConfig(max_retries=0), transport=transport)
    with pytest.raises(TransportError):
        agent.propose("propose")
    assert agent.usage.calls == 0
