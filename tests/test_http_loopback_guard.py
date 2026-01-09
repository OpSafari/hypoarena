"""Endpoint guard: local mocks are fine, remote hosts need consent."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from hypoarena.errors import TransportError
from hypoarena.http_agent import HttpAgent, HttpConfig, check_endpoint, is_loopback


def body() -> str:
    return json.dumps(
        {
            "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {},
        }
    )


class RecordingTransport:
    """Transport double that only counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, str]:
        self.calls += 1
        return 200, body()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:8000/v1/chat/completions", True),
        ("http://localhost:9/v1/chat/completions", True),
        ("http://[::1]:9/v1/chat/completions", True),
        ("https://api.example.com/v1/chat/completions", False),
        ("https://10.0.0.5/v1/chat/completions", False),
    ],
)
def test_is_loopback_classifies_hosts(url: str, expected: bool) -> None:
    assert is_loopback(url) is expected


def test_check_endpoint_allows_loopback_and_explicit_remote() -> None:
    check_endpoint("http://127.0.0.1:9/v1", allow_remote=False)
    check_endpoint("https://api.example.com/v1", allow_remote=True)


def test_check_endpoint_refuses_remote_by_default() -> None:
    with pytest.raises(TransportError, match="non-loopback"):
        check_endpoint("https://api.example.com/v1", allow_remote=False)


def test_the_agent_never_posts_to_a_remote_host_unless_allowed() -> None:
    transport = RecordingTransport()
    agent = HttpAgent(
        HttpConfig(base_url="https://api.example.com/v1"), transport=transport
    )
    with pytest.raises(TransportError):
        agent.propose("propose")
    assert transport.calls == 0
    assert agent.usage.calls == 0


def test_the_agent_posts_to_loopback_without_extra_consent() -> None:
    transport = RecordingTransport()
    agent = HttpAgent(HttpConfig(), transport=transport)
    assert agent.propose("propose").text == "ok"
    assert transport.calls == 1


def test_remote_calls_succeed_once_explicitly_allowed() -> None:
    transport = RecordingTransport()
    agent = HttpAgent(
        HttpConfig(base_url="https://api.example.com/v1", allow_remote=True),
        transport=transport,
    )
    assert agent.propose("propose").text == "ok"
    assert transport.calls == 1
