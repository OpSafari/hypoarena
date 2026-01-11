"""Failure paths over a real loopback socket.

Status codes, malformed bodies and slow handlers are produced by the mock server
so the retry policy is exercised end to end. Timeouts use a 0.1 s budget, which
keeps the suite fast while still going through ``socket.timeout``.
"""

from __future__ import annotations

import pytest

from hypoarena.errors import TransportError
from hypoarena.http_agent import HttpAgent, HttpConfig
from loopback import MockChatServer, chat_body


def agent_for(server: MockChatServer, **overrides: object) -> HttpAgent:
    config = HttpConfig(base_url=server.base_url, **overrides)  # type: ignore[arg-type]
    return HttpAgent(config, name="loopback")


def test_retryable_statuses_are_retried_until_a_reply_arrives() -> None:
    replies = [
        (503, {"error": "busy"}),
        (429, {"error": "slow down"}),
        (200, chat_body("late")),
    ]
    with MockChatServer(replies) as server:
        response = agent_for(server, max_retries=3).propose("propose")
        assert len(server.received) == 3
    assert response.text == "late"


def test_exhausted_retries_report_the_last_status() -> None:
    replies = [(500, {"error": "boom"})] * 4
    with MockChatServer(replies) as server:
        with pytest.raises(TransportError) as info:
            agent_for(server, max_retries=2).propose("propose")
        assert len(server.received) == 3
    assert info.value.status == 500
    assert info.value.attempts == 3


def test_client_errors_are_not_retried() -> None:
    with MockChatServer([(404, {"error": "missing"})]) as server:
        with pytest.raises(TransportError, match="HTTP 404"):
            agent_for(server, max_retries=3).propose("propose")
        assert len(server.received) == 1


def test_malformed_json_bodies_are_reported() -> None:
    with (
        MockChatServer([(200, b"{not json")]) as server,
        pytest.raises(TransportError, match="invalid JSON"),
    ):
        agent_for(server).propose("propose")


def test_a_response_without_choices_is_rejected() -> None:
    with (
        MockChatServer([(200, {"model": "m", "choices": []})]) as server,
        pytest.raises(TransportError, match="no choices"),
    ):
        agent_for(server).propose("propose")


def test_slow_handlers_trip_the_timeout() -> None:
    with (
        MockChatServer(delay=0.6) as server,
        pytest.raises(TransportError, match="unreachable"),
    ):
        agent_for(server, timeout=0.1, max_retries=0).propose("propose")


def test_a_backoff_of_zero_keeps_retries_instantaneous() -> None:
    replies = [(503, {}), (503, {}), (200, chat_body("finally"))]
    with MockChatServer(replies) as server:
        response = agent_for(server, max_retries=2, retry_backoff=0.0).propose(
            "propose"
        )
    assert response.text == "finally"
