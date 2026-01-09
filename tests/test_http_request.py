"""Chat request payloads: shape, validation and optional fields."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.http_agent import ChatMessage, ChatRequest


def test_message_payload_shape() -> None:
    assert ChatMessage("user", "hello").to_dict() == {
        "role": "user",
        "content": "hello",
    }


def test_messages_validate_role_and_content() -> None:
    with pytest.raises(ValidationError, match="unknown message role"):
        ChatMessage("tool", "hello")
    with pytest.raises(ValidationError, match="blank"):
        ChatMessage("user", "   ")


def test_request_payload_is_golden() -> None:
    request = ChatRequest(
        model="mock-model",
        messages=(ChatMessage("system", "be brief"), ChatMessage("user", "propose")),
        temperature=0.2,
        max_tokens=64,
    )
    assert request.to_payload() == {
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "propose"},
        ],
        "temperature": 0.2,
        "max_tokens": 64,
    }


def test_max_tokens_is_omitted_when_unset() -> None:
    payload = ChatRequest("m", (ChatMessage("user", "x"),)).to_payload()
    assert "max_tokens" not in payload
    assert payload["temperature"] == 0.0


def test_requests_need_messages_and_sane_numbers() -> None:
    with pytest.raises(ValidationError, match="at least one message"):
        ChatRequest("m", ())
    with pytest.raises(ValidationError, match="temperature"):
        ChatRequest("m", (ChatMessage("user", "x"),), temperature=-1.0)
    with pytest.raises(ValidationError, match="max_tokens"):
        ChatRequest("m", (ChatMessage("user", "x"),), max_tokens=0)
