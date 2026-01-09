"""Chat response parsing: happy path, defaults and malformed bodies."""

from __future__ import annotations

import json

import pytest

from hypoarena.errors import TransportError
from hypoarena.http_agent import ChatResponse


def payload(**overrides: object) -> str:
    body: dict[str, object] = {
        "id": "chatcmpl-1",
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "a hypothesis"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }
    body.update(overrides)
    return json.dumps(body)


def test_a_well_formed_response_is_parsed() -> None:
    response = ChatResponse.from_payload(payload())
    assert response.text == "a hypothesis"
    assert response.model == "mock-model"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 4
    assert response.total_tokens == 16
    assert response.finish_reason == "stop"
    assert response.response_id == "chatcmpl-1"


def test_missing_usage_and_ids_degrade_to_defaults() -> None:
    body = json.loads(payload())
    del body["usage"]
    del body["id"]
    del body["model"]
    response = ChatResponse.from_payload(json.dumps(body))
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0
    assert response.response_id == ""
    assert response.model == ""


def test_invalid_json_is_a_transport_error() -> None:
    with pytest.raises(TransportError, match="invalid JSON"):
        ChatResponse.from_payload("{not json")


@pytest.mark.parametrize(
    "body",
    [
        "[]",
        '{"choices": []}',
        '{"choices": [1]}',
        '{"choices": [{}]}',
        '{"choices": [{"message": {}}]}',
        '{"choices": [{"message": {"content": 3}}]}',
    ],
)
def test_malformed_shapes_are_rejected(body: str) -> None:
    with pytest.raises(TransportError):
        ChatResponse.from_payload(body)


def test_non_numeric_usage_counters_are_ignored() -> None:
    body = json.loads(payload())
    body["usage"] = {"prompt_tokens": "many", "completion_tokens": None}
    response = ChatResponse.from_payload(json.dumps(body))
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0


def test_a_missing_finish_reason_defaults_to_stop() -> None:
    body = json.loads(payload())
    del body["choices"][0]["finish_reason"]
    assert ChatResponse.from_payload(json.dumps(body)).finish_reason == "stop"
