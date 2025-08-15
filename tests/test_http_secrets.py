"""Credential hygiene: keys are used for auth but never stored or reported."""

from __future__ import annotations

import json

import pytest

from hypoarena.errors import TransportError
from hypoarena.http_agent import HttpAgent, HttpConfig
from hypoarena.serialize import http_config_from_dict, http_config_to_dict
from loopback import MockChatServer, chat_body

SECRET = "sk-super-secret-value"


def test_the_key_is_sent_as_an_authorization_header() -> None:
    with MockChatServer([(200, chat_body())]) as server:
        agent = HttpAgent(
            HttpConfig(base_url=server.base_url, api_key=SECRET), name="keyed"
        )
        agent.propose("propose")
        headers = server.received[0]["headers"]
    assert headers["authorization"] == f"Bearer {SECRET}"


def test_the_key_never_appears_in_error_details() -> None:
    with MockChatServer([(500, {"error": "boom"})]) as server:
        agent = HttpAgent(
            HttpConfig(base_url=server.base_url, api_key=SECRET, max_retries=0)
        )
        with pytest.raises(TransportError) as info:
            agent.propose("propose")
    rendered = str(info.value) + json.dumps(info.value.details, default=str)
    assert SECRET not in rendered


def test_serialized_configs_mask_the_key() -> None:
    encoded = http_config_to_dict(HttpConfig(api_key=SECRET))
    assert encoded["api_key"] == "***"
    assert SECRET not in json.dumps(encoded)


def test_decoding_never_restores_a_credential() -> None:
    payload = http_config_to_dict(HttpConfig(api_key=SECRET, max_retries=4))
    restored = http_config_from_dict(payload)
    assert restored.api_key is None
    assert restored.max_retries == 4
    assert restored.retry_statuses == HttpConfig().retry_statuses


def test_decoded_configs_reject_tampering() -> None:
    payload = http_config_to_dict(HttpConfig())
    payload["proxy"] = "http://collector.example"
    with pytest.raises(Exception) as info:  # noqa: B017, PT011 - type checked below
        http_config_from_dict(payload)
    assert info.value.__class__.__name__ == "SchemaError"


def test_config_fingerprint_is_stable_without_the_key() -> None:
    keyed = HttpConfig(api_key=SECRET).fingerprint()
    masked = HttpConfig(api_key="another-secret").fingerprint()
    assert keyed == masked
    assert SECRET not in keyed


def test_an_artifact_dump_never_contains_the_key() -> None:
    payload = http_config_to_dict(HttpConfig(api_key=SECRET))
    artifact = json.dumps({"http": payload, "notes": "run metadata"})
    assert SECRET not in artifact
    assert "***" in artifact
