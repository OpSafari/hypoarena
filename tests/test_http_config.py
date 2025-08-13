"""Adapter configuration: validation, endpoint building and redaction."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.http_agent import DEFAULT_RETRY_STATUSES, HttpConfig


def test_defaults_target_a_local_mock() -> None:
    config = HttpConfig()
    assert config.base_url == "http://127.0.0.1:8000/v1"
    assert config.model == "mock-model"
    assert config.api_key is None
    assert config.timeout == 5.0
    assert config.max_retries == 2
    assert config.retry_statuses == DEFAULT_RETRY_STATUSES


def test_endpoint_joins_base_and_path() -> None:
    assert HttpConfig(base_url="http://127.0.0.1:9/v1").endpoint() == (
        "http://127.0.0.1:9/v1/chat/completions"
    )
    assert HttpConfig(base_url="http://127.0.0.1:9/v1/").endpoint() == (
        "http://127.0.0.1:9/v1/chat/completions"
    )


def test_invalid_urls_and_numbers_are_rejected() -> None:
    with pytest.raises(ValidationError, match="http or https"):
        HttpConfig(base_url="file:///tmp/socket")
    with pytest.raises(ValidationError, match="must name a host"):
        HttpConfig(base_url="http://")
    with pytest.raises(ValidationError, match="model"):
        HttpConfig(model="  ")
    with pytest.raises(ValidationError, match="timeout"):
        HttpConfig(timeout=0)
    with pytest.raises(ValidationError, match="max_retries"):
        HttpConfig(max_retries=-1)
    with pytest.raises(ValidationError, match="retry_backoff"):
        HttpConfig(retry_backoff=-0.5)
    with pytest.raises(ValidationError, match="status codes"):
        HttpConfig(retry_statuses=(99,))


def test_headers_carry_a_key_only_when_one_is_configured() -> None:
    assert "Authorization" not in HttpConfig().headers()
    keyed = HttpConfig(api_key="sk-test")
    assert keyed.headers()["Authorization"] == "Bearer sk-test"
    assert keyed.headers()["Content-Type"] == "application/json"


def test_redaction_never_exposes_the_key() -> None:
    config = HttpConfig(api_key="sk-secret-value")
    redacted = config.redacted()
    assert redacted["api_key"] == "***"
    assert "sk-secret-value" not in str(redacted)
    assert HttpConfig().redacted()["api_key"] is None


def test_redacted_view_keeps_the_retry_settings() -> None:
    redacted = HttpConfig(max_retries=3, retry_backoff=0.5).redacted()
    assert redacted["max_retries"] == 3
    assert redacted["retry_backoff"] == 0.5
    assert redacted["retry_statuses"] == list(DEFAULT_RETRY_STATUSES)
