"""OpenAI-compatible HTTP adapter.

The client speaks the widely used chat-completions payload shape so it can point
at any compatible service, but nothing in this repository calls a real endpoint:
tests bind a mock server on ``127.0.0.1`` and the default configuration refuses
to send credentials anywhere. API keys are never serialized, logged or included
in error details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlparse

from hypoarena.errors import (
    TransportError,
    ValidationError,
)

DEFAULT_RETRY_STATUSES: tuple[int, ...] = (408, 429, 500, 502, 503, 504)
DEFAULT_TIMEOUT = 5.0
REDACTED = "***"
CHAT_COMPLETIONS_PATH = "/chat/completions"


@dataclass(frozen=True)
class HttpConfig:
    """Connection and retry settings for one adapter.

    ``retry_backoff`` is the base delay multiplied by the attempt number; tests
    set it to zero so retry behaviour is exercised without sleeping.
    """

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "mock-model"
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = 2
    retry_backoff: float = 0.0
    retry_statuses: tuple[int, ...] = DEFAULT_RETRY_STATUSES

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValidationError(
                "base_url must use http or https", base_url=self.base_url
            )
        if not parsed.netloc:
            raise ValidationError("base_url must name a host", base_url=self.base_url)
        if not self.model.strip():
            raise ValidationError("model must not be blank")
        if self.timeout <= 0:
            raise ValidationError("timeout must be > 0", timeout=self.timeout)
        if self.max_retries < 0:
            raise ValidationError(
                "max_retries must be >= 0", max_retries=self.max_retries
            )
        if self.retry_backoff < 0:
            raise ValidationError(
                "retry_backoff must be >= 0", retry_backoff=self.retry_backoff
            )
        for status in self.retry_statuses:
            if not 100 <= status <= 599:
                raise ValidationError(
                    "retry_statuses must be HTTP status codes", status=status
                )

    def endpoint(self, path: str = CHAT_COMPLETIONS_PATH) -> str:
        """Return the absolute URL for one API path."""
        return f"{self.base_url.rstrip('/')}{path}"

    def headers(self) -> dict[str, str]:
        """Return request headers, adding authorization only when a key is set."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def redacted(self) -> dict[str, object]:
        """Return a safe view for logs and artifacts: the key is never included."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key": REDACTED if self.api_key else None,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_backoff": self.retry_backoff,
            "retry_statuses": list(self.retry_statuses),
        }


MESSAGE_ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class ChatMessage:
    """One chat message in the request payload."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in MESSAGE_ROLES:
            raise ValidationError(
                "unknown message role", role=self.role, allowed=list(MESSAGE_ROLES)
            )
        if not self.content.strip():
            raise ValidationError("message content must not be blank")

    def to_dict(self) -> dict[str, str]:
        """Return the wire representation of this message."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatRequest:
    """A chat-completions request body."""

    model: str
    messages: tuple[ChatMessage, ...]
    temperature: float = 0.0
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValidationError("a chat request needs at least one message")
        if self.temperature < 0:
            raise ValidationError(
                "temperature must be >= 0", temperature=self.temperature
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValidationError(
                "max_tokens must be >= 1 when set", max_tokens=self.max_tokens
            )

    def to_payload(self) -> dict[str, object]:
        """Return the JSON body to POST.

        ``max_tokens`` is omitted when unset so services that reject a null value
        still accept the request.
        """
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [message.to_dict() for message in self.messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload


@dataclass(frozen=True)
class ChatResponse:
    """The parts of a chat-completions response this toolkit relies on."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    response_id: str

    @property
    def total_tokens(self) -> int:
        """Sum of the reported prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_payload(cls, body: str) -> ChatResponse:
        """Parse a JSON response body, rejecting malformed shapes.

        Parsing is strict on purpose: a truncated or reshaped upstream response
        should fail the call, not silently produce an empty hypothesis.
        """
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            raise TransportError(
                "upstream returned invalid JSON", reason=str(error)
            ) from None
        if not isinstance(payload, dict):
            raise TransportError("upstream response is not a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TransportError("upstream response has no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise TransportError("upstream choice is not a JSON object")
        message = first.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TransportError("upstream choice has no message content")
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            raise TransportError("upstream usage is not a JSON object")
        return cls(
            text=message["content"],
            model=str(payload.get("model", "")),
            prompt_tokens=_as_count(usage.get("prompt_tokens")),
            completion_tokens=_as_count(usage.get("completion_tokens")),
            finish_reason=str(first.get("finish_reason") or "stop"),
            response_id=str(payload.get("id", "")),
        )


def _as_count(value: object) -> int:
    """Coerce a usage counter, treating missing or invalid values as zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))
