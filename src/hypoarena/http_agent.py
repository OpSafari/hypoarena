"""OpenAI-compatible HTTP adapter.

The client speaks the widely used chat-completions payload shape so it can point
at any compatible service, but nothing in this repository calls a real endpoint:
tests bind a mock server on ``127.0.0.1`` and the default configuration refuses
to send credentials anywhere. API keys are never serialized, logged or included
in error details.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from time import sleep
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from hypoarena.agents import (
    AgentRequest,
    AgentResponse,
    BaseAgent,
)
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


class Transport(Protocol):
    """One POST call; returns the HTTP status and the raw response body."""

    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, str]: ...


@dataclass
class UrllibTransport:
    """Default transport built on the standard library.

    HTTP error statuses are returned to the caller (so the retry policy can see
    them); only connection-level failures raise, because those have no status to
    report.
    """

    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, str]:
        """POST ``payload`` as JSON and return ``(status, body)``."""
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        request = Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                return int(response.status), response.read().decode("utf-8")
        except HTTPError as error:
            return error.code, error.read().decode("utf-8", errors="replace")
        except (URLError, TimeoutError, OSError) as error:
            raise TransportError(f"cannot reach {url}", reason=str(error)) from error


class HttpAgent(BaseAgent):
    """Adapter that turns agent requests into chat-completions calls."""

    def __init__(
        self,
        config: HttpConfig | None = None,
        *,
        name: str = "http",
        transport: Transport | None = None,
        system_prompt: str = "",
    ) -> None:
        super().__init__(name)
        self.config = config or HttpConfig()
        self.transport: Transport = transport or UrllibTransport()
        self.system_prompt = system_prompt

    def messages_for(self, request: AgentRequest) -> tuple[ChatMessage, ...]:
        """Flatten a request into chat messages: prompt first, context after."""
        content = "\n".join((request.prompt, *request.context)).strip()
        messages = []
        if self.system_prompt.strip():
            messages.append(ChatMessage("system", self.system_prompt))
        messages.append(ChatMessage("user", content))
        return tuple(messages)

    def respond(self, request: AgentRequest) -> AgentResponse:
        """POST one request, applying the retry policy from the config."""
        payload = ChatRequest(
            model=self.config.model,
            messages=self.messages_for(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ).to_payload()
        url = self.config.endpoint()
        headers = self.config.headers()
        attempts = 0
        while True:
            attempts += 1
            try:
                status, body = self.transport.post(
                    url, payload, headers, self.config.timeout
                )
            except TransportError:
                if attempts > self.config.max_retries:
                    raise TransportError(
                        "upstream unreachable after retries",
                        attempts=attempts,
                        url=url,
                    ) from None
                self.wait_before_retry(attempts)
                continue
            if (
                status in self.config.retry_statuses
                and attempts <= self.config.max_retries
            ):
                self.wait_before_retry(attempts)
                continue
            if not 200 <= status < 300:
                raise TransportError(
                    f"upstream returned HTTP {status}", status=status, attempts=attempts
                )
            chat = ChatResponse.from_payload(body)
            return AgentResponse(
                request_id=request.request_id,
                text=chat.text,
                agent=self.name,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
                model=chat.model or self.config.model,
                finish_reason=chat.finish_reason,
            )

    def wait_before_retry(self, attempt: int) -> None:
        """Sleep the configured backoff; zero keeps retries instantaneous."""
        delay = self.config.retry_backoff * attempt
        if delay > 0:
            sleep(delay)
