"""OpenAI-compatible HTTP adapter.

The client speaks the widely used chat-completions payload shape so it can point
at any compatible service, but nothing in this repository calls a real endpoint:
tests bind a mock server on ``127.0.0.1`` and the default configuration refuses
to send credentials anywhere. API keys are never serialized, logged or included
in error details.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from hypoarena.errors import (
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
