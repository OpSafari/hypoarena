"""A local loopback mock of the chat-completions API.

The server binds an ephemeral port on ``127.0.0.1`` and serves scripted replies,
so adapter tests exercise real socket behaviour (headers, status codes, timeouts)
without touching any external service.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def chat_body(
    text: str = "ok",
    *,
    model: str = "mock-model",
    prompt_tokens: int = 5,
    completion_tokens: int = 2,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Return a well-formed chat-completions response body."""
    return {
        "id": "chatcmpl-mock",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class MockChatServer:
    """Context manager serving queued ``(status, body)`` replies over loopback."""

    def __init__(
        self, replies: Sequence[tuple[int, Any]] = (), *, delay: float = 0.0
    ) -> None:
        self.replies: list[tuple[int, Any]] = list(replies)
        self.delay = delay
        self.received: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - http.server naming
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    payload = {"unparsed": raw.decode("utf-8", "replace")}
                server.received.append(
                    {
                        "path": self.path,
                        "payload": payload,
                        "headers": {
                            key.lower(): value for key, value in self.headers.items()
                        },
                    }
                )
                if server.delay:
                    time.sleep(server.delay)
                if server.replies:
                    status, body = server.replies.pop(0)
                else:
                    status, body = 200, chat_body()
                encoded = (
                    body
                    if isinstance(body, bytes)
                    else json.dumps(body).encode("utf-8")
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args: Any) -> None:
                """Keep the test output clean."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> MockChatServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        """Return the versioned base URL of the running mock."""
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def port(self) -> int:
        """Return the ephemeral port the mock bound."""
        return int(self._server.server_address[1])
