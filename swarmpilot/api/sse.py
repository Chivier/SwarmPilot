"""Server-sent events: parse an engine's byte stream, frame a client's."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


async def iter_sse(
    chunks: AsyncIterator[bytes],
) -> AsyncIterator[tuple[str | None, str]]:
    """Yield ``(event, data)`` for each event in an SSE byte stream.

    Chunks may split events, lines and UTF-8 characters anywhere. Comment
    lines and events without data are skipped; several ``data:`` lines of
    one event are joined with newlines, as the SSE spec says.

    Args:
        chunks: Raw response body chunks.

    Yields:
        The event name (None when the event has none) and its data.
    """
    buf = b""
    event: str | None = None
    data: list[str] = []
    async for chunk in chunks:
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = buf[:nl].rstrip(b"\r").decode("utf-8", errors="replace")
            buf = buf[nl + 1 :]
            if not line:
                if data:
                    yield event, "\n".join(data)
                event, data = None, []
                continue
            if line.startswith(":"):
                continue
            key, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if key == "data":
                data.append(value)
            elif key == "event":
                event = value
    tail = buf.decode("utf-8", errors="replace").strip()
    if tail.startswith("data:"):
        data.append(tail[5:].lstrip())
    if data:
        yield event, "\n".join(data)


def frame(data: Any, event: str | None = None) -> bytes:
    """Return one SSE event as bytes, with ``data`` serialized as JSON."""
    payload = data if isinstance(data, str) else _dumps(data)
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode()


def _dumps(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
