"""Helpers the built-in styles share."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from swarmpilot.api.base import TranslationError


def new_id(prefix: str) -> str:
    """Return a fresh id such as ``msg_1f3a...``."""
    return f"{prefix}{uuid.uuid4().hex[:24]}"


def now() -> int:
    """Return the current Unix time in whole seconds."""
    return int(time.time())


def as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return ``value`` if it is an object, else raise a 400."""
    if not isinstance(value, Mapping):
        raise TranslationError(f"'{name}' must be an object", param=name)
    return value


def as_list(value: Any, name: str) -> list[Any]:
    """Return ``value`` if it is a list, else raise a 400."""
    if not isinstance(value, list):
        raise TranslationError(f"'{name}' must be an array", param=name)
    return value


def as_str(value: Any, name: str) -> str:
    """Return ``value`` if it is a string, else raise a 400."""
    if not isinstance(value, str):
        raise TranslationError(f"'{name}' must be a string", param=name)
    return value


def opt_int(body: Mapping[str, Any], key: str) -> int | None:
    """Return an optional integer field, refusing other types."""
    v = body.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise TranslationError(f"'{key}' must be an integer", param=key)
    return v


def opt_float(body: Mapping[str, Any], key: str) -> float | None:
    """Return an optional finite number field, refusing other types."""
    v = body.get(key)
    if v is None:
        return None
    if (
        isinstance(v, bool)
        or not isinstance(v, int | float)
        or not math.isfinite(v)
    ):
        raise TranslationError(f"'{key}' must be a number", param=key)
    return float(v)


def stop_list(value: Any, name: str) -> list[str]:
    """Return stop sequences given as a string or a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(s, str) for s in value):
        return list(value)
    raise TranslationError(
        f"'{name}' must be a string or an array of strings", param=name
    )


def parse_arguments(arguments: str) -> dict[str, Any]:
    """Parse tool-call arguments for styles that carry them as an object.

    Models sometimes emit invalid JSON; the raw text is then kept under
    ``"_raw"`` rather than lost.
    """
    try:
        value = json.loads(arguments) if arguments else {}
    except ValueError:
        return {"_raw": arguments}
    return value if isinstance(value, dict) else {"_raw": arguments}


def dump_arguments(value: Any) -> str:
    """Return tool-call arguments as compact JSON text."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def split_data_url(url: str) -> tuple[str, str] | None:
    """Return ``(media_type, base64 data)`` of a base64 ``data:`` URL."""
    if not url.startswith("data:"):
        return None
    head, sep, data = url[5:].partition(",")
    if not sep or not head.endswith(";base64"):
        return None
    return head[: -len(";base64")] or "application/octet-stream", data


def text_of(content: Any) -> str:
    """Join the text of a string or a list of ``{"text": ...}`` parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, Mapping) and isinstance(p.get("text"), str)
        )
    return ""


async def collect(agen: Any) -> list[Any]:
    """Drain an async iterator into a list (tests and small helpers)."""
    return [x async for x in agen]
