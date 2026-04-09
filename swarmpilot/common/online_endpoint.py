"""Utilities for online API endpoint integration.

Provides helper functions that map online API credentials to
SwarmPilot's PlatformInfo format and extract predictor features
from API request/response pairs.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any


def platform_info_from_online_endpoint(
    base_url: str,
    api_key: str,
) -> dict[str, str]:
    """Generate PlatformInfo dict for an online API endpoint.

    Maps online API credentials to PlatformInfo fields so each
    (provider, key) combination gets a unique predictor model:

    - software_name  = hash(base_url)   — identifies the provider
    - software_version = hash(api_key)  — identifies the key/tier
    - hardware_name  = "cloud"          — fixed; extract_gpu_specs()
      returns None for this value

    Args:
        base_url: API base URL, e.g. "https://api.anthropic.com".
        api_key: API key string.

    Returns:
        Dict with software_name, software_version, hardware_name.
    """
    return {
        "software_name": hashlib.sha256(base_url.encode()).hexdigest()[:16],
        "software_version": hashlib.sha256(api_key.encode()).hexdigest()[:16],
        "hardware_name": "cloud",
    }


def extract_online_features(
    request_body: dict[str, Any],
    response: dict[str, Any],
    is_streaming: bool = False,
) -> dict[str, Any]:
    """Extract predictor features from an online API request/response.

    Args:
        request_body: The request sent to the API.
        response: The API response (or usage info).
        is_streaming: Whether streaming was used.

    Returns:
        Feature dict suitable for predictor training/prediction.
    """
    messages = request_body.get("messages", [])
    # Rough token estimate from message text (fallback).
    input_text = " ".join(
        m.get("content", "")
        for m in messages
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    )
    input_tokens_estimate = len(input_text) // 4

    usage = response.get("usage", {})
    output_tokens = usage.get("completion_tokens", 0)
    tools = request_body.get("tools", [])

    return {
        "input_tokens": usage.get("prompt_tokens", input_tokens_estimate),
        "estimated_output_tokens": output_tokens,
        "model_tier": _infer_model_tier(request_body.get("model", "")),
        "is_streaming": int(is_streaming),
        "has_tools": int(len(tools) > 0),
        "time_of_day_hour": datetime.now().hour,
    }


def _infer_model_tier(model_name: str) -> int:
    """Map model name to tier: 0=small, 1=medium, 2=large.

    Args:
        model_name: Model identifier string.

    Returns:
        Integer tier (0, 1, or 2).
    """
    model_lower = model_name.lower()
    if any(k in model_lower for k in ("haiku", "-mini", "flash", "8b")):
        return 0
    if any(k in model_lower for k in ("sonnet", "4o", "pro", "70b")):
        return 1
    if any(k in model_lower for k in ("opus", "o1", "ultra", "405b")):
        return 2
    return 1  # default to medium
