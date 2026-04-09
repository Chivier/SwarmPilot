"""Configuration loader for online API endpoints.

Parses a YAML file that defines online API endpoints (e.g. Claude,
OpenAI, Gemini) to be registered as Scheduler instances at startup.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field


class OnlineEndpointEntry(BaseModel):
    """A single online API endpoint configuration.

    Attributes:
        name: Human-readable name, used to derive instance_id.
        base_url: API base URL (e.g. "https://api.anthropic.com").
        api_key_env: Environment variable holding the API key.
        model_id: Model identifier for scheduling.
        concurrency_limit: Max concurrent requests (for Planner
            capacity matrix; not enforced by Predictor).
    """

    name: str
    base_url: str
    api_key_env: str
    model_id: str
    concurrency_limit: int = Field(default=10)


class OnlineEndpointsConfig(BaseModel):
    """Top-level config wrapping a list of endpoints.

    Attributes:
        endpoints: List of online endpoint entries.
    """

    endpoints: list[OnlineEndpointEntry] = Field(default_factory=list)


def load_online_endpoints_config(
    path: str | None = None,
) -> list[OnlineEndpointEntry]:
    """Load online endpoint configuration from a YAML file.

    Resolution order for the config path:
    1. Explicit ``path`` argument.
    2. ``ONLINE_ENDPOINTS_CONFIG`` environment variable.

    Returns an empty list (no-op) when:
    - No path is resolved.
    - The resolved path does not point to an existing file.

    Args:
        path: Optional explicit path to the YAML config file.

    Returns:
        List of OnlineEndpointEntry objects.
    """
    resolved = path or os.getenv("ONLINE_ENDPOINTS_CONFIG", "")
    if not resolved:
        return []

    config_path = Path(resolved)
    if not config_path.is_file():
        logger.debug(f"Online endpoints config not found: {resolved}")
        return []

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        logger.warning(f"Invalid online endpoints config: {resolved}")
        return []

    parsed = OnlineEndpointsConfig.model_validate(raw)
    logger.info(
        f"Loaded {len(parsed.endpoints)} online endpoint(s) " f"from {resolved}"
    )
    return parsed.endpoints
