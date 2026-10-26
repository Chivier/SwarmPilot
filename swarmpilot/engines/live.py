"""Synchronous helpers that talk to a live engine instance.

The Planner's deployment path is synchronous, while adapter hooks are
async. These helpers run an adapter hook on a private event loop in a
worker thread, so they work whether or not the caller already runs a loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from loguru import logger

from swarmpilot.engines.base import EngineAdapter

T = TypeVar("T")


def _run(fn: Callable[[httpx.AsyncClient], Awaitable[T]]) -> T:
    async def main() -> T:
        async with httpx.AsyncClient() as client:
            return await fn(client)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, main()).result()


def _base_url(endpoint: str) -> str:
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"http://{endpoint}"


def probe_platform_info(
    adapter: EngineAdapter,
    endpoint: str,
    hardware_name: str = "gpu",
    ready_timeout_s: float = 0.0,
    poll_interval_s: float = 2.0,
) -> dict[str, str]:
    """Wait for an instance to be healthy, then build its platform info.

    The engine version comes from the running instance, so predictor
    models are keyed on what actually runs.

    Args:
        adapter: The instance's engine adapter.
        endpoint: Instance address (``host:port`` or a URL).
        hardware_name: Hardware label for the platform info.
        ready_timeout_s: How long to wait for the health path. Zero checks
            once without waiting.
        poll_interval_s: Delay between health checks.

    Returns:
        Platform info dict. The version is ``"unknown"`` if the instance
        never became healthy or did not report a version.
    """
    base = _base_url(endpoint)

    async def probe(client: httpx.AsyncClient) -> str:
        deadline = time.monotonic() + ready_timeout_s
        while not await adapter.health(client, base):
            if time.monotonic() >= deadline:
                logger.warning(
                    f"{adapter.name} at {base} not healthy after "
                    f"{ready_timeout_s:.0f}s; registering with unknown version"
                )
                return "unknown"
            await asyncio.sleep(poll_interval_s)
        return await adapter.version(client, base)

    version = _run(probe)
    return adapter.platform_info(version, hardware_name)
