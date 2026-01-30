"""Planner registrar service for multi-scheduler architecture (PYLET-024).

This module registers the scheduler with the planner on startup and
deregisters on shutdown. If registration fails after retries, the
scheduler exits (fail-hard behavior).

Example:
    from swarmpilot.scheduler.services.planner_registrar import PlannerRegistrar
    from swarmpilot.scheduler.config import config

    registrar = PlannerRegistrar(config.planner_registration)
    await registrar.start()  # Raises RuntimeError on failure
    ...
    await registrar.stop()   # Best-effort deregistration
"""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from ..config import PlannerRegistrationConfig


class PlannerRegistrar:
    """Registers the scheduler with the planner service.

    On startup, sends a POST to /scheduler/register on the planner.
    On shutdown, sends a POST to /scheduler/deregister.

    Attributes:
        config: Planner registration configuration.
    """

    def __init__(self, config: PlannerRegistrationConfig) -> None:
        """Initialize the planner registrar.

        Args:
            config: Registration configuration.
        """
        self._config = config
        self._registered = False

    async def start(self) -> None:
        """Register with the planner.

        Retries up to max_retries times with retry_delay between attempts.

        Raises:
            RuntimeError: If registration fails after all retries.
        """
        if not self._config.enabled:
            logger.info(
                "Planner registration disabled "
                "(PLANNER_REGISTRATION_URL, SCHEDULER_MODEL_ID, or "
                "SCHEDULER_SELF_URL not set)"
            )
            return

        logger.info(
            f"Registering with planner: model_id={self._config.model_id} "
            f"self_url={self._config.self_url} "
            f"planner_url={self._config.planner_url}"
        )

        last_error: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._config.timeout) as client:
                    response = await client.post(
                        f"{self._config.planner_url.rstrip('/')}"
                        f"/v1/scheduler/register",
                        json={
                            "model_id": self._config.model_id,
                            "scheduler_url": self._config.self_url,
                        },
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            replaced = data.get("replaced_previous", False)
                            logger.info(
                                f"Registered with planner successfully "
                                f"(attempt {attempt}, "
                                f"replaced_previous={replaced})"
                            )
                            self._registered = True
                            return

                    error_msg = (
                        f"Registration failed: HTTP {response.status_code} "
                        f"- {response.text}"
                    )
                    logger.warning(
                        f"Attempt {attempt}/{self._config.max_retries}: " f"{error_msg}"
                    )
                    last_error = RuntimeError(error_msg)

            except httpx.RequestError as e:
                logger.warning(
                    f"Attempt {attempt}/{self._config.max_retries}: "
                    f"Connection error: {e}"
                )
                last_error = e

            if attempt < self._config.max_retries:
                logger.info(f"Retrying in {self._config.retry_delay}s...")
                await asyncio.sleep(self._config.retry_delay)

        raise RuntimeError(
            f"Failed to register with planner after "
            f"{self._config.max_retries} attempts: {last_error}"
        )

    async def stop(self) -> None:
        """Deregister from the planner (best-effort).

        This is called on graceful shutdown. Errors are logged but
        not raised since the scheduler is shutting down anyway.
        """
        if not self._registered:
            return

        logger.info(f"Deregistering from planner: model_id={self._config.model_id}")

        try:
            async with httpx.AsyncClient(timeout=self._config.timeout) as client:
                response = await client.post(
                    f"{self._config.planner_url.rstrip('/')}" f"/v1/scheduler/deregister",
                    json={"model_id": self._config.model_id},
                )

                if response.status_code == 200:
                    logger.info("Deregistered from planner successfully")
                else:
                    logger.warning(
                        f"Deregistration returned: {response.status_code} "
                        f"- {response.text}"
                    )

        except Exception as e:
            logger.warning(f"Deregistration failed (best-effort): {e}")

        self._registered = False

    @property
    def is_registered(self) -> bool:
        """Check if currently registered with planner."""
        return self._registered
