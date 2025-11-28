"""
Automatic reporter for sending uncompleted task counts to planner.

This module provides a background task that periodically reports the total
number of uncompleted (pending + running) tasks to the planner service's
/submit_target endpoint.
"""

from typing import TYPE_CHECKING, Optional
import asyncio
import httpx
from loguru import logger

from .http_error_logger import log_http_error

if TYPE_CHECKING:
    from .task_registry import TaskRegistry

from .model import TaskStatus


class PlannerReporter:
    """Background task that periodically reports uncompleted tasks to planner."""

    def __init__(
        self,
        task_registry: "TaskRegistry",
        planner_url: str,
        interval: float,
        timeout: float = 5.0,
    ):
        """
        Initialize the planner reporter.

        Args:
            task_registry: TaskRegistry instance for getting task counts
            planner_url: Base URL of the planner service
            interval: Reporting interval in seconds
            timeout: HTTP request timeout in seconds
        """
        self._task_registry = task_registry
        self._planner_url = planner_url
        self._interval = interval
        self._timeout = timeout

        self._model_id: Optional[str] = None
        self._reporter_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._http_client = httpx.AsyncClient(
            timeout=timeout,
            verify=False,  # Disable SSL verification for internal network
        )

    def set_model_id(self, model_id: str) -> None:
        """
        Set the model ID for reporting.

        This should be called when the first instance registers to the scheduler.

        Args:
            model_id: Model ID to use for reporting
        """
        if self._model_id is None:
            self._model_id = model_id
            logger.info(f"Planner reporter model_id set to: {model_id}")
        else:
            logger.debug(f"Model ID already set to {self._model_id}, ignoring {model_id}")

    async def start(self) -> None:
        """Start the background reporter loop."""
        if self._reporter_task is not None:
            logger.warning("Planner reporter already running")
            return

        self._shutdown = False
        self._reporter_task = asyncio.create_task(self._report_loop())
        logger.info(
            f"Planner reporter started: URL={self._planner_url}, "
            f"interval={self._interval}s"
        )

    async def shutdown(self) -> None:
        """Shutdown the reporter gracefully."""
        logger.info("Shutting down planner reporter...")
        self._shutdown = True

        if self._reporter_task:
            self._reporter_task.cancel()
            try:
                await self._reporter_task
            except asyncio.CancelledError:
                pass

        await self._http_client.aclose()
        self._reporter_task = None
        logger.info("Planner reporter shutdown complete")

    async def _report_loop(self) -> None:
        """Background loop that reports to planner at configured interval."""
        logger.debug("Planner report loop started")

        while not self._shutdown:
            try:
                await asyncio.sleep(self._interval)

                if self._shutdown:
                    break

                # Only report if model_id is set
                if self._model_id is None:
                    logger.debug("Skipping report: model_id not set yet (no instances registered)")
                    continue

                await self._report_to_planner()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in planner report loop: {e}", exc_info=True)
                # Continue running despite errors

        logger.debug("Planner report loop stopped")

    async def _report_to_planner(self) -> None:
        """Report current uncompleted task count to planner."""
        try:
            # Get uncompleted task count (pending + running)
            pending = await self._task_registry.get_count_by_status(TaskStatus.PENDING)
            running = await self._task_registry.get_count_by_status(TaskStatus.RUNNING)
            total_uncompleted = pending + running

            # POST to planner's /submit_target endpoint
            response = await self._http_client.post(
                f"{self._planner_url}/submit_target",
                json={
                    "model_id": self._model_id,
                    "value": float(total_uncompleted),
                }
            )
            response.raise_for_status()

            logger.debug(
                f"Reported to planner: model_id={self._model_id}, "
                f"uncompleted={total_uncompleted} (pending={pending}, running={running})"
            )

        except httpx.HTTPStatusError as e:
            log_http_error(
                e,
                request_body={
                    "model_id": self._model_id,
                    "value": float(total_uncompleted),
                },
                context="planner report",
            )
            logger.warning(
                f"Planner report failed with status {e.response.status_code}: "
                f"{e.response.text}"
            )
        except httpx.HTTPError as e:
            log_http_error(
                e,
                request_url=f"{self._planner_url}/submit_target",
                request_method="POST",
                request_body={
                    "model_id": self._model_id,
                    "value": float(total_uncompleted),
                },
                context="planner report connection error",
            )
            logger.warning(f"Planner report HTTP error: {e}")
        except Exception as e:
            logger.error(f"Planner report error: {e}", exc_info=True)
