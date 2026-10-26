"""Automatic reporter for sending uncompleted task counts to planner.

This module provides a background task that periodically reports the total
number of uncompleted (pending + running) tasks to the planner service's
/submit_target endpoint.

With an instance registry and an arrival counter it also reports what the
Planner's tail scaler reads (item 3.4): the arrival rate, a q0.9 estimate
of the outstanding work from the queue sketches, and the calls other
models' tasks made.
"""

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Optional

import httpx
import numpy as np
from loguru import logger

from swarmpilot.scheduler.utils.http_error_logger import log_http_error

if TYPE_CHECKING:
    from swarmpilot.scheduler.registry.instance_registry import (
        InstanceRegistry,
    )
    from swarmpilot.scheduler.registry.task_registry import TaskRegistry
    from swarmpilot.scheduler.utils.arrival_counter import ArrivalCounter
    from swarmpilot.scheduler.utils.throughput_tracker import ThroughputTracker

from swarmpilot.scheduler.models import (
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
    TaskStatus,
)

# Quantile of the outstanding work reported as ``work_s``.
WORK_QUANTILE = 0.9


class PlannerReporter:
    """Background task that periodically reports uncompleted tasks to planner."""

    def __init__(
        self,
        task_registry: "TaskRegistry",
        planner_url: str,
        interval: float,
        timeout: float = 5.0,
        throughput_tracker: Optional["ThroughputTracker"] = None,
        instance_registry: Optional["InstanceRegistry"] = None,
        arrivals: Optional["ArrivalCounter"] = None,
    ):
        """Initialize the planner reporter.

        Args:
            task_registry: TaskRegistry instance for getting task counts
            planner_url: Base URL of the planner service
            interval: Reporting interval in seconds
            timeout: HTTP request timeout in seconds
            throughput_tracker: Optional throughput tracker for reporting execution times
            instance_registry: Source of the queue sketches for ``work_s``.
            arrivals: Counter of task arrivals and their calling models.
        """
        self._task_registry = task_registry
        self._planner_url = planner_url
        self._interval = interval
        self._timeout = timeout
        self._throughput_tracker = throughput_tracker
        self._instance_registry = instance_registry
        self._arrivals = arrivals

        self._model_id: str | None = None
        self._reporter_task: asyncio.Task | None = None
        self._shutdown = False
        self._http_client = httpx.AsyncClient(
            timeout=timeout,
            verify=False,  # Disable SSL verification for internal network
        )

    def set_model_id(self, model_id: str) -> None:
        """Set the model ID for reporting.

        This should be called when the first instance registers to the scheduler.

        Args:
            model_id: Model ID to use for reporting
        """
        if self._model_id is None:
            self._model_id = model_id
            logger.info(f"Planner reporter model_id set to: {model_id}")
        else:
            logger.debug(
                f"Model ID already set to {self._model_id}, ignoring {model_id}"
            )

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
            with suppress(asyncio.CancelledError):
                await self._reporter_task

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
                    logger.debug(
                        "Skipping report: model_id not set yet (no instances registered)"
                    )
                    continue

                await self._report_to_planner()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.opt(exception=True).error(
                    f"[planner_reporter] Error in planner report loop: {e}"
                )
                # Continue running despite errors

        logger.debug("Planner report loop stopped")

    async def _report_to_planner(self) -> None:
        """Report current uncompleted task count to planner."""
        try:
            # Get uncompleted task count (pending + running)
            pending = await self._task_registry.get_count_by_status(
                TaskStatus.PENDING
            )
            running = await self._task_registry.get_count_by_status(
                TaskStatus.RUNNING
            )
            total_uncompleted = pending + running
            payload = await self._target_payload(float(total_uncompleted))

            # POST to planner's /submit_target endpoint
            try:
                response = await self._http_client.post(
                    f"{self._planner_url}/v1/submit_target",
                    json=payload,
                )
            except BaseException as e:
                raise RuntimeError(
                    f"Planner submit_target call failed: {e}"
                ) from e
            response.raise_for_status()

            logger.debug(
                f"[PlannerReporter] Reported to planner: model_id={self._model_id}, "
                f"uncompleted={total_uncompleted} (pending={pending}, running={running})"
            )

            # Report throughput data if tracker is available
            await self._report_throughput()

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
                request_url=f"{self._planner_url}/v1/submit_target",
                request_method="POST",
                request_body={
                    "model_id": self._model_id,
                    "value": float(total_uncompleted),
                },
                context="planner report connection error",
            )
            logger.warning(f"Planner report HTTP error: {e}")
        except (RuntimeError, TimeoutError) as e:
            logger.opt(exception=True).error(
                f"[planner_reporter] Planner report error: {e}"
            )

    async def _target_payload(self, uncompleted: float) -> dict[str, Any]:
        """The /v1/submit_target body, with tail-scaler fields if known."""
        payload: dict[str, Any] = {
            "model_id": self._model_id,
            "value": uncompleted,
        }
        if self._arrivals is not None:
            window = self._arrivals.take()
            if window.seconds > 0:
                payload["arrival_rate"] = window.rate
                payload["calls_from"] = window.call_rates()
        if self._instance_registry is not None:
            payload["work_s"] = await self._tail_work_s()
        return payload

    async def _tail_work_s(self) -> float:
        """Upper estimate of the model's outstanding work, in seconds.

        Sums each instance's q0.9 queue wait times its capacity (a sketch
        of an instance serving ``n`` tasks at once holds work divided by
        ``n``). The sum of quantiles bounds the quantile of the sum.
        """
        registry = self._instance_registry
        instances = await registry.list_all(model_id=self._model_id)
        queues = await registry.get_all_queue_info(
            [i.instance_id for i in instances]
        )
        total_ms = 0.0
        for iid, q in queues.items():
            if isinstance(q, InstanceQueueProbabilistic) and q.values:
                wait = float(np.interp(WORK_QUANTILE, q.quantiles, q.values))
            elif isinstance(q, InstanceQueueExpectError):
                wait = q.expected_time_ms + q.error_margin_ms
            else:
                continue
            total_ms += max(wait, 0.0) * registry.get_capacity(iid)
        return total_ms / 1000.0

    async def _report_throughput(self) -> None:
        """Report throughput data for instances with recent data to planner.

        Only reports throughput for instances that have completed tasks since the
        last reporting interval. This prevents stale data from being reported.
        """
        if self._throughput_tracker is None:
            return

        try:
            # Only get averages for instances with new data since last report
            try:
                averages = await self._throughput_tracker.get_averages_for_recent_instances_seconds()
            except BaseException as e:
                raise RuntimeError(
                    f"Failed to fetch recent throughput averages: {e}"
                ) from e

            if not averages:
                logger.debug(
                    "[PlannerReporter] No instances with new throughput data to report"
                )
                return

            for (
                instance_endpoint,
                avg_execution_time_seconds,
            ) in averages.items():
                try:
                    await self._http_client.post(
                        f"{self._planner_url}/v1/submit_throughput",
                        json={
                            "instance_url": instance_endpoint,
                            "avg_execution_time": avg_execution_time_seconds,
                            "model_id": self._model_id,
                        },
                    )
                    logger.debug(
                        f"Reported throughput to planner: instance={instance_endpoint}, "
                        f"avg_execution_time={avg_execution_time_seconds:.3f}s"
                    )
                except httpx.HTTPError as e:
                    logger.warning(
                        f"Failed to report throughput for {instance_endpoint}: {e}"
                    )

            logger.debug(
                f"[PlannerReporter] Reported throughput for {len(averages)} instance(s) with recent data"
            )
        except (RuntimeError, TimeoutError) as e:
            logger.opt(exception=True).error(
                f"[planner_reporter] Throughput report error: {e}"
            )
