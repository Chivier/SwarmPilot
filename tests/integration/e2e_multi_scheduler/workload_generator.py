"""Multi-Scheduler Workload Generator for E2E Testing.

This module provides a workload generator that routes requests to the
correct scheduler based on model_id. Each model type has its own scheduler,
and tasks are submitted to the appropriate scheduler URL.

Features:
- Per-model QPS configuration
- Correct routing to model-specific schedulers
- Concurrent submissions with semaphore
- Detailed per-model statistics

Usage:
    from workload_generator import (
        MultiSchedulerWorkloadConfig,
        MultiSchedulerWorkloadGenerator,
    )

    config = MultiSchedulerWorkloadConfig(
        scheduler_urls={"model_a": "http://localhost:8010", ...},
        model_qps={"model_a": 3.0, "model_b": 2.0},
        duration_seconds=60.0,
    )

    generator = MultiSchedulerWorkloadGenerator(config)
    result = await generator.generate_workload()
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger


@dataclass
class MultiSchedulerWorkloadConfig:
    """Configuration for multi-scheduler workload generation.

    Attributes:
        scheduler_urls: Mapping of model_id to scheduler URL.
        model_qps: Mapping of model_id to QPS for that model.
        duration_seconds: Total test duration.
        sleep_time_range: Range for random sleep time (min, max).
        task_id_prefix: Prefix for task IDs.
        timeout_seconds: HTTP request timeout.
        max_concurrent_submissions: Max parallel submissions.
    """

    scheduler_urls: dict[str, str]
    model_qps: dict[str, float]
    duration_seconds: float = 60.0
    sleep_time_range: tuple[float, float] = (0.1, 1.0)
    task_id_prefix: str = "multi"
    timeout_seconds: float = 30.0
    max_concurrent_submissions: int = 50

    @property
    def model_ids(self) -> list[str]:
        """List of model IDs."""
        return list(self.scheduler_urls.keys())

    @property
    def total_qps(self) -> float:
        """Total QPS across all models."""
        return sum(self.model_qps.values())

    def get_model_task_count(self, model_id: str) -> int:
        """Calculate total tasks for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Number of tasks to submit for this model.
        """
        qps = self.model_qps.get(model_id, 0)
        return int(qps * self.duration_seconds)

    @property
    def total_tasks(self) -> int:
        """Total number of tasks to submit."""
        return sum(self.get_model_task_count(m) for m in self.model_ids)


@dataclass
class MultiSchedulerSubmissionResult:
    """Result of a single task submission.

    Attributes:
        task_id: Unique task identifier.
        model_id: Model this task was submitted for.
        scheduler_url: URL of the scheduler it was submitted to.
        sleep_time: Requested sleep time.
        submitted_at: Timestamp of submission.
        response_time_ms: Time to receive response.
        success: Whether submission succeeded.
        error: Error message if failed.
        scheduler_response: Response from scheduler.
    """

    task_id: str
    model_id: str
    scheduler_url: str
    sleep_time: float
    submitted_at: float
    response_time_ms: float
    success: bool
    error: str | None = None
    scheduler_response: dict[str, Any] | None = None


@dataclass
class MultiSchedulerWorkloadResult:
    """Aggregate result of workload generation.

    Attributes:
        total_tasks: Total tasks submitted.
        successful_submissions: Count of successful submissions.
        failed_submissions: Count of failed submissions.
        start_time: Workload start timestamp.
        end_time: Workload end timestamp.
        actual_duration: Actual duration in seconds.
        target_qps: Target total QPS.
        actual_qps: Achieved total QPS.
        submission_results: List of individual submission results.
        model_stats: Per-model statistics.
    """

    total_tasks: int
    successful_submissions: int
    failed_submissions: int
    start_time: float
    end_time: float
    actual_duration: float
    target_qps: float
    actual_qps: float
    submission_results: list[MultiSchedulerSubmissionResult]
    model_stats: dict[str, dict[str, Any]]

    def submission_success_rate(self) -> float:
        """Calculate submission success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_submissions / self.total_tasks

    def get_submission_latencies(self) -> list[float]:
        """Get list of successful submission latencies in ms."""
        return [r.response_time_ms for r in self.submission_results if r.success]

    def get_failed_submissions(self) -> list[MultiSchedulerSubmissionResult]:
        """Get list of failed submissions."""
        return [r for r in self.submission_results if not r.success]

    def get_model_latencies(self, model_id: str) -> list[float]:
        """Get submission latencies for a specific model.

        Args:
            model_id: Model identifier.

        Returns:
            List of latencies in ms.
        """
        return [
            r.response_time_ms
            for r in self.submission_results
            if r.success and r.model_id == model_id
        ]


class MultiSchedulerWorkloadGenerator:
    """Workload generator that routes to correct scheduler per model.

    This generator submits tasks to model-specific schedulers based on
    the scheduler_urls mapping. Each model can have a different QPS.
    """

    def __init__(self, config: MultiSchedulerWorkloadConfig):
        """Initialize workload generator.

        Args:
            config: Workload configuration.
        """
        self.config = config
        self._results: list[MultiSchedulerSubmissionResult] = []
        self._semaphore: asyncio.Semaphore | None = None
        self._task_counters: dict[str, int] = {}

    async def generate_workload(self) -> MultiSchedulerWorkloadResult:
        """Generate workload across multiple schedulers.

        Submits tasks at the configured QPS rate for each model,
        routing to the correct scheduler URL.

        Returns:
            MultiSchedulerWorkloadResult with aggregate metrics.
        """
        logger.info(
            f"Starting multi-scheduler workload generation: "
            f"total_qps={self.config.total_qps}, "
            f"duration={self.config.duration_seconds}s, "
            f"total_tasks={self.config.total_tasks}, "
            f"models={self.config.model_ids}"
        )

        for model_id, qps in self.config.model_qps.items():
            task_count = self.config.get_model_task_count(model_id)
            logger.info(f"  {model_id}: qps={qps:.2f}, tasks={task_count}")

        self._results = []
        self._task_counters = {m: 0 for m in self.config.model_ids}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_submissions)

        start_time = time.time()
        tasks = []

        # Build task schedule: list of (submit_time, model_id) tuples
        schedule = self._build_schedule()

        async with httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            limits=httpx.Limits(
                max_connections=self.config.max_concurrent_submissions * 2,
                max_keepalive_connections=self.config.max_concurrent_submissions,
            ),
        ) as client:
            for i, (target_time, model_id) in enumerate(schedule):
                # Wait until target time
                current_time = time.time()
                wait_time = (start_time + target_time) - current_time
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # Create submission task
                task = asyncio.create_task(self._submit_task(client, model_id, i))
                tasks.append(task)

                # Log progress every 10%
                if (i + 1) % max(1, len(schedule) // 10) == 0:
                    elapsed = time.time() - start_time
                    current_qps = (i + 1) / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {i + 1}/{len(schedule)} tasks submitted "
                        f"({(i + 1) / len(schedule) * 100:.1f}%), "
                        f"current_qps={current_qps:.2f}"
                    )

            # Wait for all submissions to complete
            logger.info("Waiting for all submissions to complete...")
            await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        actual_duration = end_time - start_time

        # Calculate per-model statistics
        model_stats = self._calculate_model_stats()

        result = MultiSchedulerWorkloadResult(
            total_tasks=len(schedule),
            successful_submissions=sum(1 for r in self._results if r.success),
            failed_submissions=sum(1 for r in self._results if not r.success),
            start_time=start_time,
            end_time=end_time,
            actual_duration=actual_duration,
            target_qps=self.config.total_qps,
            actual_qps=(
                len(self._results) / actual_duration if actual_duration > 0 else 0
            ),
            submission_results=self._results.copy(),
            model_stats=model_stats,
        )

        logger.success(
            f"Workload generation complete: "
            f"submitted={result.successful_submissions}/{result.total_tasks}, "
            f"failed={result.failed_submissions}, "
            f"actual_qps={result.actual_qps:.2f}, "
            f"duration={result.actual_duration:.2f}s"
        )

        return result

    def _build_schedule(self) -> list[tuple[float, str]]:
        """Build task submission schedule.

        Interleaves tasks from different models based on their QPS ratios.

        Returns:
            List of (relative_time, model_id) tuples sorted by time.
        """
        schedule = []

        for model_id in self.config.model_ids:
            qps = self.config.model_qps.get(model_id, 0)
            if qps <= 0:
                continue

            interval = 1.0 / qps
            task_count = self.config.get_model_task_count(model_id)

            for i in range(task_count):
                submit_time = i * interval
                schedule.append((submit_time, model_id))

        # Sort by submit time
        schedule.sort(key=lambda x: x[0])
        return schedule

    async def _submit_task(
        self,
        client: httpx.AsyncClient,
        model_id: str,
        task_num: int,
    ) -> None:
        """Submit a single task to the correct scheduler.

        Args:
            client: HTTP client to use.
            model_id: Model this task is for.
            task_num: Global task number.
        """
        async with self._semaphore:
            scheduler_url = self.config.scheduler_urls[model_id]

            # Generate random sleep time
            sleep_time = random.uniform(*self.config.sleep_time_range)

            # Generate unique task ID
            model_task_num = self._task_counters[model_id]
            self._task_counters[model_id] += 1
            task_id = (
                f"{self.config.task_id_prefix}-{model_id}-"
                f"{model_task_num:06d}-{uuid.uuid4().hex[:6]}"
            )

            # Build request payload
            # Note: exp_runtime triggers experiment mode in the scheduler,
            # which bypasses the trained model requirement for scheduling.
            # exp_runtime is in milliseconds.
            # path: Custom endpoint path for the sleep model instance
            # (scheduler defaults to /v1/completions for vLLM, we use /inference
            # which accepts just {"sleep_time": X} matching what the scheduler sends)
            payload = {
                "task_id": task_id,
                "model_id": model_id,
                "task_input": {"sleep_time": sleep_time},
                "metadata": {
                    "sleep_time": sleep_time,
                    "exp_runtime": sleep_time * 1000,  # Convert to ms
                    "path": "inference",  # Route to sleep model inference endpoint
                },
            }

            submit_start = time.time()

            try:
                response = await client.post(
                    f"{scheduler_url}/task/submit",
                    json=payload,
                )

                response_time_ms = (time.time() - submit_start) * 1000
                success = response.status_code == 200

                if success:
                    scheduler_response = response.json()
                    error = None
                    logger.debug(
                        f"Task {task_id} submitted to {model_id} scheduler "
                        f"(sleep={sleep_time:.3f}s, response_time={response_time_ms:.1f}ms)"
                    )
                else:
                    scheduler_response = None
                    error = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.warning(f"Task {task_id} submission failed: {error}")

            except httpx.TimeoutException:
                response_time_ms = (time.time() - submit_start) * 1000
                success = False
                error = "Request timeout"
                scheduler_response = None
                logger.warning(f"Task {task_id} submission timed out")

            except httpx.HTTPError as e:
                response_time_ms = (time.time() - submit_start) * 1000
                success = False
                error = f"HTTP error: {str(e)}"
                scheduler_response = None
                logger.warning(f"Task {task_id} submission failed: {error}")

            except Exception as e:
                response_time_ms = (time.time() - submit_start) * 1000
                success = False
                error = f"Unexpected error: {str(e)}"
                scheduler_response = None
                logger.error(f"Task {task_id} submission failed unexpectedly: {e}")

            # Record result
            result = MultiSchedulerSubmissionResult(
                task_id=task_id,
                model_id=model_id,
                scheduler_url=scheduler_url,
                sleep_time=sleep_time,
                submitted_at=submit_start,
                response_time_ms=response_time_ms,
                success=success,
                error=error,
                scheduler_response=scheduler_response,
            )
            self._results.append(result)

    def _calculate_model_stats(self) -> dict[str, dict[str, Any]]:
        """Calculate per-model statistics.

        Returns:
            Dict mapping model_id to stats dict.
        """
        stats = {}
        for model_id in self.config.model_ids:
            model_results = [r for r in self._results if r.model_id == model_id]
            successful = [r for r in model_results if r.success]

            latencies = [r.response_time_ms for r in successful]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            stats[model_id] = {
                "submitted": len(model_results),
                "successful": len(successful),
                "failed": len(model_results) - len(successful),
                "target_qps": self.config.model_qps.get(model_id, 0),
                "scheduler_url": self.config.scheduler_urls.get(model_id, ""),
                "avg_latency_ms": avg_latency,
            }
        return stats


async def wait_for_multi_scheduler_tasks(
    scheduler_urls: dict[str, str],
    model_task_ids: dict[str, list[str]],
    timeout_seconds: float = 300.0,
    poll_interval: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Wait for tasks to complete across multiple schedulers.

    Polls each scheduler for its tasks' status.

    Args:
        scheduler_urls: Mapping of model_id to scheduler URL.
        model_task_ids: Mapping of model_id to list of task IDs.
        timeout_seconds: Maximum time to wait.
        poll_interval: Interval between polls.

    Returns:
        Dict mapping task_id to task info.
    """
    total_tasks = sum(len(ids) for ids in model_task_ids.values())
    logger.info(f"Waiting for {total_tasks} tasks to complete...")

    start_time = time.time()
    completed_tasks: dict[str, dict[str, Any]] = {}

    # Track pending tasks per model
    pending_per_model: dict[str, set[str]] = {
        model_id: set(task_ids) for model_id, task_ids in model_task_ids.items()
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        while (
            any(pending_per_model.values())
            and (time.time() - start_time) < timeout_seconds
        ):
            # Poll each scheduler for its pending tasks
            for model_id, pending_tasks in pending_per_model.items():
                if not pending_tasks:
                    continue

                scheduler_url = scheduler_urls[model_id]
                pending_list = list(pending_tasks)

                # Poll in batches
                batch_size = 50
                for i in range(0, len(pending_list), batch_size):
                    batch = pending_list[i : i + batch_size]

                    for task_id in batch:
                        try:
                            response = await client.get(
                                f"{scheduler_url}/task/info",
                                params={"task_id": task_id},
                            )

                            if response.status_code == 200:
                                task_info = response.json()
                                task_data = task_info.get("task", {})
                                status = task_data.get("status")

                                if status in ("completed", "failed"):
                                    completed_tasks[task_id] = task_data
                                    pending_tasks.discard(task_id)

                        except Exception as e:
                            logger.debug(f"Error polling task {task_id}: {e}")

            # Log progress
            completed = len(completed_tasks)
            total_pending = sum(len(p) for p in pending_per_model.values())

            if total_pending > 0:
                logger.debug(
                    f"Task completion: {completed}/{total_tasks} "
                    f"({completed / total_tasks * 100:.1f}%)"
                )
                await asyncio.sleep(poll_interval)

    elapsed = time.time() - start_time
    total_pending = sum(len(p) for p in pending_per_model.values())

    logger.info(
        f"Task completion wait finished: "
        f"completed={len(completed_tasks)}, "
        f"pending={total_pending}, "
        f"elapsed={elapsed:.1f}s"
    )

    return completed_tasks
