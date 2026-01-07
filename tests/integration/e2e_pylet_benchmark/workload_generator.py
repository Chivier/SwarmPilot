"""Workload Generator for E2E Testing.

This module provides a QPS-based workload generator that submits tasks
to the scheduler at a specified rate. It supports multiple model types
and tracks submission results for analysis.

Features:
- Configurable QPS and duration
- Round-robin or weighted model selection
- Random sleep_time generation within range
- Async httpx for concurrent submissions
- Detailed result tracking

Usage:
    from workload_generator import WorkloadGenerator, WorkloadConfig

    config = WorkloadConfig(
        scheduler_url="http://localhost:8000",
        qps=5.0,
        duration_seconds=60.0,
        model_ids=["sleep_model_a", "sleep_model_b", "sleep_model_c"],
    )

    generator = WorkloadGenerator(config)
    result = await generator.generate_workload()
"""

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger


@dataclass
class WorkloadConfig:
    """Configuration for workload generation."""

    scheduler_url: str
    qps: float = 5.0
    duration_seconds: float = 60.0
    model_ids: list[str] = field(
        default_factory=lambda: ["sleep_model_a", "sleep_model_b", "sleep_model_c"]
    )
    sleep_time_range: tuple[float, float] = (0.1, 1.0)
    task_id_prefix: str = "bench"
    timeout_seconds: float = 30.0
    max_concurrent_submissions: int = 50

    @property
    def total_tasks(self) -> int:
        """Calculate total number of tasks to submit."""
        return int(self.qps * self.duration_seconds)

    @property
    def interval(self) -> float:
        """Calculate interval between task submissions."""
        return 1.0 / self.qps if self.qps > 0 else 1.0


@dataclass
class SubmissionResult:
    """Result of a single task submission."""

    task_id: str
    model_id: str
    sleep_time: float
    submitted_at: float
    response_time_ms: float
    success: bool
    error: str | None = None
    scheduler_response: dict[str, Any] | None = None


@dataclass
class WorkloadResult:
    """Aggregate result of workload generation."""

    # Task counts
    total_tasks: int
    successful_submissions: int
    failed_submissions: int

    # Timing
    start_time: float
    end_time: float
    actual_duration: float

    # QPS metrics
    target_qps: float
    actual_qps: float

    # Detailed results
    submission_results: list[SubmissionResult]

    # Per-model breakdown
    model_stats: dict[str, dict[str, int]]

    def submission_success_rate(self) -> float:
        """Calculate submission success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_submissions / self.total_tasks

    def get_submission_latencies(self) -> list[float]:
        """Get list of submission latencies in ms."""
        return [r.response_time_ms for r in self.submission_results if r.success]

    def get_failed_submissions(self) -> list[SubmissionResult]:
        """Get list of failed submissions."""
        return [r for r in self.submission_results if not r.success]


class WorkloadGenerator:
    """QPS-based workload generator for scheduler testing."""

    def __init__(self, config: WorkloadConfig):
        """Initialize workload generator.

        Args:
            config: Workload configuration
        """
        self.config = config
        self._results: list[SubmissionResult] = []
        self._semaphore: asyncio.Semaphore | None = None
        self._submission_count = 0

    async def generate_workload(self) -> WorkloadResult:
        """Generate QPS-based workload.

        Submits tasks at the configured QPS rate for the specified duration.
        Uses async concurrency with semaphore to limit parallel submissions.

        Returns:
            WorkloadResult with aggregate metrics and detailed results
        """
        logger.info(
            f"Starting workload generation: "
            f"qps={self.config.qps}, "
            f"duration={self.config.duration_seconds}s, "
            f"total_tasks={self.config.total_tasks}, "
            f"models={self.config.model_ids}"
        )

        self._results = []
        self._submission_count = 0
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_submissions)

        start_time = time.time()
        tasks = []

        async with httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            limits=httpx.Limits(
                max_connections=self.config.max_concurrent_submissions * 2,
                max_keepalive_connections=self.config.max_concurrent_submissions,
            ),
        ) as client:
            for i in range(self.config.total_tasks):
                # Calculate when this task should be submitted
                target_submit_time = start_time + i * self.config.interval
                current_time = time.time()

                # Wait until target time (if we're ahead of schedule)
                if current_time < target_submit_time:
                    await asyncio.sleep(target_submit_time - current_time)

                # Create submission task
                task = asyncio.create_task(self._submit_task(client, i))
                tasks.append(task)

                # Log progress every 10%
                if (i + 1) % max(1, self.config.total_tasks // 10) == 0:
                    elapsed = time.time() - start_time
                    current_qps = (i + 1) / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {i + 1}/{self.config.total_tasks} tasks submitted "
                        f"({(i + 1) / self.config.total_tasks * 100:.1f}%), "
                        f"current_qps={current_qps:.2f}"
                    )

            # Wait for all submissions to complete
            logger.info("Waiting for all submissions to complete...")
            await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        actual_duration = end_time - start_time

        # Calculate per-model statistics
        model_stats = self._calculate_model_stats()

        result = WorkloadResult(
            total_tasks=self.config.total_tasks,
            successful_submissions=sum(1 for r in self._results if r.success),
            failed_submissions=sum(1 for r in self._results if not r.success),
            start_time=start_time,
            end_time=end_time,
            actual_duration=actual_duration,
            target_qps=self.config.qps,
            actual_qps=len(self._results) / actual_duration if actual_duration > 0 else 0,
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

    async def _submit_task(self, client: httpx.AsyncClient, task_num: int) -> None:
        """Submit a single task to the scheduler.

        Args:
            client: HTTP client to use
            task_num: Task number (for ID generation and model selection)
        """
        async with self._semaphore:
            # Select model (round-robin)
            model_id = self.config.model_ids[task_num % len(self.config.model_ids)]

            # Generate random sleep time
            sleep_time = random.uniform(*self.config.sleep_time_range)

            # Generate unique task ID
            task_id = f"{self.config.task_id_prefix}-{task_num:06d}-{uuid.uuid4().hex[:6]}"

            # Build request payload
            payload = {
                "task_id": task_id,
                "model_id": model_id,
                "task_input": {"sleep_time": sleep_time},
                "metadata": {"sleep_time": sleep_time},  # For predictor
            }

            submit_start = time.time()

            try:
                response = await client.post(
                    f"{self.config.scheduler_url}/task/submit",
                    json=payload,
                )

                response_time_ms = (time.time() - submit_start) * 1000
                success = response.status_code == 200

                if success:
                    scheduler_response = response.json()
                    error = None
                    logger.debug(
                        f"Task {task_id} submitted successfully "
                        f"(model={model_id}, sleep={sleep_time:.3f}s, "
                        f"response_time={response_time_ms:.1f}ms)"
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
            result = SubmissionResult(
                task_id=task_id,
                model_id=model_id,
                sleep_time=sleep_time,
                submitted_at=submit_start,
                response_time_ms=response_time_ms,
                success=success,
                error=error,
                scheduler_response=scheduler_response,
            )
            self._results.append(result)
            self._submission_count += 1

    def _calculate_model_stats(self) -> dict[str, dict[str, int]]:
        """Calculate per-model statistics.

        Returns:
            Dict mapping model_id to stats dict
        """
        stats = {}
        for model_id in self.config.model_ids:
            model_results = [r for r in self._results if r.model_id == model_id]
            stats[model_id] = {
                "submitted": len(model_results),
                "successful": sum(1 for r in model_results if r.success),
                "failed": sum(1 for r in model_results if not r.success),
            }
        return stats


async def wait_for_task_completion(
    scheduler_url: str,
    task_ids: list[str],
    timeout_seconds: float = 300.0,
    poll_interval: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Wait for all tasks to complete and collect results.

    Args:
        scheduler_url: Scheduler URL
        task_ids: List of task IDs to wait for
        timeout_seconds: Maximum time to wait
        poll_interval: Interval between status polls

    Returns:
        Dict mapping task_id to task info
    """
    logger.info(f"Waiting for {len(task_ids)} tasks to complete...")

    start_time = time.time()
    completed_tasks: dict[str, dict[str, Any]] = {}
    pending_tasks = set(task_ids)

    async with httpx.AsyncClient(timeout=30.0) as client:
        while pending_tasks and (time.time() - start_time) < timeout_seconds:
            # Poll for task status in batches
            batch_size = 50
            pending_list = list(pending_tasks)

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
                            # Response structure: {"success": true, "task": {...}}
                            task_data = task_info.get("task", {})
                            status = task_data.get("status")

                            if status in ("completed", "failed"):
                                completed_tasks[task_id] = task_data
                                pending_tasks.discard(task_id)

                    except Exception as e:
                        logger.debug(f"Error polling task {task_id}: {e}")

            if pending_tasks:
                # Log progress
                completed = len(task_ids) - len(pending_tasks)
                logger.debug(
                    f"Task completion: {completed}/{len(task_ids)} "
                    f"({completed / len(task_ids) * 100:.1f}%)"
                )
                await asyncio.sleep(poll_interval)

    elapsed = time.time() - start_time
    logger.info(
        f"Task completion wait finished: "
        f"completed={len(completed_tasks)}, "
        f"pending={len(pending_tasks)}, "
        f"elapsed={elapsed:.1f}s"
    )

    return completed_tasks
