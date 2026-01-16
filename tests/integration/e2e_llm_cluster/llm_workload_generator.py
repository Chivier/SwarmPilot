"""LLM Workload Generator with Configurable QPS Ratios.

This module generates OpenAI-compatible chat completion requests
with specified QPS ratios across multiple models.

Configuration:
- QPS ratio 5:1:3 for llm_fast:llm_medium:llm_slow
- Submits tasks through scheduler (not directly to instances)
- Returns results in OpenAI-compatible format
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
class LLMWorkloadConfig:
    """Configuration for LLM workload generation."""

    scheduler_url: str = "http://localhost:8000"
    total_qps: float = 10.0  # Total QPS across all models
    duration_seconds: float = 60.0

    # Model configuration (QPS ratio 5:1:3)
    model_ids: list[str] = field(
        default_factory=lambda: ["llm_fast", "llm_medium", "llm_slow"]
    )
    qps_ratios: list[float] = field(
        default_factory=lambda: [5.0, 1.0, 3.0]  # 5:1:3 ratio
    )

    # Sample prompts for realistic workload
    sample_prompts: list[str] = field(
        default_factory=lambda: [
            "Explain quantum computing in simple terms.",
            "Write a short poem about artificial intelligence.",
            "What are the key differences between Python and JavaScript?",
            "Summarize the benefits of cloud computing.",
            "How does machine learning work?",
            "Describe the future of renewable energy.",
            "What is the importance of data privacy?",
            "Explain the concept of blockchain.",
            "What are best practices for software testing?",
            "How can we improve user experience in web applications?",
        ]
    )

    def get_model_qps(self) -> dict[str, float]:
        """Calculate QPS for each model based on ratios."""
        total_ratio = sum(self.qps_ratios)
        return {
            model_id: (ratio / total_ratio) * self.total_qps
            for model_id, ratio in zip(self.model_ids, self.qps_ratios)
        }


@dataclass
class SubmissionResult:
    """Result of a single task submission."""

    task_id: str
    model_id: str
    success: bool
    timestamp: float
    latency_ms: float
    error: str | None = None


@dataclass
class LLMWorkloadResult:
    """Results from LLM workload generation."""

    submission_results: list[SubmissionResult]
    start_time: float
    end_time: float
    target_qps: float
    model_qps: dict[str, float]

    @property
    def total_tasks(self) -> int:
        """Total tasks submitted."""
        return len(self.submission_results)

    @property
    def successful_tasks(self) -> int:
        """Number of successfully submitted tasks."""
        return sum(1 for r in self.submission_results if r.success)

    @property
    def failed_tasks(self) -> int:
        """Number of failed submissions."""
        return sum(1 for r in self.submission_results if not r.success)

    @property
    def duration_seconds(self) -> float:
        """Actual duration of workload generation."""
        return self.end_time - self.start_time

    @property
    def actual_qps(self) -> float:
        """Actual QPS achieved."""
        if self.duration_seconds > 0:
            return self.total_tasks / self.duration_seconds
        return 0.0

    def tasks_by_model(self) -> dict[str, int]:
        """Count tasks submitted per model."""
        counts: dict[str, int] = {}
        for r in self.submission_results:
            counts[r.model_id] = counts.get(r.model_id, 0) + 1
        return counts


class LLMWorkloadGenerator:
    """Generates LLM workload with configurable QPS ratios."""

    def __init__(self, config: LLMWorkloadConfig):
        """Initialize the generator.

        Args:
            config: Workload configuration
        """
        self.config = config
        self._task_counter = 0

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        self._task_counter += 1
        return f"llm-{self._task_counter:06d}-{uuid.uuid4().hex[:6]}"

    def _select_model(self) -> str:
        """Select a model based on QPS ratios."""
        total_ratio = sum(self.config.qps_ratios)
        weights = [r / total_ratio for r in self.config.qps_ratios]
        return random.choices(self.config.model_ids, weights=weights)[0]

    def _generate_messages(self) -> list[dict[str, str]]:
        """Generate chat messages for a request."""
        prompt = random.choice(self.config.sample_prompts)
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    async def _submit_task(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        model_id: str,
    ) -> SubmissionResult:
        """Submit a single task to the scheduler.

        Args:
            client: HTTP client
            task_id: Task identifier
            model_id: Model to use

        Returns:
            SubmissionResult with status
        """
        start_time = time.time()

        # Create OpenAI-compatible task input
        task_input = {
            "messages": self._generate_messages(),
            "temperature": 0.7,
            "max_tokens": 256,
        }

        payload = {
            "model_id": model_id,
            "task_id": task_id,
            "task_input": task_input,
            "metadata": {
                "source": "llm_workload_generator",
                "timestamp": start_time,
            },
        }

        try:
            response = await client.post(
                f"{self.config.scheduler_url}/task/submit",
                json=payload,
                timeout=30.0,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return SubmissionResult(
                    task_id=task_id,
                    model_id=model_id,
                    success=True,
                    timestamp=start_time,
                    latency_ms=latency_ms,
                )
            else:
                return SubmissionResult(
                    task_id=task_id,
                    model_id=model_id,
                    success=False,
                    timestamp=start_time,
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return SubmissionResult(
                task_id=task_id,
                model_id=model_id,
                success=False,
                timestamp=start_time,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def generate_workload(self) -> LLMWorkloadResult:
        """Generate LLM workload with configured QPS ratios.

        Returns:
            LLMWorkloadResult with all submission results
        """
        model_qps = self.config.get_model_qps()

        logger.info(
            f"Starting LLM workload: total_qps={self.config.total_qps}, "
            f"duration={self.config.duration_seconds}s"
        )
        for model_id, qps in model_qps.items():
            logger.info(f"  {model_id}: {qps:.2f} QPS")

        results: list[SubmissionResult] = []
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds

        # Calculate interval between tasks
        interval = 1.0 / self.config.total_qps

        async with httpx.AsyncClient() as client:
            next_submit_time = start_time

            while time.time() < end_time:
                current_time = time.time()

                # Wait until next submission time
                if current_time < next_submit_time:
                    await asyncio.sleep(next_submit_time - current_time)

                # Select model and generate task
                model_id = self._select_model()
                task_id = self._generate_task_id()

                # Submit task
                result = await self._submit_task(client, task_id, model_id)
                results.append(result)

                if result.success:
                    logger.debug(
                        f"Submitted: {task_id} -> {model_id} ({result.latency_ms:.2f}ms)"
                    )
                else:
                    logger.warning(f"Failed: {task_id} -> {model_id}: {result.error}")

                # Schedule next submission
                next_submit_time += interval

        actual_end_time = time.time()

        # Log summary
        workload_result = LLMWorkloadResult(
            submission_results=results,
            start_time=start_time,
            end_time=actual_end_time,
            target_qps=self.config.total_qps,
            model_qps=model_qps,
        )

        logger.info(
            f"Workload complete: {workload_result.total_tasks} tasks "
            f"({workload_result.successful_tasks} success, {workload_result.failed_tasks} failed), "
            f"actual_qps={workload_result.actual_qps:.2f}"
        )

        tasks_by_model = workload_result.tasks_by_model()
        for model_id, count in tasks_by_model.items():
            logger.info(f"  {model_id}: {count} tasks")

        return workload_result


async def wait_for_llm_task_completion(
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
        poll_interval: Polling interval in seconds

    Returns:
        Dict mapping task_id to result with OpenAI-compatible response
    """
    logger.info(f"Waiting for {len(task_ids)} tasks to complete...")

    results: dict[str, dict[str, Any]] = {}
    pending = set(task_ids)
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        while pending and (time.time() - start_time) < timeout_seconds:
            # Check status for pending tasks
            for task_id in list(pending):
                try:
                    # Use correct scheduler endpoint: /task/info?task_id=X
                    response = await client.get(
                        f"{scheduler_url}/task/info",
                        params={"task_id": task_id},
                        timeout=10.0,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        # Scheduler returns {success: bool, task: {...}}
                        task_data = data.get("task", {})
                        status = task_data.get("status", "unknown")

                        if status in ("completed", "failed"):
                            # Flatten task data for consistent result format
                            results[task_id] = {
                                "task_id": task_id,
                                "status": status,
                                "result": task_data.get("result"),
                                "error": task_data.get("error"),
                                "execution_time_ms": task_data.get("execution_time_ms"),
                            }
                            pending.discard(task_id)

                            if status == "completed":
                                # Extract OpenAI response from result
                                result = task_data.get("result", {})
                                logger.debug(
                                    f"Task {task_id} completed: "
                                    f"model={result.get('model', 'unknown')}"
                                )
                            else:
                                logger.warning(
                                    f"Task {task_id} failed: {task_data.get('error', 'unknown')}"
                                )

                except Exception as e:
                    logger.debug(f"Error checking {task_id}: {e}")

            if pending:
                await asyncio.sleep(poll_interval)

    elapsed = time.time() - start_time
    completed = len(results)
    timed_out = len(pending)

    logger.info(
        f"Task completion: {completed}/{len(task_ids)} completed, "
        f"{timed_out} timed out after {elapsed:.2f}s"
    )

    # Mark timed-out tasks
    for task_id in pending:
        results[task_id] = {
            "task_id": task_id,
            "status": "timeout",
            "error": f"Timed out after {timeout_seconds}s",
        }

    return results


# Convenience function
async def run_llm_workload(
    scheduler_url: str,
    total_qps: float,
    duration_seconds: float,
    wait_for_completion: bool = True,
    completion_timeout: float = 300.0,
) -> tuple[LLMWorkloadResult, dict[str, dict[str, Any]] | None]:
    """Run a complete LLM workload test.

    Args:
        scheduler_url: Scheduler URL
        total_qps: Total queries per second
        duration_seconds: Test duration
        wait_for_completion: Whether to wait for task completion
        completion_timeout: Timeout for completion wait

    Returns:
        Tuple of (workload_result, task_results or None)
    """
    config = LLMWorkloadConfig(
        scheduler_url=scheduler_url,
        total_qps=total_qps,
        duration_seconds=duration_seconds,
    )

    generator = LLMWorkloadGenerator(config)
    workload_result = await generator.generate_workload()

    task_results = None
    if wait_for_completion:
        successful_ids = [
            r.task_id for r in workload_result.submission_results if r.success
        ]
        if successful_ids:
            task_results = await wait_for_llm_task_completion(
                scheduler_url=scheduler_url,
                task_ids=successful_ids,
                timeout_seconds=completion_timeout,
            )

    return workload_result, task_results
