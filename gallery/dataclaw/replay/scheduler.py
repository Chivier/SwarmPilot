"""Replay scheduler with Poisson arrival, within-group chaining, and global QPS limiting."""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING

from replay.models import GroupMetrics, ReplayGroup, RequestMetrics

if TYPE_CHECKING:
    from replay.client import ReplayClient
    from replay.metrics import MetricsCollector


class TokenBucketLimiter:
    """Async token bucket for global QPS rate limiting.

    Tokens refill continuously at the configured rate.  Each ``acquire()``
    call blocks until a token is available, ensuring the sustained request
    rate never exceeds the limit.

    Args:
        rate: Maximum requests per second.
    """

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # No token available; back off briefly.
            await asyncio.sleep(1.0 / self._rate)


class ReplayScheduler:
    """Orchestrate concurrent replay of multiple groups.

    Groups are launched with Poisson-distributed inter-arrival times.
    Within each group, steps execute sequentially with configured delays.
    A shared token bucket enforces the global QPS ceiling.

    Args:
        poisson_qps: Rate parameter for Poisson first-request arrivals.
        global_qps: Maximum requests per second across all groups.
        agent_delay_ms: Delay (ms) after an agent-generated response.
        user_delay_ms: Delay (ms) after a user-generated response.
        client: Async LLM client for sending requests.
        collector: Metrics collector for recording results.
    """

    def __init__(
        self,
        poisson_qps: float,
        global_qps: float,
        agent_delay_ms: int,
        user_delay_ms: int,
        client: ReplayClient,
        collector: MetricsCollector,
    ) -> None:
        self._poisson_qps = poisson_qps
        self._agent_delay_s = agent_delay_ms / 1000.0
        self._user_delay_s = user_delay_ms / 1000.0
        self._client = client
        self._collector = collector
        self._limiter = TokenBucketLimiter(global_qps)

    async def run_all(self, groups: list[ReplayGroup]) -> list[GroupMetrics]:
        """Schedule and execute all replay groups concurrently.

        Each group's first request is staggered by a Poisson-distributed
        inter-arrival time.  Subsequent steps within a group chain off the
        prior response with the configured delay.

        Args:
            groups: List of ReplayGroups to execute.

        Returns:
            Per-group metrics for all completed groups.
        """
        if not groups:
            return []

        # Generate cumulative Poisson offsets for each group's start time.
        offsets = self._generate_poisson_offsets(len(groups))

        tasks = [
            asyncio.create_task(self._run_group(group, offset))
            for group, offset in zip(groups, offsets)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        group_metrics: list[GroupMetrics] = []
        for result in results:
            if isinstance(result, GroupMetrics):
                group_metrics.append(result)
            elif isinstance(result, BaseException):
                # Log but don't crash on unexpected errors.
                pass
        return group_metrics

    def _generate_poisson_offsets(self, count: int) -> list[float]:
        """Generate cumulative Poisson inter-arrival offsets.

        Args:
            count: Number of groups.

        Returns:
            Sorted cumulative offset list (first is always 0).
        """
        offsets: list[float] = [0.0]
        cumulative = 0.0
        for _ in range(count - 1):
            inter_arrival = random.expovariate(self._poisson_qps)
            cumulative += inter_arrival
            offsets.append(cumulative)
        return offsets

    async def _run_group(
        self, group: ReplayGroup, start_offset: float
    ) -> GroupMetrics:
        """Execute one replay group after its Poisson-scheduled delay.

        Args:
            group: The replay group to execute.
            start_offset: Seconds to wait before sending the first request.

        Returns:
            GroupMetrics for this group.
        """
        await asyncio.sleep(start_offset)

        cumulative: list[dict] = list(group.initial_messages)
        request_metrics: list[RequestMetrics] = []
        group_start = time.monotonic()

        for i, step in enumerate(group.steps):
            # Append this step's message to the cumulative history.
            cumulative.append(step.history_message)

            # Inter-step delay (not applied to the first step—Poisson covers that).
            if i > 0:
                if step.sender_role == "user":
                    await asyncio.sleep(self._user_delay_s)
                else:
                    await asyncio.sleep(self._agent_delay_s)

            # Wait for a global QPS token.
            await self._limiter.acquire()

            # Send request with a copy of the current history.
            metrics = await self._client.send(
                messages=list(cumulative),
                model_size=step.model_size,
                group_id=group.group_id,
                step_index=step.step_index,
            )
            request_metrics.append(metrics)
            await self._collector.record(metrics)

        group_end = time.monotonic()

        completed = sum(1 for m in request_metrics if m.status == "success")
        failed = sum(1 for m in request_metrics if m.status != "success")

        return GroupMetrics(
            group_id=group.group_id,
            dataset_name=group.dataset_name,
            total_steps=len(group.steps),
            completed_steps=completed,
            failed_steps=failed,
            total_latency_ms=(group_end - group_start) * 1000,
            request_metrics=request_metrics,
        )
