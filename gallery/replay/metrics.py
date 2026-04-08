"""Async-safe metrics collector for replay requests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from replay.models import RequestMetrics


class MetricsCollector:
    """Accumulate per-request metrics with optional real-time progress callback.

    Args:
        total_steps: Total number of steps expected across all groups.
        progress_callback: Optional callable invoked after each metric is recorded.
            Signature: ``(completed: int, total: int, metric: RequestMetrics) -> None``.
    """

    def __init__(
        self,
        total_steps: int,
        progress_callback: Callable[[int, int, RequestMetrics], Any] | None = None,
    ) -> None:
        self._total = total_steps
        self._completed = 0
        self._metrics: list[RequestMetrics] = []
        self._lock = asyncio.Lock()
        self._progress_callback = progress_callback

    async def record(self, metric: RequestMetrics) -> None:
        """Record a completed request metric.

        Args:
            metric: The request metric to record.
        """
        async with self._lock:
            self._metrics.append(metric)
            self._completed += 1
            if self._progress_callback is not None:
                self._progress_callback(self._completed, self._total, metric)

    def get_all(self) -> list[RequestMetrics]:
        """Return all recorded metrics.

        Returns:
            Copy of the metrics list.
        """
        return list(self._metrics)
