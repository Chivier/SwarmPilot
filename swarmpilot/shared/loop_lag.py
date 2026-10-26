"""Event-loop lag probe.

A small background task that sleeps for a fixed interval and records how
late it wakes up. Any blocking call on the event loop (a synchronous model
fit, a blocking HTTP call, a large JSON dump) shows up directly as lag, so
this is the cheapest way to see that the loop is stalled.

The probe keeps a fixed-bucket histogram, so memory stays constant no matter
how long the service runs, and logs a warning when one wake-up is later than
``warn_ms``.
"""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import time
from typing import Any

from loguru import logger

# Upper bucket bounds in milliseconds. The last bucket is open-ended.
DEFAULT_BUCKETS_MS: tuple[float, ...] = (
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
)


class EventLoopLagMonitor:
    """Measure event-loop wake-up lag as a histogram.

    Example:
        ```python
        monitor = EventLoopLagMonitor(interval_s=0.05)
        monitor.start()
        ...
        print(monitor.snapshot())
        await monitor.stop()
        ```
    """

    def __init__(
        self,
        interval_s: float = 0.05,
        warn_ms: float = 100.0,
        buckets_ms: tuple[float, ...] = DEFAULT_BUCKETS_MS,
    ):
        """Create a monitor.

        Args:
            interval_s: How often the probe wakes up.
            warn_ms: Log a warning when one wake-up is this late or later.
            buckets_ms: Ascending histogram bucket upper bounds in ms.
        """
        self.interval_s = interval_s
        self.warn_ms = warn_ms
        self._bounds = tuple(sorted(buckets_ms))
        self._task: asyncio.Task | None = None
        self.reset()

    def reset(self) -> None:
        """Clear all recorded samples."""
        self._counts = [0] * (len(self._bounds) + 1)
        self._count = 0
        self._sum_ms = 0.0
        self._max_ms = 0.0
        self._started_at = time.time()

    @property
    def running(self) -> bool:
        """Whether the probe task is active."""
        return self._task is not None and not self._task.done()

    def record(self, lag_ms: float) -> None:
        """Add one lag sample to the histogram.

        Args:
            lag_ms: How late the probe woke up, in milliseconds.
        """
        lag_ms = max(0.0, lag_ms)
        self._counts[bisect.bisect_left(self._bounds, lag_ms)] += 1
        self._count += 1
        self._sum_ms += lag_ms
        self._max_ms = max(self._max_ms, lag_ms)
        if lag_ms >= self.warn_ms:
            logger.warning(f"[loop_lag] event loop blocked for {lag_ms:.1f} ms")

    def start(self) -> None:
        """Start the probe on the running event loop (idempotent)."""
        if self.running:
            return
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name="event-loop-lag-probe"
        )

    async def stop(self) -> None:
        """Stop the probe and wait for it to exit."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            expected = loop.time() + self.interval_s
            await asyncio.sleep(self.interval_s)
            self.record((loop.time() - expected) * 1000)

    def quantile(self, q: float) -> float:
        """Estimate the q-quantile of lag in ms from the histogram.

        Returns the upper bound of the bucket that holds the quantile, or
        the observed maximum for the open-ended last bucket.

        Args:
            q: Quantile in [0, 1].

        Returns:
            Estimated lag in milliseconds (0.0 with no samples).
        """
        if not self._count:
            return 0.0
        target = q * self._count
        seen = 0
        for i, n in enumerate(self._counts):
            seen += n
            if seen >= target and n:
                if i < len(self._bounds):
                    return min(float(self._bounds[i]), self._max_ms)
                return self._max_ms
        return self._max_ms

    def snapshot(self) -> dict[str, Any]:
        """Return the histogram and summary statistics as a dict."""
        labels = [f"le_{b:g}ms" for b in self._bounds] + ["inf"]
        return {
            "interval_ms": self.interval_s * 1000,
            "samples": self._count,
            "since": self._started_at,
            "mean_ms": round(self._sum_ms / self._count, 3)
            if self._count
            else 0.0,
            "p50_ms": self.quantile(0.5),
            "p99_ms": self.quantile(0.99),
            "max_ms": round(self._max_ms, 3),
            "buckets": dict(zip(labels, self._counts)),
        }
