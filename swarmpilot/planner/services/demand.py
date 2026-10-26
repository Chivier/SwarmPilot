"""Demand the Schedulers report, smoothed for the auto-optimizer.

Each Scheduler reports its outstanding work (``/v1/submit_target``) and the
recent execution time of each instance (``/v1/submit_throughput``). The
store keeps an exponentially weighted average of each, with the time of
the last report, so the auto-optimizer can ignore models that stopped
reporting and does not react to a single noisy sample.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


def normalize_url(url: str) -> str:
    """Return an endpoint URL in the form both services agree on."""
    return url.rstrip("/")


@dataclass(slots=True)
class _Reading:
    value: float
    at: float


class DemandStore:
    """Smoothed per-model demand and per-instance execution times."""

    def __init__(
        self,
        alpha: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Initialize the store.

        Args:
            alpha: Weight of a new report in the running average (0 to 1].
                1 keeps only the latest report.
            clock: Time source, replaceable in tests.
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._clock = clock
        self._targets: dict[str, _Reading] = {}
        self._exec_times: dict[str, _Reading] = {}

    def _update(
        self, table: dict[str, _Reading], key: str, value: float
    ) -> float:
        now = self._clock()
        old = table.get(key)
        smoothed = (
            value
            if old is None
            else self._alpha * value + (1 - self._alpha) * old.value
        )
        table[key] = _Reading(smoothed, now)
        return smoothed

    def submit_target(self, model_id: str, value: float) -> float:
        """Record a model's outstanding work and return the smoothed value."""
        return self._update(self._targets, model_id, value)

    def submit_throughput(self, instance_url: str, seconds: float) -> float:
        """Record an instance's mean execution time; return it smoothed."""
        return self._update(
            self._exec_times, normalize_url(instance_url), seconds
        )

    def targets(self, max_age: float | None = None) -> dict[str, float]:
        """Smoothed demand per model, leaving out stale reports.

        Args:
            max_age: Seconds after which a model's report is ignored.
        """
        now = self._clock()
        return {
            m: r.value
            for m, r in self._targets.items()
            if max_age is None or now - r.at <= max_age
        }

    def execution_time(
        self, instance_url: str, max_age: float | None = None
    ) -> float | None:
        """Smoothed execution time of an instance in seconds, if fresh."""
        r = self._exec_times.get(normalize_url(instance_url))
        if r is None or (
            max_age is not None and self._clock() - r.at > max_age
        ):
            return None
        return r.value

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Everything the store holds, for the inspection endpoint."""
        now = self._clock()
        return {
            "targets": {m: r.value for m, r in self._targets.items()},
            "execution_times": {
                u: r.value for u, r in self._exec_times.items()
            },
            "target_age_s": {m: now - r.at for m, r in self._targets.items()},
        }

    def clear(self) -> None:
        """Forget every report."""
        self._targets.clear()
        self._exec_times.clear()


_store: DemandStore | None = None


def get_demand_store() -> DemandStore:
    """Return the process-wide demand store."""
    global _store
    if _store is None:
        _store = DemandStore()
    return _store
