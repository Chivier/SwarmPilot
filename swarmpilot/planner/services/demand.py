"""Demand the Schedulers report, smoothed for the auto-optimizer.

Each Scheduler reports its outstanding work (``/v1/submit_target``) and the
recent execution time of each instance (``/v1/submit_throughput``). The
store keeps an exponentially weighted average of each, with the time of
the last report, so the auto-optimizer can ignore models that stopped
reporting and does not react to a single noisy sample.

A Scheduler may add its arrival rate, a q0.9 estimate of its outstanding
work, and how many of its tasks other models' tasks made (item 3.4). From
those the store estimates the call ratio ``k[i][j]``: tasks of model ``j``
each task of model ``i`` causes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

# A task fanning out further than this is more likely a reporting glitch
# (a caller whose arrivals were missed) than a real call structure.
MAX_CALL_RATIO = 64.0


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
        self._rates: dict[str, _Reading] = {}
        self._work: dict[str, _Reading] = {}
        # (caller, callee) -> calls per second.
        self._calls: dict[tuple[str, str], _Reading] = {}

    def _update(self, table: dict[Any, _Reading], key: Hashable, value: float) -> float:
        now = self._clock()
        old = table.get(key)
        smoothed = (
            value
            if old is None
            else self._alpha * value + (1 - self._alpha) * old.value
        )
        table[key] = _Reading(smoothed, now)
        return smoothed

    def submit_target(
        self,
        model_id: str,
        value: float,
        arrival_rate: float | None = None,
        work_s: float | None = None,
        calls_from: dict[str, float] | None = None,
    ) -> float:
        """Record a model's outstanding work and return the smoothed value.

        Args:
            model_id: Reporting model.
            value: Tasks pending or running.
            arrival_rate: Tasks per second since the last report.
            work_s: q0.9 outstanding work in instance-seconds.
            calls_from: Calls per second from each calling model. Only
                read alongside ``arrival_rate``; callers left out made
                no calls this period.
        """
        if work_s is not None:
            self._update(self._work, model_id, work_s)
        if arrival_rate is not None:
            self._update(self._rates, model_id, arrival_rate)
            calls = calls_from or {}
            known = {p for p, c in self._calls if c == model_id}
            for parent in known | set(calls):
                self._update(self._calls, (parent, model_id), calls.get(parent, 0.0))
        return self._update(self._targets, model_id, value)

    def submit_throughput(self, instance_url: str, seconds: float) -> float:
        """Record an instance's mean execution time; return it smoothed."""
        return self._update(self._exec_times, normalize_url(instance_url), seconds)

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

    def _fresh(
        self, table: dict[Any, _Reading], max_age: float | None
    ) -> dict[Any, float]:
        now = self._clock()
        return {
            k: r.value
            for k, r in table.items()
            if max_age is None or now - r.at <= max_age
        }

    def arrival_rates(self, max_age: float | None = None) -> dict[str, float]:
        """Smoothed tasks per second per model, from fresh reports."""
        return self._fresh(self._rates, max_age)

    def work(self, max_age: float | None = None) -> dict[str, float]:
        """Smoothed q0.9 outstanding work in seconds per model."""
        return self._fresh(self._work, max_age)

    def call_ratios(self, max_age: float | None = None) -> dict[str, dict[str, float]]:
        """Tasks of each callee one task of each caller causes.

        Returns:
            ``{caller: {callee: k}}`` for callers with a fresh, non-zero
            arrival rate; ``k`` is capped at ``MAX_CALL_RATIO``.
        """
        rates = self.arrival_rates(max_age)
        out: dict[str, dict[str, float]] = {}
        for (parent, child), calls in self._fresh(self._calls, max_age).items():
            if rates.get(parent, 0.0) > 0 and calls > 0:
                out.setdefault(parent, {})[child] = min(
                    calls / rates[parent], MAX_CALL_RATIO
                )
        return out

    def execution_time(
        self, instance_url: str, max_age: float | None = None
    ) -> float | None:
        """Smoothed execution time of an instance in seconds, if fresh."""
        r = self._exec_times.get(normalize_url(instance_url))
        if r is None or (max_age is not None and self._clock() - r.at > max_age):
            return None
        return r.value

    def snapshot(self) -> dict[str, Any]:
        """Everything the store holds, for the inspection endpoint."""
        now = self._clock()
        return {
            "targets": {m: r.value for m, r in self._targets.items()},
            "execution_times": {u: r.value for u, r in self._exec_times.items()},
            "target_age_s": {m: now - r.at for m, r in self._targets.items()},
            "arrival_rates": self.arrival_rates(),
            "work_s": self.work(),
            "call_ratios": self.call_ratios(),
        }

    def clear(self) -> None:
        """Forget every report."""
        for table in (
            self._targets,
            self._exec_times,
            self._rates,
            self._work,
            self._calls,
        ):
            table.clear()


_store: DemandStore | None = None


def get_demand_store() -> DemandStore:
    """Return the process-wide demand store."""
    global _store
    if _store is None:
        _store = DemandStore()
    return _store
