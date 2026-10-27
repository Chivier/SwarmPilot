"""Relative speed of each kind of hardware, learned from execution times.

The Schedulers report each instance's recent execution time. Fitting

    log t[i] = a[model of i] + c[hardware of i]

over the reporting instances separates how heavy a model is (``a``) from
how slow a kind of hardware is (``c``). A model that runs on one kind of
hardware only says nothing about ``c``, so each fit starts from what earlier
rounds learned, is shifted to agree with it, and is blended into it. The
speed of a kind of hardware is ``exp(-c)``, scaled so the fastest kind seen
is 1.

Execution times include the time a task waits inside the engine, so an
overloaded instance reads slower too. That is what the planner wants: it
should not count on capacity an instance does not deliver.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator


def _means(pairs: Iterator[tuple[str, float]]) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for key, value in pairs:
        sums[key].append(value)
    return {k: sum(v) / len(v) for k, v in sums.items()}


class HardwareSpeeds:
    """Smoothed relative speed per hardware name."""

    def __init__(
        self, alpha: float = 0.5, floor: float = 0.05, iterations: int = 20
    ):
        """Initialize the estimator.

        Args:
            alpha: Weight of a new fit in the running estimate (0 to 1].
            floor: Smallest speed reported, so no instance counts as
                nothing.
            iterations: Alternating least-squares passes per fit.
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.floor = floor
        self.iterations = iterations
        # Hardware -> log slowness; only differences matter.
        self._c: dict[str, float] = {}

    def observe(self, samples: Iterable[tuple[str, str, float]]) -> None:
        """Fold one round of execution times into the estimate.

        Args:
            samples: (model, hardware, execution time in seconds) per
                instance.
        """
        obs = [(m, h, math.log(t)) for m, h, t in samples if t > 0]
        if not obs:
            return
        seen = {h for _, h, _ in obs}
        c = {h: self._c.get(h, 0.0) for h in seen}
        for _ in range(self.iterations):
            a = _means((m, lt - c[h]) for m, h, lt in obs)
            c = _means((h, lt - a[m]) for m, h, lt in obs)
        known = [h for h in seen if h in self._c]
        if known:
            shift = sum(self._c[h] - c[h] for h in known) / len(known)
        else:
            shift = -min(c.values())
        for h in seen:
            fit = c[h] + shift
            old = self._c.get(h)
            self._c[h] = (
                fit
                if old is None
                else (1 - self.alpha) * old + self.alpha * fit
            )

    def speed(self, hardware: str) -> float | None:
        """Speed of a kind of hardware relative to the fastest, or None."""
        if hardware not in self._c:
            return None
        best = min(self._c.values())
        return max(self.floor, math.exp(best - self._c[hardware]))

    def snapshot(self) -> dict[str, float]:
        """Speed of every kind of hardware seen."""
        return {h: round(self.speed(h) or 0.0, 3) for h in sorted(self._c)}
