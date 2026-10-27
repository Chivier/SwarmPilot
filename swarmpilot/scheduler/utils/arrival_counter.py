"""Arrivals per reporting period, by calling model (plan item 3.4).

The Planner's tail scaler needs each model's arrival rate and how many of
its tasks other models' tasks made. A client marks a call made on behalf
of another model's task with the ``X-SwarmPilot-Parent`` header on the
proxy path, or ``metadata["parent_model"]`` on ``/v1/task/submit``.

Counting is two dictionary updates on the event loop, so it adds nothing
measurable to the request path.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

PARENT_HEADER = "x-swarmpilot-parent"
PARENT_KEY = "parent_model"


@dataclass(frozen=True)
class ArrivalWindow:
    """Arrivals in one reporting period.

    Attributes:
        seconds: Length of the period.
        arrivals: Tasks submitted.
        calls_from: Of those, tasks made on behalf of each calling model.
    """

    seconds: float
    arrivals: int
    calls_from: dict[str, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        """Tasks per second."""
        return self.arrivals / self.seconds if self.seconds > 0 else 0.0

    def call_rates(self) -> dict[str, float]:
        """Calls per second by calling model."""
        if self.seconds <= 0:
            return {}
        return {p: n / self.seconds for p, n in self.calls_from.items()}


class ArrivalCounter:
    """Counts task arrivals until the reporter takes the window."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        """Initialize the counter.

        Args:
            clock: Time source, replaceable in tests.
        """
        self._clock = clock
        self._since = clock()
        self._arrivals = 0
        self._calls: Counter[str] = Counter()

    def record(self, parent: str | None = None) -> None:
        """Count one task, made on behalf of ``parent`` if given."""
        self._arrivals += 1
        if parent:
            self._calls[parent] += 1

    def add(self, count: int, calls_from: dict[str, int]) -> None:
        """Count ``count`` tasks another data plane received.

        Args:
            count: Tasks that arrived.
            calls_from: Of those, tasks made on behalf of each model.
        """
        self._arrivals += count
        self._calls.update(calls_from)

    def take(self) -> ArrivalWindow:
        """Return the arrivals since the last call and start a new window."""
        now = self._clock()
        window = ArrivalWindow(
            seconds=now - self._since,
            arrivals=self._arrivals,
            calls_from=dict(self._calls),
        )
        self._since = now
        self._arrivals = 0
        self._calls = Counter()
        return window


def parent_of(metadata: dict | None) -> str | None:
    """The calling model recorded in task metadata, if any."""
    if not metadata:
        return None
    parent = metadata.get(PARENT_KEY)
    return parent if isinstance(parent, str) and parent else None


arrival_counter = ArrivalCounter()
