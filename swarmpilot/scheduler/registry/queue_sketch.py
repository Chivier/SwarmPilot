"""Queue sketches: what each instance still has to work through.

A sketch summarises the predicted runtime of every task routed to an
instance that has not finished yet. Expect-error sketches keep a sum of
expected times and a root-sum-square of error margins. Probabilistic
sketches keep a runtime value per quantile.

Every task's contribution is a :class:`Charge`. The registry keeps one
charge per task, adds it when the task is routed, and subtracts exactly the
same charge when the task completes, fails or moves, so a sketch returns to
empty once its tasks are done.

Probabilistic sketches add tasks comonotonically: the queue's q-quantile
grows by the task's q-quantile. That is what the earlier Monte Carlo update
with shared percentiles converged to, without its sampling noise, and it
makes the update exactly reversible.

An instance that serves ``servers`` requests at once works through its
backlog that many times faster, so each charge is divided by ``servers``:
the sketch is the wait before a new task starts, not the sum of the
runtimes ahead of it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from swarmpilot.scheduler import core
from swarmpilot.scheduler.models import (
    InstanceQueueBase,
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)

# Quantiles below the median too, so the lower half of a runtime
# distribution is not clamped to its median.
DEFAULT_QUANTILES: tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
)


@dataclass(frozen=True, slots=True)
class Charge:
    """One task's predicted runtime, as added to an instance's sketch."""

    instance_id: str
    time_ms: float
    error_ms: float = 0.0
    quantiles: tuple[tuple[float, float], ...] | None = None

    @classmethod
    def of(
        cls,
        instance_id: str,
        time_ms: float,
        error_ms: float | None = None,
        quantiles: Mapping[float, float] | None = None,
    ) -> Charge:
        """Build a charge from prediction fields."""
        return cls(
            instance_id=instance_id,
            time_ms=float(time_ms),
            error_ms=float(error_ms or 0.0),
            quantiles=(
                tuple(
                    sorted((float(q), float(v)) for q, v in quantiles.items())
                )
                if quantiles
                else None
            ),
        )

    def at(self, grid: Sequence[float]) -> list[float]:
        """The task's runtime at each quantile of ``grid``.

        Quantiles between the predicted ones are interpolated linearly;
        quantiles outside them take the nearest predicted value.
        """
        if not self.quantiles:
            return [self.time_ms] * len(grid)
        qs = self.quantiles
        out = []
        for g in grid:
            if g <= qs[0][0]:
                out.append(qs[0][1])
                continue
            if g >= qs[-1][0]:
                out.append(qs[-1][1])
                continue
            for (q0, v0), (q1, v1) in pairwise(qs):
                if g <= q1:
                    out.append(v0 + (v1 - v0) * (g - q0) / (q1 - q0))
                    break
        return out


def empty_like(queue: InstanceQueueBase) -> InstanceQueueBase:
    """Return an empty sketch of the same kind and quantiles."""
    if isinstance(queue, InstanceQueueProbabilistic):
        return InstanceQueueProbabilistic(
            instance_id=queue.instance_id,
            quantiles=list(queue.quantiles),
            values=[0.0] * len(queue.quantiles),
        )
    return InstanceQueueExpectError(
        instance_id=queue.instance_id, expected_time_ms=0.0, error_margin_ms=0.0
    )


def apply(
    queue: InstanceQueueBase, charge: Charge, sign: int, servers: int = 1
) -> InstanceQueueBase:
    """Add (``sign=1``) or remove (``sign=-1``) a charge.

    Args:
        queue: Current sketch.
        charge: The task's contribution.
        sign: 1 to add, -1 to remove.
        servers: Requests the instance serves at once; the charge is
            divided by it.

    Returns:
        The updated sketch; values never go below zero.
    """
    rc = core.rust()
    if rc is not None:
        return _apply_rust(rc, queue, charge, sign, servers)
    scale = sign / max(1, servers)
    if isinstance(queue, InstanceQueueProbabilistic):
        task = charge.at(queue.quantiles)
        values = []
        floor = 0.0
        for v, t in zip(queue.values, task, strict=True):
            # Stay non-negative and non-decreasing despite rounding.
            floor = max(floor, v + scale * t)
            values.append(floor)
        return InstanceQueueProbabilistic(
            instance_id=queue.instance_id,
            quantiles=queue.quantiles,
            values=values,
        )
    if isinstance(queue, InstanceQueueExpectError):
        # x * x, not x ** 2: libm's pow is not always correctly rounded,
        # and the Rust core must match to the last bit.
        err = scale * charge.error_ms
        var = queue.error_margin_ms * queue.error_margin_ms + sign * (err * err)
        return InstanceQueueExpectError(
            instance_id=queue.instance_id,
            expected_time_ms=max(
                0.0, queue.expected_time_ms + scale * charge.time_ms
            ),
            error_margin_ms=math.sqrt(max(0.0, var)),
        )
    return queue


def _apply_rust(
    rc: Any,
    queue: InstanceQueueBase,
    charge: Charge,
    sign: int,
    servers: int,
) -> InstanceQueueBase:
    """:func:`apply` computed by the Rust core."""
    servers = min(max(1, servers), 2**32 - 1)
    if isinstance(queue, InstanceQueueProbabilistic):
        return InstanceQueueProbabilistic(
            instance_id=queue.instance_id,
            quantiles=queue.quantiles,
            values=rc.apply_probabilistic(
                queue.values,
                queue.quantiles,
                charge.time_ms,
                charge.quantiles,
                sign,
                servers,
            ),
        )
    if isinstance(queue, InstanceQueueExpectError):
        expected, error = rc.apply_expect_error(
            queue.expected_time_ms,
            queue.error_margin_ms,
            charge.time_ms,
            charge.error_ms,
            sign,
            servers,
        )
        return InstanceQueueExpectError(
            instance_id=queue.instance_id,
            expected_time_ms=expected,
            error_margin_ms=error,
        )
    return queue
