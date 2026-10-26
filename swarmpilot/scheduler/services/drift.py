"""Predictor drift detection (plan item 3.1).

Every completed task that was routed with a prediction is scored against
its actual runtime. The detector keeps, per (model, platform, prompt
group), a sliding window of two tail signals at ``tau`` (0.9):

* the relative pinball loss ``rho_tau(y - q) / q``, compared with the
  baseline it settled at after the model was installed, and
* the miss rate ``P(y > q)``, compared with its target ``1 - tau``.

The baseline is the window's mean loss once ``min_samples`` tasks are in.
After that, a group drifts when its window's mean loss exceeds
``loss_ratio`` times the baseline, or its miss rate exceeds ``miss_ratio``
times ``1 - tau``. The training client then
retrains that (model, platform) on recent samples only.

Prompt groups are log2 buckets of the adapter's first feature
(``prompt_length`` for LLMs, ``steps`` for diffusion), so a shift that
hits only long prompts is not averaged away by short ones.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

#: z-score of the 0.9 quantile, for expect-error predictions.
_Z90 = 1.2815515655446004


def tail_estimate(
    tau: float,
    quantiles: Mapping[float, float] | None,
    expected_ms: float | None,
    error_ms: float | None,
) -> float | None:
    """The predicted ``tau`` quantile of a task's runtime, if known.

    Quantile predictions are interpolated; an expect-error prediction is
    read as a normal with the error as its standard deviation.
    """
    if quantiles:
        qs = sorted((float(q), float(v)) for q, v in quantiles.items())
        if tau <= qs[0][0]:
            return qs[0][1]
        for (q0, v0), (q1, v1) in pairwise(qs):
            if tau <= q1:
                return v0 + (v1 - v0) * (tau - q0) / (q1 - q0)
        return qs[-1][1]
    if expected_ms is None:
        return None
    z = _Z90 if tau == 0.9 else 0.0
    return expected_ms + z * (error_ms or 0.0)


def prompt_group(features: Mapping[str, Any]) -> int:
    """Log2 bucket of the first numeric feature; 0 when there is none."""
    for value in features.values():
        if isinstance(value, int | float) and not isinstance(value, bool):
            return int(math.log2(1 + max(0.0, float(value))))
    return 0


@dataclass
class _Group:
    losses: deque[float]
    misses: deque[float]
    baseline: float | None = None
    since_reset: int = 0
    drifted: bool = False


@dataclass
class DriftDetector:
    """Sliding-window tail-loss drift detector.

    Attributes:
        tau: Tail quantile scored.
        window: Scored tasks kept per group.
        min_samples: Scored tasks needed before a group can drift (and
            before its baseline is fixed).
        loss_ratio: Drift when recent loss exceeds this times the baseline.
        miss_ratio: Drift when the miss rate exceeds this times ``1 - tau``
            and its binomial z-score exceeds ``miss_z``.
    """

    tau: float = 0.9
    window: int = 200
    min_samples: int = 50
    loss_ratio: float = 2.0
    miss_ratio: float = 3.0
    miss_z: float = 4.0
    _groups: dict[tuple[str, str, int], _Group] = field(default_factory=dict)

    def observe(
        self,
        key: tuple[str, str],
        features: Mapping[str, Any],
        predicted_tail_ms: float,
        actual_ms: float,
    ) -> bool:
        """Score one task. Returns True when its group has just drifted.

        Args:
            key: (model_id, platform key) the prediction came from.
            features: Task features, for the prompt group.
            predicted_tail_ms: Predicted ``tau`` quantile.
            actual_ms: Measured runtime.
        """
        if predicted_tail_ms <= 0 or not math.isfinite(predicted_tail_ms):
            return False
        gkey = (key[0], key[1], prompt_group(features))
        g = self._groups.get(gkey)
        if g is None:
            g = _Group(deque(maxlen=self.window), deque(maxlen=self.window))
            self._groups[gkey] = g
        diff = actual_ms - predicted_tail_ms
        loss = max(self.tau * diff, (self.tau - 1) * diff) / predicted_tail_ms
        g.losses.append(loss)
        g.misses.append(1.0 if diff > 0 else 0.0)
        g.since_reset += 1
        if g.drifted or g.since_reset < self.min_samples:
            return False
        # Whole-window means: 200 tasks keep a 0.1 miss rate's noise far
        # below the 0.2 threshold, and a real shift still crosses it after
        # a few dozen tasks.
        mean_loss = sum(g.losses) / len(g.losses)
        if g.baseline is None:
            g.baseline = mean_loss
            return False
        n = len(g.misses)
        p = 1 - self.tau
        misses = sum(g.misses)
        # The miss rate must be both well above target and improbable for
        # a calibrated model (binomial z > miss_z), so a short window after
        # a reset does not fire on noise.
        z = (misses - n * p) / math.sqrt(n * p * (1 - p))
        if mean_loss > self.loss_ratio * max(g.baseline, 1e-9) or (
            misses / n > self.miss_ratio * p and z > self.miss_z
        ):
            g.drifted = True
            return True
        return False

    def reset(self, key: tuple[str, str]) -> None:
        """Start fresh for every group of ``key`` (after a retrain)."""
        for gkey in [k for k in self._groups if k[:2] == key]:
            del self._groups[gkey]

    def rearm(self, key: tuple[str, str]) -> None:
        """Let ``key`` drift again after a retrain that installed nothing.

        Its baseline stays, and it needs ``min_samples`` new tasks first.
        """
        for gkey, g in self._groups.items():
            if gkey[:2] == key:
                g.drifted = False
                g.since_reset = 0

    def drifted(self) -> set[tuple[str, str]]:
        """(model, platform) keys with a group that has drifted."""
        return {k[:2] for k, g in self._groups.items() if g.drifted}

    def snapshot(self) -> list[dict[str, Any]]:
        """Per-group state, for status endpoints and tests."""
        out = []
        for (model, platform, group), g in self._groups.items():
            n = len(g.losses)
            out.append(
                {
                    "model_id": model,
                    "platform": platform,
                    "prompt_group": group,
                    "samples": n,
                    "loss": sum(g.losses) / n if n else None,
                    "baseline": g.baseline,
                    "miss_rate": sum(g.misses) / n if n else None,
                    "drifted": g.drifted,
                }
            )
        return out
