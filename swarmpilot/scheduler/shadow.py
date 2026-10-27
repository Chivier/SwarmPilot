"""Shadow mode: compare the gateway's routing with the Python proxy's.

With ``SCHEDULER_GATEWAY_SHADOW_URL`` set, the Python proxy keeps serving
every request and also mirrors each routing decision to a
``swarmpilot-gateway`` started with ``--shadow``. The gateway routes the
same request (translation, features, predictions, strategy) on its own
state but sends nothing to an engine: it answers with the instance it
would have chosen, and the two choices are compared.

To make every strategy comparable decision by decision, the gateway is
given what the Python decision depended on beyond the shared state:

- the random draws the Python strategy made (``probabilistic`` samples,
  ``po2`` pairs, ``random`` picks), replayed instead of the gateway's own
  generator, and
- the order of events: decisions and charge releases are sent in the
  order they happened, on one ordered channel, so the gateway's queue
  sketches take and lose the same charges as the Scheduler's.

What can still differ is what the shadow measures: predictions, sketches,
strategy code, and state the gateway has not received yet (instances and
models arrive through the control channel, up to 0.2 s late).
``GET /v1/gateway/shadow`` reports the result.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any

import httpx
import numpy as np
from loguru import logger

#: The draws of the decision being made in this context, or None.
_draws: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "swarmpilot_shadow_draws", default=None
)


class RecordingRng:
    """A numpy ``Generator`` that records draws made while recording.

    Strategies draw through ``random``, ``integers`` and ``choice``; each
    draw made inside :meth:`ShadowMirror.recording` is appended to that
    decision's list in the shape the gateway replays. Everything else is
    the wrapped generator's.
    """

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def random(self, size: Any = None, *args: Any, **kwargs: Any) -> Any:
        """``Generator.random``, recorded."""
        out = self._rng.random(size, *args, **kwargs)
        rec = _draws.get()
        if rec is not None:
            rec.append({"random": np.atleast_1d(out).tolist()})
        return out

    def integers(self, *args: Any, **kwargs: Any) -> Any:
        """``Generator.integers``, recorded."""
        out = self._rng.integers(*args, **kwargs)
        rec = _draws.get()
        if rec is not None:
            rec.append({"integers": int(out)})
        return out

    def choice(self, a: Any, *args: Any, **kwargs: Any) -> Any:
        """``Generator.choice``, recorded (pairs only)."""
        out = self._rng.choice(a, *args, **kwargs)
        rec = _draws.get()
        if rec is not None:
            pair = np.atleast_1d(out).tolist()
            rec.append(
                {"choice2": pair} if len(pair) == 2 else {"other": "choice"}
            )
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rng, name)


def _percentiles(xs: deque[float]) -> dict[str, float | None]:
    if not xs:
        return {"p50": None, "p99": None}
    a = np.fromiter(xs, dtype=float)
    return {
        "p50": round(float(np.percentile(a, 50)), 1),
        "p99": round(float(np.percentile(a, 99)), 1),
    }


class ShadowMirror:
    """Mirrors decisions to a shadow gateway and compares its answers.

    Events go out in order, in batches, from one background task. When the
    queue is full (the gateway is slower than the traffic, or down) events
    are dropped and counted: the gateway's sketches then miss charges, so
    later comparisons are less exact until those tasks have finished.
    """

    def __init__(
        self,
        url: str,
        *,
        max_queue: int = 10_000,
        batch: int = 256,
        keep: int = 10_000,
        recent: int = 50,
    ):
        self.url = url.rstrip("/") + "/_gateway/shadow"
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(max_queue)
        self._batch = batch
        self._mirrored: set[str] = set()
        self._task: asyncio.Task | None = None
        self.decisions = 0
        self.matched = 0
        self.mismatched = 0
        self.errors = 0
        self.dropped = 0
        self.draws_diverged = 0
        self.by_strategy: dict[str, dict[str, int]] = {}
        self._python_us: deque[float] = deque(maxlen=keep)
        self._gateway_us: deque[float] = deque(maxlen=keep)
        self.recent_mismatches: deque[dict[str, Any]] = deque(maxlen=recent)
        self.last_error: str | None = None

    def start(self) -> None:
        """Start sending events (call from the running event loop)."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="shadow")

    async def stop(self) -> None:
        """Stop sending; events still queued are dropped."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    @contextmanager
    def recording(self) -> Iterator[list[dict[str, Any]]]:
        """Record the draws of the decision made inside the block."""
        draws: list[dict[str, Any]] = []
        token = _draws.set(draws)
        try:
            yield draws
        finally:
            _draws.reset(token)

    def decided(
        self,
        *,
        task_id: str,
        path: str,
        body: Any,
        instance_id: str,
        draws: list[dict[str, Any]],
        decide_us: float,
        strategy: str,
    ) -> None:
        """Mirror the Python proxy's decision for a request.

        Args:
            task_id: The task; its charge is released under this id.
            path: The path the client called (before translation).
            body: The body the client sent (before translation).
            instance_id: The instance the Python proxy chose.
            draws: The draws the decision made (from :meth:`recording`).
            decide_us: How long the Python decision took.
            strategy: The strategy's name, for the per-strategy counts.
        """
        if self._put(
            {
                "op": "decide",
                "task_id": task_id,
                "path": path,
                "body": body,
                "draws": draws,
                "python": instance_id,
                "python_us": decide_us,
                "strategy": strategy,
            }
        ):
            self._mirrored.add(task_id)

    def released(self, task_id: str) -> None:
        """The Scheduler released a mirrored task's charge."""
        if task_id in self._mirrored:
            self._mirrored.discard(task_id)
            self._put({"op": "release", "task_id": task_id})

    def _put(self, event: dict[str, Any]) -> bool:
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            return False

    async def _run(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                events = [await self._queue.get()]
                while len(events) < self._batch and not self._queue.empty():
                    events.append(self._queue.get_nowait())
                try:
                    r = await client.post(
                        self.url,
                        json={
                            "events": [
                                {
                                    k: v
                                    for k, v in ev.items()
                                    if k not in ("python", "python_us")
                                }
                                for ev in events
                            ]
                        },
                    )
                    r.raise_for_status()
                    results = r.json()["results"]
                except Exception as e:  # gateway down, refused, bad answer
                    self.errors += sum(ev["op"] == "decide" for ev in events)
                    self.last_error = f"{type(e).__name__}: {e}"
                    logger.warning(f"[SHADOW] Mirroring failed: {e}")
                    continue
                for ev, res in zip(events, results, strict=False):
                    if ev["op"] == "decide":
                        self._compare(ev, res)

    def _compare(self, ev: dict[str, Any], res: dict[str, Any]) -> None:
        self.decisions += 1
        self._python_us.append(ev["python_us"])
        counts = self.by_strategy.setdefault(
            ev["strategy"], {"decisions": 0, "mismatched": 0}
        )
        counts["decisions"] += 1
        if "error" in res:
            self.errors += 1
            self.last_error = res["error"]
            return
        self._gateway_us.append(res.get("decide_us", 0.0))
        if res.get("draws_diverged"):
            self.draws_diverged += 1
        if res.get("instance_id") == ev["python"]:
            self.matched += 1
            return
        self.mismatched += 1
        counts["mismatched"] += 1
        self.recent_mismatches.append(
            {
                "task_id": ev["task_id"],
                "path": ev["path"],
                "strategy": ev["strategy"],
                "python": ev["python"],
                "gateway": res.get("instance_id"),
                "at": time.time(),
            }
        )
        logger.debug(
            f"[SHADOW] {ev['task_id']}: Python chose {ev['python']}, "
            f"the gateway {res.get('instance_id')}"
        )

    def summary(self) -> dict[str, Any]:
        """The comparison so far, for ``GET /v1/gateway/shadow``."""
        compared = self.matched + self.mismatched
        return {
            "enabled": True,
            "url": self.url,
            "decisions": self.decisions,
            "compared": compared,
            "matched": self.matched,
            "mismatched": self.mismatched,
            "mismatch_rate": (
                round(self.mismatched / compared, 6) if compared else None
            ),
            "errors": self.errors,
            "dropped": self.dropped,
            "draws_diverged": self.draws_diverged,
            "queued": self._queue.qsize(),
            "by_strategy": self.by_strategy,
            "decide_us": {
                "python": _percentiles(self._python_us),
                "gateway": _percentiles(self._gateway_us),
            },
            "recent_mismatches": list(self.recent_mismatches),
            "last_error": self.last_error,
        }
