"""Auto-optimize loop: Scheduler demand drives the deployment (item 3.3).

Every ``AUTO_OPTIMIZE_INTERVAL`` seconds the loop reads the demand the
Schedulers reported (outstanding work per model and execution time per
instance), runs the swarm optimizer on the current deployment, and applies
the result through the deployment service, whose removals drain first.

Three guards keep it from thrashing:

* It acts only when the new deployment improves the objective by at least
  ``AUTO_OPTIMIZE_CHANGE_THRESHOLD`` (hysteresis).
* The optimizer's change factor ``a`` caps how many machines change in one
  round; at least one may always change.
* A model that is deployed keeps at least ``min_replicas`` instances, so a
  quiet model is never left with nowhere to route.

Two scalers turn reports into each model's share of the machines:

* ``count`` (item 3.3): share proportional to outstanding tasks times mean
  execution time. It sees only work already queued, so a model whose
  replicas keep up looks idle and loses them, and a model called by another
  is scaled only after the calls pile up.
* ``tail`` (item 3.4, the default): share proportional to the instances
  each model needs, in seconds of work per second::

      need[j] = rate[j] * s[j] + work[j] / H
      rate[j] = external[j] + sum_i k[i][j] * out[i]
      out[i]  = min(rate[i] + n[i] / H, replicas[i] / s[i])

  ``external[j]`` is the arrival rate no other model caused, and ``k[i][j]``
  the calls to ``j`` each task of ``i`` makes, so a callee is sized from
  what its callers will finish over the next ``H`` seconds (the interval):
  their arrivals plus their ``n[i]`` outstanding tasks, capped by the rate
  their current replicas serve. A caller's surge reaches its callees in the
  same round, while a caller that is the bottleneck keeps its machines.
  ``work[j]`` is the q0.9 of the outstanding work from the Scheduler's queue
  sketches, so heavy-tailed models get headroom, and never less than the
  outstanding tasks times ``s[j]``. The loop falls back to
  ``count`` while any reporting model sends no arrival rate (an older
  Scheduler).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from loguru import logger

from swarmpilot.planner.core.optimizer_factory import (
    create_optimizer,
    run_optimization,
)
from swarmpilot.planner.services.demand import DemandStore


class _Instance(Protocol):
    model_id: str
    endpoint: str | None


class DeploymentTarget(Protocol):
    """What the loop needs from the deployment service."""

    def get_active_instances(self) -> Sequence[_Instance]:
        """Instances serving traffic now."""
        ...

    def apply_deployment(
        self, target_state: dict[str, int], wait_for_ready: bool = True
    ) -> Any:
        """Reconcile to ``target_state`` (model -> replica count)."""
        ...


@dataclass(frozen=True)
class AutoPlan:
    """One round's decision.

    Attributes:
        target_state: Replica count per model to apply.
        current_state: Replica count per model before the change.
        changes: Machines that change model.
        score_before: Objective of the current deployment.
        score_after: Objective of the new deployment.
        demand: Demand per model the plan was computed from: tasks for
            the count scaler, instances needed for the tail scaler.
        scaler: Which scaler produced the demand.
    """

    target_state: dict[str, int]
    current_state: dict[str, int]
    changes: int
    score_before: float
    score_after: float
    demand: dict[str, float]
    scaler: str = "count"


SCALERS = ("tail", "count")


class AutoOptimizer:
    """Periodically re-plans the deployment from reported demand."""

    def __init__(
        self,
        store: DemandStore,
        service: DeploymentTarget,
        interval: float = 60.0,
        change_threshold: float = 0.05,
        change_factor: float = 0.3,
        min_replicas: int = 1,
        algorithm: str = "integer_programming",
        objective_method: str = "relative_error",
        stale_after: float | None = None,
        scaler: str = "tail",
        horizon: float | None = None,
    ):
        """Initialize the loop.

        Args:
            store: Where Scheduler reports land.
            service: Deployment service that applies a plan.
            interval: Seconds between rounds.
            change_threshold: Minimum objective improvement to act on.
            change_factor: Share of machines that may change per round.
            min_replicas: Replicas a deployed model always keeps.
            algorithm: Optimizer name, as for ``create_optimizer``.
            objective_method: Optimizer objective.
            stale_after: Seconds after which a model's demand report is
                ignored. Defaults to three intervals.
            scaler: ``tail`` or ``count``; see the module docstring.
            horizon: Seconds the tail scaler allows to drain backlog.
                Defaults to the interval.
        """
        if scaler not in SCALERS:
            raise ValueError(f"scaler must be one of {SCALERS}, got {scaler}")
        self._store = store
        self._service = service
        self.interval = interval
        self.change_threshold = change_threshold
        self.change_factor = change_factor
        self.min_replicas = max(0, min_replicas)
        self.algorithm = algorithm
        self.objective_method = objective_method
        self.stale_after = stale_after if stale_after is not None else 3 * interval
        self.scaler = scaler
        self.horizon = horizon if horizon is not None else interval
        self.last_plan: AutoPlan | None = None
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Planning (pure, no I/O)
    # ------------------------------------------------------------------

    def _exec_times(
        self, instances: Sequence[_Instance], models: list[str]
    ) -> dict[str, float]:
        per_model: dict[str, list[float]] = {m: [] for m in models}
        for inst in instances:
            if inst.endpoint:
                t = self._store.execution_time(
                    inst.endpoint,
                    max_age=self.stale_after,
                    model_id=inst.model_id,
                )
                if t is not None:
                    per_model[inst.model_id].append(t)
        known = [t for ts in per_model.values() for t in ts]
        fallback = float(np.mean(known)) if known else 1.0
        return {
            m: float(np.mean(ts)) if ts else fallback for m, ts in per_model.items()
        }

    def plan(self, instances: Sequence[_Instance]) -> AutoPlan | None:
        """Decide this round's deployment, or None to leave it as is.

        Args:
            instances: Active instances.

        Returns:
            The plan to apply, or None when demand is unknown or the gain
            is below the change threshold.
        """
        M = len(instances)
        if M == 0:
            return None
        current = Counter(inst.model_id for inst in instances)
        outstanding = self._store.targets(max_age=self.stale_after)
        rates = self._store.arrival_rates(max_age=self.stale_after)
        known = sorted(set(current) | set(outstanding) | set(rates))
        exec_times = self._exec_times(instances, known)
        scaler = "count"
        if self.scaler == "tail" and rates and set(outstanding) <= set(rates):
            scaler = "tail"
            demand = self._tail_demand(outstanding, rates, exec_times, current)
        else:
            demand = outstanding
        demand = {m: v for m, v in demand.items() if v > 0}
        if not demand:
            return None
        models = sorted(set(current) | set(demand))
        index = {m: j for j, m in enumerate(models)}

        if scaler == "tail":
            # Demand is already in instances; every machine supplies one.
            throughput = np.ones(len(models))
        else:
            throughput = np.array([1.0 / exec_times[m] for m in models])
        B = np.tile(throughput, (M, 1))
        target = np.array([demand.get(m, 0.0) for m in models])
        initial = np.array([index[inst.model_id] for inst in instances])
        optimizer = create_optimizer(
            self.algorithm,
            M=M,
            N=len(models),
            B=B,
            initial=initial,
            a=max(self.change_factor, 1.0 / M),
            target=target,
        )
        before = optimizer.objective_function(initial, self.objective_method)
        deployment, _, _ = run_optimization(
            optimizer, objective_method=self.objective_method
        )
        counts = Counter(models[j] for j in deployment if j >= 0)
        self._keep_min_replicas(counts, current)
        # Score the final counts even if keeping a replica took one more
        # move than the optimizer's cap allows.
        optimizer.max_changes = M
        after = optimizer.objective_function(
            self._as_deployment(counts, instances, index),
            self.objective_method,
        )
        if counts == current or before - after < self.change_threshold:
            return None
        changes = sum(max(0, counts[m] - current[m]) for m in models)
        return AutoPlan(
            target_state={m: counts[m] for m in models if counts[m] > 0},
            current_state=dict(current),
            changes=changes,
            score_before=float(before),
            score_after=float(after),
            demand=demand,
            scaler=scaler,
        )

    def _tail_demand(
        self,
        outstanding: dict[str, float],
        rates: dict[str, float],
        exec_times: dict[str, float],
        current: Counter,
    ) -> dict[str, float]:
        """Instances each model needs, from rates, tail work and calls."""
        k = self._store.call_ratios(max_age=self.stale_after)
        work = self._store.work(max_age=self.stale_after)
        models = list(exec_times)
        into: dict[str, list[tuple[str, float]]] = {m: [] for m in models}
        for caller, callees in k.items():
            for callee, ratio in callees.items():
                if callee in into:
                    into[callee].append((caller, ratio))

        # Arrivals no other model caused.
        external = {
            j: max(
                0.0,
                rates.get(j, 0.0) - sum(r * rates.get(i, 0.0) for i, r in into[j]),
            )
            for j in models
        }
        # Tasks per second each model will finish over the horizon: its
        # arrivals plus its backlog, but no faster than its replicas now
        # serve. Callees see k times that, so a caller's surge reaches its
        # callees in this round, and a caller that is the bottleneck does
        # not hand its machines to callees its backlog cannot feed yet.
        # Chains settle in len(models) passes; cycles converge because the
        # capacity cap bounds every flow.
        cap = {j: current.get(j, 0) / exec_times[j] for j in models}
        out = {j: min(rates.get(j, 0.0), cap[j]) for j in models}
        rate = dict(external)
        for _ in range(4 * len(models) + 16):
            rate = {
                j: external[j] + sum(r * out[i] for i, r in into[j]) for j in models
            }
            nxt = {
                j: min(rate[j] + outstanding.get(j, 0.0) / self.horizon, cap[j])
                for j in models
            }
            settled = all(abs(nxt[j] - out[j]) < 1e-9 for j in models)
            out = nxt
            if settled:
                break

        # The sketches hold no work for tasks routed without a trained
        # predictor, so the backlog is never taken below its mean.
        return {
            j: rate[j] * exec_times[j]
            + max(work.get(j, 0.0), outstanding.get(j, 0.0) * exec_times[j])
            / self.horizon
            for j in models
        }

    def _keep_min_replicas(self, counts: Counter, current: Counter) -> None:
        for m, have in current.items():
            floor = min(self.min_replicas, have)
            while counts[m] < floor:
                donor = max(counts, key=lambda k: counts[k])
                if counts[donor] <= max(self.min_replicas, 1):
                    break
                counts[donor] -= 1
                counts[m] += 1

    @staticmethod
    def _as_deployment(
        counts: Counter,
        instances: Sequence[_Instance],
        index: dict[str, int],
    ) -> np.ndarray:
        """A machine assignment with these counts and the fewest moves."""
        spare = Counter(counts)
        out = np.full(len(instances), -1)
        for i, inst in enumerate(instances):
            if spare[inst.model_id] > 0:
                spare[inst.model_id] -= 1
                out[i] = index[inst.model_id]
        free = [i for i in range(len(instances)) if out[i] == -1]
        for m, n in spare.items():
            for _ in range(n):
                out[free.pop()] = index[m]
        return out

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    async def run_once(self) -> AutoPlan | None:
        """Plan and, if worthwhile, apply one round.

        Returns:
            The plan applied, or None if nothing changed.
        """
        instances = list(self._service.get_active_instances())
        plan = self.plan(instances)
        if plan is None:
            return None
        logger.info(
            f"[AUTO_OPTIMIZE] {plan.scaler} demand={plan.demand} "
            f"{plan.current_state} -> {plan.target_state} "
            f"score {plan.score_before:.3f} -> {plan.score_after:.3f}"
        )
        # Reconciling blocks (deploys, drains); keep it off the loop.
        result = await asyncio.to_thread(
            self._service.apply_deployment, plan.target_state
        )
        self.last_plan = plan
        self._record(plan, getattr(result, "success", True))
        return plan

    def _record(self, plan: AutoPlan, success: bool) -> None:
        from swarmpilot.planner.instance_timeline_tracker import (
            get_timeline_tracker,
        )

        models = sorted(plan.target_state)
        with suppress(Exception):
            get_timeline_tracker().record_migration(
                event_type="auto_optimize",
                instance_counts=plan.target_state,
                changes_count=plan.changes,
                success=success,
                target_distribution=[plan.demand.get(m, 0.0) for m in models],
                score=plan.score_after,
            )

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.opt(exception=True).error(f"[AUTO_OPTIMIZE] round failed: {e}")

    def start(self) -> None:
        """Start the background loop on the running event loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            logger.info(
                f"Auto-optimize loop started (interval={self.interval}s, "
                f"threshold={self.change_threshold}, "
                f"change_factor={self.change_factor}, "
                f"scaler={self.scaler})"
            )

    async def stop(self) -> None:
        """Stop the background loop."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
