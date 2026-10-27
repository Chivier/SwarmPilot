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

Without ``elastic`` the loop only re-divides the instances that exist. With
it (tail scaler only) it also sizes the pool. ``need[j]`` is in tasks in
flight, so the pool wants ``ceil(sum(need) / target_concurrency)``
instances, the target concurrency per instance that autoscalers such as
Knative use:

* It grows at once, into GPUs no instance holds (``spare``), up to
  ``max_instances``.
* It shrinks only after ``scale_down_rounds`` rounds in a row that want
  fewer (then counts again from zero), by at most the change factor's
  share per round, taking instances from the models with the most
  surplus, never below ``min_instances`` or a model's ``min_replicas``.
  Fresh reports of no traffic shrink it too; a pool above
  ``max_instances`` shrinks without waiting.
"""

from __future__ import annotations

import asyncio
import math
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

    # Elastic pools also call ``spare_capacity() -> int``: instances that
    # still fit on free GPUs. A service without it never grows the pool.


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
        size_before: Instances before the change.
        size_after: Instances after it; differs only in an elastic pool.
    """

    target_state: dict[str, int]
    current_state: dict[str, int]
    changes: int
    score_before: float
    score_after: float
    demand: dict[str, float]
    scaler: str = "count"
    size_before: int = 0
    size_after: int = 0


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
        elastic: bool = False,
        target_concurrency: float = 64.0,
        min_instances: int = 1,
        max_instances: int = 0,
        scale_down_rounds: int = 3,
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
            elastic: Also size the pool (tail scaler only); see the
                module docstring.
            target_concurrency: Tasks in flight one instance should
                carry in an elastic pool.
            min_instances: Smallest elastic pool.
            max_instances: Largest elastic pool; 0 for no limit but the
                free GPUs.
            scale_down_rounds: Rounds in a row that must want a smaller
                pool before it shrinks.
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
        self.stale_after = (
            stale_after if stale_after is not None else 3 * interval
        )
        self.scaler = scaler
        self.horizon = horizon if horizon is not None else interval
        if target_concurrency <= 0:
            raise ValueError(
                f"target_concurrency must be positive, got {target_concurrency}"
            )
        self.elastic = elastic
        self.target_concurrency = target_concurrency
        self.min_instances = max(1, min_instances)
        self.max_instances = max(0, max_instances)
        self.scale_down_rounds = max(1, scale_down_rounds)
        self._low_rounds = 0
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
            m: float(np.mean(ts)) if ts else fallback
            for m, ts in per_model.items()
        }

    def plan(
        self, instances: Sequence[_Instance], spare: int = 0
    ) -> AutoPlan | None:
        """Decide this round's deployment, or None to leave it as is.

        Args:
            instances: Active instances.
            spare: Instances that still fit on free GPUs; an elastic pool
                grows into them.

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
        # Fresh reports of no traffic at all still size an elastic pool
        # (down); stale ones fall back to the count scaler and change
        # nothing.
        elastic = self.elastic and scaler == "tail"
        if not demand and not elastic:
            return None
        models = sorted(set(current) | set(demand))
        index = {m: j for j, m in enumerate(models)}
        size = M
        if elastic:
            size = self._pool_size(M, demand, current, spare)
        if not demand and size >= M:
            return None

        if scaler == "tail":
            # Demand is already in instances; every machine supplies one.
            throughput = np.ones(len(models))
        else:
            throughput = np.array([1.0 / exec_times[m] for m in models])
        target = np.array([demand.get(m, 0.0) for m in models])

        if size < M:
            counts = self._shrink(current, demand, size)
        else:
            # New machines start undeployed. Integer programming counts
            # placing one as a change, so it gets that many moves on top
            # of what the change factor allows; simulated annealing places
            # them before it starts counting.
            initial = np.array(
                [index[inst.model_id] for inst in instances] + [-1] * (size - M)
            )
            a = max(self.change_factor, 1.0 / M)
            if size > M:
                moves = a * M
                if self.algorithm == "integer_programming":
                    moves += size - M
                a = min(1.0, (moves + 1e-9) / size)
            optimizer = create_optimizer(
                self.algorithm,
                M=size,
                N=len(models),
                B=np.tile(throughput, (size, 1)),
                initial=initial,
                a=a,
                target=target,
            )
            deployment, _, _ = run_optimization(
                optimizer, objective_method=self.objective_method
            )
            counts = Counter(models[j] for j in deployment if j >= 0)
            self._keep_min_replicas(counts, current)
        if target.any():
            before = self._score(current, models, throughput, target)
            after = self._score(counts, models, throughput, target)
        else:
            before = after = 0.0
        if counts == current:
            return None
        if size == M and before - after < self.change_threshold:
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
            size_before=M,
            size_after=sum(counts.values()),
        )

    def _pool_size(
        self,
        M: int,
        demand: dict[str, float],
        current: Counter,
        spare: int,
    ) -> int:
        """Instances an elastic pool should have after this round."""
        floor = max(
            self.min_instances,
            sum(min(self.min_replicas, n) for n in current.values()),
        )
        ceiling = M + max(0, spare)
        if self.max_instances:
            ceiling = min(ceiling, self.max_instances)
        wanted = math.ceil(sum(demand.values()) / self.target_concurrency)
        wanted = max(floor, min(wanted, ceiling))
        if wanted >= M:
            self._low_rounds = 0
            return wanted
        # A pool above max_instances shrinks at once; one that is merely
        # quiet waits, then counts its quiet rounds again from zero.
        over_cap = bool(self.max_instances) and self.max_instances < M
        self._low_rounds += 1
        if self._low_rounds < self.scale_down_rounds and not over_cap:
            return M
        self._low_rounds = 0
        step = max(1, int(self.change_factor * M))
        return max(wanted, M - step)

    def _shrink(
        self, current: Counter, demand: dict[str, float], size: int
    ) -> Counter:
        """Drop instances from the models with the most surplus."""
        counts = Counter(current)
        total = sum(demand.values())
        floor = {m: min(self.min_replicas, n) for m, n in current.items()}
        while sum(counts.values()) > size:
            share = {
                m: counts[m]
                - (size * demand.get(m, 0.0) / total if total else 0)
                for m in counts
                if counts[m] > max(floor.get(m, 0), 0)
            }
            if not share:
                break
            victim = max(share, key=lambda m: share[m])
            counts[victim] -= 1
        return +counts

    def _score(
        self,
        counts: Counter,
        models: list[str],
        throughput: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Objective of a deployment with these replica counts."""
        deployment = np.array(
            [j for j, m in enumerate(models) for _ in range(counts[m])]
        )
        if len(deployment) == 0:
            return float("inf")
        optimizer = create_optimizer(
            self.algorithm,
            M=len(deployment),
            N=len(models),
            B=np.tile(throughput, (len(deployment), 1)),
            initial=deployment,
            a=1.0,
            target=target,
        )
        return optimizer.objective_function(deployment, self.objective_method)

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
                rates.get(j, 0.0)
                - sum(r * rates.get(i, 0.0) for i, r in into[j]),
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
                j: external[j] + sum(r * out[i] for i, r in into[j])
                for j in models
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

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    async def run_once(self) -> AutoPlan | None:
        """Plan and, if worthwhile, apply one round.

        Returns:
            The plan applied, or None if nothing changed.
        """
        instances = list(self._service.get_active_instances())
        spare = 0
        if self.elastic and hasattr(self._service, "spare_capacity"):
            spare = await asyncio.to_thread(self._service.spare_capacity)
        plan = self.plan(instances, spare=spare)
        if plan is None:
            return None
        logger.info(
            f"[AUTO_OPTIMIZE] {plan.scaler} demand={plan.demand} "
            f"{plan.current_state} -> {plan.target_state} "
            f"score {plan.score_before:.3f} -> {plan.score_after:.3f} "
            f"size {plan.size_before} -> {plan.size_after}"
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
                logger.opt(exception=True).error(
                    f"[AUTO_OPTIMIZE] round failed: {e}"
                )

    def start(self) -> None:
        """Start the background loop on the running event loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            logger.info(
                f"Auto-optimize loop started (interval={self.interval}s, "
                f"threshold={self.change_threshold}, "
                f"change_factor={self.change_factor}, "
                f"scaler={self.scaler}, elastic={self.elastic})"
            )

    async def stop(self) -> None:
        """Stop the background loop."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
