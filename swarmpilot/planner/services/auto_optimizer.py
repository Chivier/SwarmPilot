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

With ``hardware_aware`` (tail scaler only) machines are no longer equal.
The loop learns each kind of hardware's speed from the execution times the
Schedulers report (``HardwareSpeeds``; the fastest kind is 1) and counts an
instance as that much capacity:

* The optimizer sees each machine's speed, so it decides which machines
  move, and the plan names them: the deployment service removes those
  instances and starts their replacements on the same workers, instead of
  removing the oldest instances and placing the new ones anywhere.
* An elastic pool is sized in capacity: it wants
  ``sum(need) / target_concurrency`` units of the fastest hardware, grows
  into the fastest free GPUs first, and shrinks by its slowest instances,
  taken from the models with the most surplus capacity.
* When the pool cannot carry all the need, arrivals (``rate * s``) are
  served before backlog, each split max-min fairly, so neither a model
  whose backlog is already timing out nor one too heavy to serve can take
  the machines a lighter model needs to keep up.
* A move costs a drain and a cold start, so each machine that moves must
  improve the objective by ``change_threshold`` on its own, an instance
  that started serving less than ``min_age`` ago does not move, and each
  model keeps its floor on machines that do not move this round (removals
  drain before replacements start).
* In an elastic pool, a model that takes several GPUs or one kind of
  hardware can find no free place while smaller models hold the GPUs it
  needs (in pieces, or whole workers when each instance takes one). When the model furthest below its share fits on no free
  GPUs, the loop makes room (``_make_room``): it removes instances of
  models that stay less short than it from one worker, and starts the
  model there.

Before every deployed model has reported once, the loop waits (up to
``stale_after`` seconds from its first round), so a model whose Scheduler
has not reported yet is not read as idle.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter, deque
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from loguru import logger

from swarmpilot.planner.core.optimizer_factory import (
    create_optimizer,
    run_optimization,
)
from swarmpilot.planner.pylet.hardware import matches
from swarmpilot.planner.services.demand import DemandStore
from swarmpilot.planner.services.hardware_speed import HardwareSpeeds


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
    # Hardware-aware pools call ``free_slots() -> [(worker, hardware)]``
    # (one entry per free GPU) instead, and pass ``remove`` (PyLet IDs)
    # and ``place`` (workers per model) to ``apply_deployment``.
    # ``model_requirements(model) -> (gpus, hardware pattern or None)``,
    # when present, says what one instance of a model needs; without it
    # every model takes one GPU of any kind.


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
        remove: PyLet IDs of the instances to remove (hardware-aware).
        place: Workers for each model's new instances (hardware-aware);
            None lets PyLet choose.
        speeds: Hardware speeds the plan used (hardware-aware).
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
    remove: tuple[str, ...] = ()
    place: dict[str, tuple[str | None, ...]] = field(default_factory=dict)
    speeds: dict[str, float] = field(default_factory=dict)


SCALERS = ("tail", "count")
_DEFAULT_HW = "gpu"


def _hardware(inst: _Instance) -> str:
    return getattr(inst, "hardware", None) or _DEFAULT_HW


Requirement = tuple[int, str | None]
_ANY: Requirement = (1, None)


def _fits(req: Requirement, gpus: int, hardware: str | None) -> bool:
    """Whether ``gpus`` GPUs of ``hardware`` hold one instance of ``req``."""
    need, pattern = req
    if gpus < need:
        return False
    return pattern is None or (
        hardware is not None and matches(hardware, pattern)
    )


def _undo_needless_moves(
    deployment: list[int],
    initial: list[int],
    speed: np.ndarray,
    ok=lambda i, j: True,
) -> None:
    """Keep machines where they are when an equal one could move instead.

    The optimizer sees capacity only, so it may move machine ``i`` off
    model ``m`` and give ``m`` an equally fast machine ``k`` in return.
    Swapping the two assignments keeps every model's capacity and saves
    a redeploy, if machine ``k`` can run what ``i`` was given
    (``ok(k, model index)``).
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(deployment)):
            m = initial[i]
            if m < 0 or deployment[i] == m:
                continue
            for k in range(len(deployment)):
                if (
                    k != i
                    and deployment[k] == m
                    and initial[k] != m
                    and abs(speed[k] - speed[i]) < 1e-9
                    and (deployment[i] < 0 or ok(k, deployment[i]))
                ):
                    deployment[i], deployment[k] = m, deployment[i]
                    changed = True
                    break


def _water_fill(need: dict[str, float], supply: float) -> dict[str, float]:
    """Max-min fair split of ``supply`` among ``need``."""
    out: dict[str, float] = {}
    left = supply
    items = sorted(need.items(), key=lambda kv: kv[1])
    for k, (m, v) in enumerate(items):
        give = min(v, left / (len(items) - k))
        out[m] = float(give)
        left -= give
    return out


def _serving_since(inst: _Instance) -> float:
    """When an instance started serving, or was created if not known."""
    since = getattr(inst, "active_since", None)
    if since is None:
        since = getattr(inst, "created_at", None)
    return since or 0.0


def _share_error(
    deployment: Sequence[int], speed: np.ndarray, target: np.ndarray
) -> float:
    """Relative-error objective of a deployment of weighted machines."""
    capacity = np.zeros(len(target))
    for i, j in enumerate(deployment):
        if j >= 0:
            capacity[j] += speed[i]
    if capacity.sum() <= 0 or target.sum() <= 0:
        return 0.0 if target.sum() <= 0 else float("inf")
    return float(
        np.abs(capacity / capacity.sum() - target / target.sum()).sum()
    )


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
        hardware_aware: bool = False,
        min_age: float | None = None,
        demand_hold: float | None = None,
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
            hardware_aware: Weigh machines by learned hardware speed
                (tail scaler only); see the module docstring.
            min_age: Seconds a hardware-aware loop leaves an instance
                where it is after it starts serving. Defaults to four
                intervals.
            demand_hold: Seconds a hardware-aware loop keeps planning
                for a model's highest recent demand after it falls.
                Defaults to two intervals; 0 turns it off.
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
        self.hardware_aware = hardware_aware
        self.min_age = min_age if min_age is not None else 4 * interval
        self.demand_hold = (
            demand_hold if demand_hold is not None else 2 * interval
        )
        self._recent_demand: deque[
            tuple[float, dict[str, float], dict[str, float]]
        ] = deque()
        self.speeds = HardwareSpeeds(alpha=0.3)
        self._low_rounds = 0
        self._first_plan_at: float | None = None
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
        self,
        instances: Sequence[_Instance],
        spare: int = 0,
        slots: Sequence[tuple[str | None, str | None]] | None = None,
    ) -> AutoPlan | None:
        """Decide this round's deployment, or None to leave it as is.

        Args:
            instances: Active instances.
            spare: Instances that still fit on free GPUs; an elastic pool
                grows into them.
            slots: (worker, hardware) of each free place, for a
                hardware-aware pool; defaults to ``spare`` unknown places.

        Returns:
            The plan to apply, or None when demand is unknown or the gain
            is below the change threshold.
        """
        M = len(instances)
        if M == 0:
            return None
        current = Counter(inst.model_id for inst in instances)
        if self._waiting_for_reports(current):
            return None
        outstanding = self._store.targets(max_age=self.stale_after)
        rates = self._store.arrival_rates(max_age=self.stale_after)
        known = sorted(set(current) | set(outstanding) | set(rates))
        exec_times = self._exec_times(instances, known)
        scaler = "count"
        steady: dict[str, float] = {}
        if self.scaler == "tail" and rates and set(outstanding) <= set(rates):
            scaler = "tail"
            steady, backlog = self._tail_parts(
                outstanding, rates, exec_times, current
            )
            demand = {m: steady[m] + backlog[m] for m in steady}
        else:
            demand = outstanding
        demand = {m: v for m, v in demand.items() if v > 0}
        if self.hardware_aware and scaler == "tail":
            if slots is None:
                slots = [(None, None)] * max(0, spare)
            demand, steady = self._held(demand, steady, set(current))
            return self._plan_by_speed(instances, slots, demand, steady)
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
            # Machines are counted, not placed, so one that runs a model
            # changes only to a model of the same shape (GPUs and hardware
            # pattern). New places are free GPUs counted in the default
            # GPU count; any model may take one, and what does not fit is
            # reported by the deploy (the hardware-aware loop places by
            # shape).
            reqs = self._requirements(models)
            shape = [reqs[models[j]] if j >= 0 else None for j in initial]
            B = np.tile(throughput, (size, 1))
            for i in range(size):
                for j, m in enumerate(models):
                    if (
                        shape[i] is not None
                        and j != initial[i]
                        and reqs[m] != shape[i]
                    ):
                        B[i, j] = 0
            optimizer = create_optimizer(
                self.algorithm,
                M=size,
                N=len(models),
                B=B,
                initial=initial,
                a=a,
                target=target,
            )
            deployment, _, _ = run_optimization(
                optimizer, objective_method=self.objective_method
            )
            deployment = [
                (
                    int(j)
                    if j == initial[i]
                    or shape[i] is None
                    or (j >= 0 and reqs[models[j]] == shape[i])
                    else int(initial[i])
                )
                for i, j in enumerate(deployment)
            ]
            counts = Counter(models[j] for j in deployment if j >= 0)
            self._keep_min_replicas(counts, current, reqs)
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
        steady, backlog = self._tail_parts(
            outstanding, rates, exec_times, current
        )
        return {j: steady[j] + backlog[j] for j in steady}

    def _tail_parts(
        self,
        outstanding: dict[str, float],
        rates: dict[str, float],
        exec_times: dict[str, float],
        current: Counter,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """The tail need split into arrivals (``rate * s``) and backlog."""
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
        steady = {j: rate[j] * exec_times[j] for j in models}
        backlog = {
            j: max(work.get(j, 0.0), outstanding.get(j, 0.0) * exec_times[j])
            / self.horizon
            for j in models
        }
        return steady, backlog

    def _held(
        self,
        demand: dict[str, float],
        steady: dict[str, float],
        deployed: set[str],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Each model's highest demand over the last ``demand_hold`` s.

        Bursty traffic swings a model's backlog from round to round, and
        a burst period near the interval aliases with the rounds: every
        round then sees one model up and the other down and trades
        machines back, each trade a drain and a cold start. Demand rises
        at once but falls only after it has stayed down for the hold.
        Only models that are deployed or report demand now are held.
        """
        if self.demand_hold <= 0:
            return demand, steady
        now = time.time()
        recent = self._recent_demand
        recent.append((now, dict(demand), dict(steady)))
        while now - recent[0][0] > self.demand_hold + 1e-6:
            recent.popleft()
        live = deployed | set(demand)
        held = {m: max(d.get(m, 0.0) for _, d, _ in recent) for m in live}
        held_steady = {
            m: max(s.get(m, 0.0) for _, _, s in recent) for m in live
        }
        return (
            {m: v for m, v in held.items() if v > 0},
            {m: v for m, v in held_steady.items() if m in steady or v > 0},
        )

    def _waiting_for_reports(self, current: Counter) -> bool:
        """True while a deployed model has never reported, at first."""
        now = time.monotonic()
        if self._first_plan_at is None:
            self._first_plan_at = now
        if now - self._first_plan_at >= self.stale_after:
            return False
        return not set(current) <= set(self._store.targets())

    # ------------------------------------------------------------------
    # Hardware-aware planning
    # ------------------------------------------------------------------

    def _learn_speeds(self, instances: Sequence[_Instance]) -> None:
        samples = []
        for inst in instances:
            if not inst.endpoint:
                continue
            t = self._store.execution_time(
                inst.endpoint, max_age=self.stale_after, model_id=inst.model_id
            )
            if t is not None:
                samples.append((inst.model_id, _hardware(inst), t))
        self.speeds.observe(samples)

    def _speed_of(self, hardware: str | None, default: float) -> float:
        speed = self.speeds.speed(hardware or _DEFAULT_HW)
        return default if speed is None else speed

    def _requirements(self, models) -> dict[str, Requirement]:
        """GPUs per instance and hardware pattern of each model."""
        lookup = getattr(self._service, "model_requirements", None)
        out: dict[str, Requirement] = {}
        for m in models:
            req = None
            if lookup is not None:
                try:
                    req = lookup(m)
                except Exception as e:
                    logger.debug(f"No requirements for {m}: {e}")
            out[m] = (max(1, int(req[0])), req[1] or None) if req else _ANY
        return out

    def _plan_by_speed(
        self,
        instances: Sequence[_Instance],
        slots: Sequence[tuple[str | None, str | None]],
        demand: dict[str, float],
        steady: dict[str, float],
    ) -> AutoPlan | None:
        """Plan with each machine weighted by its hardware's speed.

        Models may need different GPU counts and kinds of hardware
        (``model_requirements``). A machine changes model only to one
        that takes as many GPUs and runs on its hardware, and the pool
        grows each model into free GPUs that hold one of its instances.
        """
        M = len(instances)
        current = Counter(inst.model_id for inst in instances)
        models = sorted(set(current) | set(demand))
        index = {m: j for j, m in enumerate(models)}
        reqs = self._requirements(models)
        self._learn_speeds(instances)
        known = [
            v
            for v in (self.speeds.speed(_hardware(i)) for i in instances)
            if v is not None
        ]
        default = float(np.median(known)) if known else 1.0
        speed = [self._speed_of(_hardware(i), default) for i in instances]
        gpus = [
            int(getattr(inst, "gpu_count", 0) or reqs[inst.model_id][0])
            for inst in instances
        ]
        hardware = [getattr(inst, "hardware", None) for inst in instances]
        pool = self._free_pool(slots, default)
        floor = {m: min(self.min_replicas, n) for m, n in current.items()}
        # A move costs a drain and a cold start; an instance that just
        # arrived stays put until it has paid that back.
        now = time.time()
        young = {
            i
            for i, inst in enumerate(instances)
            if now - _serving_since(inst) < self.min_age
        }

        whole = getattr(self._service, "whole_nodes", False) is True

        def ok(i: int, j: int) -> bool:
            """Machine ``i`` can become an instance of model ``j``."""
            if i >= M or j == index[instances[i].model_id]:
                return True
            if i in young:
                return False
            req = reqs[models[j]]
            # A whole worker takes any model it holds; a share of one
            # only a model of the same size.
            same = whole or req[0] == gpus[i]
            return same and _fits(req, gpus[i], hardware[i])

        added: list[tuple[str, str | None, float]] = []
        victims: list[int] = []
        room = None
        if self.elastic:
            room = self._make_room(
                instances,
                speed,
                gpus,
                pool,
                demand,
                steady,
                reqs,
                floor,
                young,
            )
        if room is not None:
            m, w, v, victims = room
            added = [(m, w, v)]
        elif self.elastic:
            grow, victims = self._resize(
                speed, bool(pool), demand, current, floor
            )
            if grow:
                added = self._grow(
                    instances, speed, pool, demand, steady, reqs, whole
                )
            elif victims:
                victims = self._pick_victims(
                    instances, speed, demand, floor, len(victims), young
                )
        if not demand and not added and not victims:
            return None

        initial = [index[i.model_id] for i in instances] + [-1] * len(added)
        machine_speed = np.array(speed + [v for _, _, v in added])
        workers = [getattr(i, "worker_id", None) for i in instances] + [
            w for _, w, _ in added
        ]
        supply = machine_speed.sum() - sum(speed[i] for i in victims)
        target_map = self._allocate(
            demand, steady, supply * self.target_concurrency
        )
        target = np.array([target_map.get(m, 0.0) for m in models])

        if added:
            # A growing round only adds: each new machine already has the
            # model it was placed for. Making room also drops the
            # instances in its way.
            deployment = initial[:M] + [index[m] for m, _, _ in added]
            for i in victims:
                deployment[i] = -1
        elif victims:
            deployment = list(initial)
            for i in victims:
                deployment[i] = -1
        elif not target.any():
            return None
        else:
            a = max(self.change_factor, 1.0 / M)
            B = np.tile(machine_speed[:, None], (1, len(models)))
            for i in range(M):
                for j in range(len(models)):
                    if not ok(i, j):
                        B[i, j] = 0
            optimizer = create_optimizer(
                self.algorithm,
                M=M,
                N=len(models),
                B=B,
                initial=np.array(initial),
                a=a,
                target=target,
            )
            result, _, _ = run_optimization(
                optimizer, objective_method=self.objective_method
            )
            deployment = [int(j) for j in result]
            # A machine the optimizer emptied or gave to a model it cannot
            # run (no capacity there) stays where it is.
            for i in range(M):
                if deployment[i] < 0 or not ok(i, deployment[i]):
                    deployment[i] = initial[i]
            self._keep_fastest(
                deployment, initial, machine_speed, floor, index, ok
            )
            _undo_needless_moves(deployment, initial, machine_speed, ok)
            # Whatever the passes above did, no machine ends on a model
            # it cannot run.
            for i in range(M):
                if not ok(i, deployment[i]):
                    deployment[i] = initial[i]

        changed = [
            i for i in range(len(deployment)) if deployment[i] != initial[i]
        ]
        if not changed:
            return None
        before = _share_error(initial, machine_speed, target)
        after = _share_error(deployment, machine_speed, target)
        # Every machine that moves must buy its share of the threshold.
        resized = bool(added or victims)
        moved = sum(1 for i in changed if initial[i] >= 0)
        if not resized and before - after < self.change_threshold * moved:
            return None

        counts = Counter(models[j] for j in deployment if j >= 0)
        remove = tuple(
            pid
            for i in changed
            if i < M
            and (pid := getattr(instances[i], "pylet_id", None)) is not None
        )
        place: dict[str, list[str | None]] = {}
        for i in changed:
            if deployment[i] >= 0:
                place.setdefault(models[deployment[i]], []).append(workers[i])
        return AutoPlan(
            target_state={m: counts[m] for m in models if counts[m] > 0},
            current_state=dict(current),
            changes=sum(1 for i in changed if deployment[i] >= 0),
            score_before=before,
            score_after=after,
            demand=target_map,
            scaler="tail",
            size_before=M,
            size_after=sum(counts.values()),
            remove=remove,
            place={m: tuple(w) for m, w in place.items()},
            speeds=self.speeds.snapshot(),
        )

    def _free_pool(
        self,
        slots: Sequence[tuple[str | None, str | None]],
        default: float,
    ) -> dict[object, list]:
        """Free GPUs per worker: key -> [worker, hardware, speed, GPUs].

        Slots without a worker are single places PyLet will choose.
        """
        pool: dict[object, list] = {}
        for k, (w, h) in enumerate(slots):
            key = w if w is not None else ("?", k)
            if key in pool:
                pool[key][3] += 1
            else:
                pool[key] = [w, h, self._speed_of(h, default), 1]
        return pool

    def _resize(
        self,
        speed: list[float],
        has_free: bool,
        demand: dict[str, float],
        current: Counter,
        floor: dict[str, int],
    ) -> tuple[bool, list[int]]:
        """Whether to grow, or instances to drop (by index).

        The dropped instances returned here are only as many as the
        pool should lose; ``_pick_victims`` chooses which, and ``_grow``
        where to grow.
        """
        M = len(speed)
        capacity = sum(speed)
        wanted = sum(demand.values()) / self.target_concurrency
        room = not self.max_instances or self.max_instances > M
        if wanted > capacity and has_free and room:
            self._low_rounds = 0
            return True, []

        # The slowest instances above the floors could go.
        smallest = max(self.min_instances, sum(floor.values()))
        over_cap = bool(self.max_instances) and self.max_instances < M
        drop: list[int] = []
        left = capacity
        for i in sorted(range(M), key=lambda i: speed[i]):
            if M - len(drop) <= smallest:
                break
            must = over_cap and M - len(drop) > self.max_instances
            if not must and left - speed[i] < wanted:
                break
            drop.append(i)
            left -= speed[i]
        if not drop:
            self._low_rounds = 0
            return False, []
        self._low_rounds += 1
        if self._low_rounds < self.scale_down_rounds and not over_cap:
            return False, []
        self._low_rounds = 0
        step = max(1, int(self.change_factor * M))
        return False, drop[:step]

    def _grow(
        self,
        instances: Sequence[_Instance],
        speed: list[float],
        pool: dict[object, list],
        demand: dict[str, float],
        steady: dict[str, float],
        reqs: dict[str, Requirement],
        whole: bool = False,
    ) -> list[tuple[str, str | None, float]]:
        """New instances for the models short of capacity.

        The model furthest below its share goes first, onto the free
        place that suits it best (``_best_place``), until the pool has
        the capacity it wants, no model short of capacity fits anywhere,
        or ``max_instances`` is reached. With ``whole`` workers an
        instance takes all of its worker.

        Returns:
            (model, worker, speed) per new instance.
        """
        tc = self.target_concurrency
        have: dict[str, float] = {}
        for inst, v in zip(instances, speed, strict=True):
            have[inst.model_id] = have.get(inst.model_id, 0.0) + v
        capacity = sum(speed)
        wanted = sum(demand.values()) / tc
        reachable = capacity + sum(p[2] * p[3] for p in pool.values())
        goal = {
            m: v / tc
            for m, v in self._allocate(demand, steady, reachable * tc).items()
        }
        room = (
            self.max_instances - len(instances) if self.max_instances else None
        )
        added: list[tuple[str, str | None, float]] = []
        while capacity < wanted and (room is None or len(added) < room):
            short = sorted(
                (m for m in goal if goal[m] - have.get(m, 0.0) > 1e-9),
                key=lambda m: -(goal[m] - have.get(m, 0.0)),
            )
            pick = None
            for m in short:
                key = self._best_place(
                    pool, reqs[m], goal[m] - have.get(m, 0.0), whole
                )
                if key is not None:
                    pick = (m, key)
                    break
            if pick is None:
                break
            m, key = pick
            slot = pool[key]
            slot[3] = 0 if whole else slot[3] - reqs[m][0]
            added.append((m, slot[0], slot[2]))
            have[m] = have.get(m, 0.0) + slot[2]
            capacity += slot[2]
        return added

    def _make_room(
        self,
        instances: Sequence[_Instance],
        speed: list[float],
        gpus: list[int],
        pool: dict[object, list],
        demand: dict[str, float],
        steady: dict[str, float],
        reqs: dict[str, Requirement],
        floor: dict[str, int],
        young: set[int],
    ) -> tuple[str, str, float, list[int]] | None:
        """Room for the model furthest short that fits on no free GPUs.

        That model's instance goes on the worker where the fewest GPUs
        must be freed, by removing instances of other models. An
        instance can go only if its model ends less short than this one
        is now and keeps its floor, and only if it is not young, so the
        largest shortfall always shrinks and two models do not take a
        worker back and forth.

        Returns:
            (model, worker, speed of the new instance, instances to
            remove by index), or None when the model is short by less
            than half the instance it would get, fits on free GPUs, or
            nowhere can be freed.
        """
        tc = self.target_concurrency
        have: dict[str, float] = {}
        count: Counter = Counter()
        for inst, v in zip(instances, speed, strict=True):
            have[inst.model_id] = have.get(inst.model_id, 0.0) + v
            count[inst.model_id] += 1
        reachable = sum(speed) + sum(p[2] * p[3] for p in pool.values())
        goal = {
            m: v / tc
            for m, v in self._allocate(demand, steady, reachable * tc).items()
        }
        if not goal:
            return None
        m = max(goal, key=lambda x: goal[x] - have.get(x, 0.0))
        short = goal[m] - have.get(m, 0.0)
        req = reqs[m]
        if req == _ANY or self._best_place(pool, req, short) is not None:
            return None

        # Workers, their hardware, free GPUs and instances.
        workers: dict[str, list] = {}
        for p in pool.values():
            if p[0] is not None:
                workers[p[0]] = [p[1], p[2], p[3], []]
        for i, inst in enumerate(instances):
            w = getattr(inst, "worker_id", None)
            if w is None:
                continue
            if w not in workers:
                hw = getattr(inst, "hardware", None)
                workers[w] = [hw, speed[i], 0, []]
            workers[w][3].append(i)

        best = None
        for w, (hw, v, free, members) in workers.items():
            # Short by less than half of what the worker adds: not worth
            # a drain and a cold start.
            if short < v / 2 or not _fits(req, req[0], hw):
                continue
            # Instances of the models with the most to spare go first.
            spare = {}
            for i in members:
                x = instances[i].model_id
                spare[i] = have[x] - goal.get(x, 0.0)
            freed = free
            lost: dict[str, float] = {}
            gone: Counter = Counter()
            chosen: list[int] = []
            for i in sorted(members, key=lambda i: (-spare[i], speed[i])):
                if freed >= req[0]:
                    break
                x = instances[i].model_id
                if x == m or i in young:
                    continue
                if count[x] - gone[x] <= floor.get(x, 0):
                    continue
                left = have[x] - lost.get(x, 0.0) - speed[i]
                if goal.get(x, 0.0) - left >= short - 1e-9:
                    continue
                chosen.append(i)
                lost[x] = lost.get(x, 0.0) + speed[i]
                gone[x] += 1
                freed += gpus[i]
            if freed < req[0] or not chosen:
                continue
            cost = (sum(gpus[i] for i in chosen), sum(lost.values()))
            if best is None or cost < best[0]:
                best = (cost, w, v, chosen)
        if best is None:
            return None
        _, w, v, chosen = best
        logger.info(
            f"[AUTO_OPTIMIZE] making room for {m} on {w}: removing "
            f"{len(chosen)} instance(s) of "
            f"{sorted({instances[i].model_id for i in chosen})}"
        )
        return m, w, v, chosen

    @staticmethod
    def _best_place(
        pool: dict[object, list],
        req: Requirement,
        short: float,
        whole: bool = False,
    ) -> object | None:
        """The free place for one instance of ``req`` short ``short``.

        The slowest hardware that covers the shortfall on its own, else
        the fastest, so fast GPUs stay free for the models that need
        them; then the worker with the fewest free GPUs that hold the
        instance (best fit), so large nodes stay whole for multi-GPU
        models. With ``whole`` workers the smallest worker comes first,
        since an instance takes all of it.
        """
        fits = [k for k, p in pool.items() if _fits(req, p[3], p[1])]
        if not fits:
            return None
        order = {k: n for n, k in enumerate(pool)}
        if whole:
            smallest = min(pool[k][3] for k in fits)
            fits = [k for k in fits if pool[k][3] == smallest]
        enough = [k for k in fits if pool[k][2] >= short - 1e-9]
        if enough:
            return min(enough, key=lambda k: (pool[k][2], pool[k][3], order[k]))
        return min(fits, key=lambda k: (-pool[k][2], pool[k][3], order[k]))

    def _pick_victims(
        self,
        instances: Sequence[_Instance],
        speed: list[float],
        demand: dict[str, float],
        floor: dict[str, int],
        count: int,
        young: set[int] = frozenset(),
    ) -> list[int]:
        """Slowest instances of the models with the most surplus capacity.

        Instances that just arrived go last.
        """
        alive = set(range(len(instances)))
        total = sum(demand.values())
        victims: list[int] = []
        for _ in range(count):
            capacity: dict[str, float] = {}
            members: dict[str, list[int]] = {}
            for i in alive:
                m = instances[i].model_id
                capacity[m] = capacity.get(m, 0.0) + speed[i]
                members.setdefault(m, []).append(i)
            left = sum(capacity.values())
            surplus = {
                m: capacity[m]
                - (left * demand.get(m, 0.0) / total if total else 0.0)
                for m in members
                if len(members[m]) > floor.get(m, 0)
            }
            if not surplus:
                break
            victim_model = max(surplus, key=lambda m: surplus[m])
            i = min(members[victim_model], key=lambda i: (i in young, speed[i]))
            alive.discard(i)
            victims.append(i)
        return victims

    def _allocate(
        self,
        demand: dict[str, float],
        steady: dict[str, float],
        supply: float,
    ) -> dict[str, float]:
        """Need to aim for when the supply falls short of it.

        Arrivals come first, then backlog, each shared max-min fairly: a
        model that needs less than an equal share gets all it needs, so a
        light model is not starved by a heavy one that cannot be served
        anyway.
        """
        total = sum(demand.values())
        if total <= supply or supply <= 0:
            return dict(demand)
        base = {m: min(steady.get(m, 0.0), v) for m, v in demand.items()}
        if sum(base.values()) <= 0:
            return _water_fill(demand, supply)
        got = _water_fill(base, supply)
        left = supply - sum(got.values())
        if left > 1e-9:
            extra = _water_fill({m: demand[m] - got[m] for m in demand}, left)
            got = {m: got[m] + extra[m] for m in demand}
        return got

    def _keep_fastest(
        self,
        deployment: list[int],
        initial: list[int],
        speed: np.ndarray,
        floor: dict[str, int],
        index: dict[str, int],
        ok=lambda i, j: True,
    ) -> None:
        """Keep each model's floor on machines that do not move.

        Removals drain before replacements start, so a model that trades
        all its machines for others serves nothing in between. A model
        below its floor of unmoved machines keeps its slowest outgoing
        one, and the machine coming to it that is closest in speed goes
        to that machine's new model instead, so both keep their capacity,
        if it can run that model (``ok(machine, model index)``).
        """
        for m, need in floor.items():
            j = index[m]
            kept = sum(
                1
                for i in range(len(deployment))
                if initial[i] == j and deployment[i] == j
            )
            for _ in range(max(0, need - kept)):
                out = [
                    i
                    for i in range(len(deployment))
                    if initial[i] == j and deployment[i] != j
                ]
                if not out:
                    break
                i = min(out, key=lambda i: speed[i])
                incoming = [
                    x
                    for x in range(len(deployment))
                    if deployment[x] == j
                    and initial[x] != j
                    and ok(x, deployment[i])
                ]
                if incoming:
                    x = min(incoming, key=lambda x: abs(speed[x] - speed[i]))
                    deployment[x] = deployment[i]
                deployment[i] = j

    def _keep_min_replicas(
        self,
        counts: Counter,
        current: Counter,
        reqs: dict[str, Requirement] | None = None,
    ) -> None:
        """Give a model below its floor machines of its own shape."""
        reqs = reqs or {}
        for m, have in current.items():
            floor = min(self.min_replicas, have)
            while counts[m] < floor:
                same = [
                    k for k in counts if reqs.get(k, _ANY) == reqs.get(m, _ANY)
                ]
                if not same:
                    break
                donor = max(same, key=lambda k: counts[k])
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
        slots = None
        service = self._service
        if (
            self.elastic
            and self.hardware_aware
            and hasattr(service, "free_slots")
        ):
            slots = await asyncio.to_thread(service.free_slots)
        if (
            self.elastic
            and slots is None
            and hasattr(service, "spare_capacity")
        ):
            spare = await asyncio.to_thread(service.spare_capacity)
        plan = self.plan(instances, spare=spare, slots=slots)
        if plan is None:
            return None
        logger.info(
            f"[AUTO_OPTIMIZE] {plan.scaler} demand={plan.demand} "
            f"{plan.current_state} -> {plan.target_state} "
            f"score {plan.score_before:.3f} -> {plan.score_after:.3f} "
            f"size {plan.size_before} -> {plan.size_after}"
            + (
                f" remove={len(plan.remove)} place={plan.place} "
                f"speeds={plan.speeds}"
                if self.hardware_aware
                else ""
            )
        )
        # Reconciling blocks (deploys, drains); keep it off the loop.
        if plan.remove or plan.place:
            result = await asyncio.to_thread(
                self._service.apply_deployment,
                plan.target_state,
                remove=plan.remove,
                place=plan.place,
            )
        else:
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
