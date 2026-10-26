"""SLO misses of the tail scaler against the count scaler (item 3.4).

A discrete-event simulation of an agent workload on a fixed pool of
machines, each serving one model at a time. Agent sessions arrive at the
``llm`` model. Every ``llm`` step calls the ``tool`` model (a diffusion-
like, heavy-tailed service) with probability ``p_tool``, and every tool
result goes back to the ``llm`` with probability ``p_back``, so calls form
a cycle. Calls carry their caller as the parent model, as the
``X-SwarmPilot-Parent`` header does.

Every ``report_s`` the Schedulers' reports go into a real ``DemandStore``:
outstanding tasks, arrival rate, calls by caller, q0.9 outstanding work
(normal approximation of the sum of the outstanding tasks' service times,
which is what the convolved queue sketches hold) and each instance's mean
execution time. Every ``interval_s`` the real ``AutoOptimizer`` (integer
programming, change factor 0.3) plans, and the plan is applied: a machine
that changes model finishes its task, then needs ``startup_s`` before it
serves the new model. Each model has one FIFO queue served by its ready
machines.

The arrival rate steps up and then down. A task misses its SLO when its
latency (wait plus service) exceeds ``slo_factor`` times its model's mean
service time. Both scalers run on the same seeds.

Usage:
    uv run python tests/perf/agent_scaling.py [seeds=10]
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from swarmpilot.planner.services.auto_optimizer import AutoOptimizer
from swarmpilot.planner.services.demand import DemandStore

Z90 = 1.2816


@dataclass
class Model:
    """Service time of one model: lognormal with this mean and sigma."""

    mean: float
    sigma: float

    @property
    def var(self) -> float:
        return self.mean**2 * (math.exp(self.sigma**2) - 1)

    def sample(self, rng: np.random.Generator) -> float:
        mu = math.log(self.mean) - self.sigma**2 / 2
        return float(rng.lognormal(mu, self.sigma))


# Short agent steps: the LLM step is quick, the tool is heavy-tailed.
SHORT = {
    "models": {"llm": Model(1.0, 0.3), "tool": Model(2.0, 1.0)},
    "machines": 12,
    "phases": ((600, 1.0), (900, 2.0), (900, 0.8)),
}
# Long agent steps: tool calls trail an LLM surge by several seconds.
LONG = {
    "models": {"llm": Model(6.0, 0.5), "tool": Model(2.0, 1.0)},
    "machines": 24,
    "phases": ((600, 0.6), (900, 1.2), (900, 0.5)),
}
SCENARIOS = {"short_steps": SHORT, "long_steps": LONG}


@dataclass
class Task:
    """One task; ``service`` is drawn when it is submitted."""

    model: str
    created: float
    parent: str | None
    service: float


@dataclass
class Machine:
    """A machine serving ``model``, or moving to ``next_model``."""

    idx: int
    model: str
    gen: int = 0
    ready_at: float = 0.0
    busy: Task | None = None
    next_model: str | None = None
    done: list[float] = field(default_factory=list)

    @property
    def endpoint(self) -> str:
        return f"http://m{self.idx}-g{self.gen}"


class _Inst:
    def __init__(self, m: Machine):
        self.model_id = m.next_model or m.model
        self.endpoint = m.endpoint


class Sim:
    """The cluster, the Schedulers' reports and the Planner loop."""

    def __init__(
        self,
        scaler: str,
        seed: int,
        models: dict[str, Model],
        machines: int,
        phases: tuple[tuple[float, float], ...],
        initial: dict[str, int] | None = None,
        p_tool: float = 0.8,
        p_back: float = 0.5,
        report_s: float = 5.0,
        interval_s: float = 30.0,
        startup_s: float = 20.0,
        slo_factor: float = 4.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.models = models
        self.phases = phases
        self.p_tool, self.p_back = p_tool, p_back
        self.report_s, self.interval_s = report_s, interval_s
        self.startup_s, self.slo_factor = startup_s, slo_factor
        initial = initial or {"llm": machines // 2, "tool": machines // 2}
        self.machines = [
            Machine(i, m)
            for i, m in enumerate(
                m for m, n in initial.items() for _ in range(n)
            )
        ]
        self.queues: dict[str, deque[Task]] = {m: deque() for m in self.models}
        self.now = 0.0
        self.events: list = []
        self._seq = itertools.count()
        self.store = DemandStore(alpha=0.5, clock=lambda: self.now)
        self.opt = AutoOptimizer(
            self.store, self, interval=interval_s, scaler=scaler
        )
        self.arrivals: Counter[str] = Counter()
        self.calls: dict[str, Counter[str]] = {
            m: Counter() for m in self.models
        }
        self.latency: dict[str, list[float]] = {m: [] for m in self.models}
        self.misses: Counter[str] = Counter()
        self.moves = 0

    # Deployment service interface for the AutoOptimizer -----------------

    def get_active_instances(self) -> list[_Inst]:
        return [_Inst(m) for m in self.machines]

    def apply_deployment(self, target_state, wait_for_ready=True):
        have = Counter(_Inst(m).model_id for m in self.machines)
        spare = {
            m: have[m] - target_state.get(m, 0)
            for m in have
            if have[m] > target_state.get(m, 0)
        }
        need = [
            m
            for m, n in target_state.items()
            for _ in range(max(0, n - have[m]))
        ]
        for model, n in spare.items():
            pool = [m for m in self.machines if _Inst(m).model_id == model]
            # Move idle machines first.
            pool.sort(key=lambda m: m.busy is not None)
            for m in pool[:n]:
                if not need:
                    break
                self.moves += 1
                m.next_model = need.pop()
                if m.busy is None:
                    self._switch(m)

    # Events --------------------------------------------------------------

    def _at(self, t: float, kind: str, *args) -> None:
        heapq.heappush(self.events, (t, next(self._seq), kind, args))

    def _rate(self, t: float) -> float:
        end = 0.0
        for length, rate in self.phases:
            end += length
            if t < end:
                return rate
        return 0.0

    def _submit(self, model: str, parent: str | None) -> None:
        task = Task(
            model, self.now, parent, self.models[model].sample(self.rng)
        )
        self.arrivals[model] += 1
        if parent:
            self.calls[model][parent] += 1
        self.queues[model].append(task)
        self._dispatch(model)

    def _switch(self, m: Machine) -> None:
        m.model, m.next_model = m.next_model, None
        m.gen += 1
        m.done = []
        m.ready_at = self.now + self.startup_s
        self._at(m.ready_at, "ready", m)

    def _dispatch(self, model: str) -> None:
        q = self.queues[model]
        for m in self.machines:
            if not q:
                return
            if (
                m.model == model
                and m.busy is None
                and m.next_model is None
                and m.ready_at <= self.now
            ):
                m.busy = q.popleft()
                self._at(self.now + m.busy.service, "done", m)

    def _complete(self, m: Machine) -> None:
        task, m.busy = m.busy, None
        m.done.append(task.service)
        lat = self.now - task.created
        self.latency[task.model].append(lat)
        if lat > self.slo_factor * self.models[task.model].mean:
            self.misses[task.model] += 1
        if task.model == "llm" and self.rng.random() < self.p_tool:
            self._submit("tool", "llm")
        elif task.model == "tool" and self.rng.random() < self.p_back:
            self._submit("llm", "tool")
        if m.next_model is not None:
            self._switch(m)
        else:
            self._dispatch(m.model)

    def _report(self) -> None:
        for model, spec in self.models.items():
            running = [
                m.busy
                for m in self.machines
                if m.busy is not None and m.busy.model == model
            ]
            n = len(self.queues[model]) + len(running)
            work = n * spec.mean + Z90 * math.sqrt(n * spec.var)
            self.store.submit_target(
                model,
                float(n),
                arrival_rate=self.arrivals[model] / self.report_s,
                work_s=work,
                calls_from={
                    p: c / self.report_s for p, c in self.calls[model].items()
                },
            )
        for m in self.machines:
            if m.done:
                self.store.submit_throughput(
                    m.endpoint, float(np.mean(m.done)), model_id=m.model
                )
                m.done = []
        self.arrivals.clear()
        for c in self.calls.values():
            c.clear()

    def run(self) -> dict:
        horizon = sum(length for length, _ in self.phases)
        self._at(float(self.rng.exponential(1 / self._rate(0))), "arrive")
        self._at(self.report_s, "report")
        self._at(self.interval_s, "plan")
        while self.events:
            t, _, kind, args = heapq.heappop(self.events)
            self.now = t
            if kind == "arrive":
                self._submit("llm", None)
                rate = self._rate(t)
                nxt = t + float(self.rng.exponential(1 / rate)) if rate else 0
                if rate and nxt < horizon:
                    self._at(nxt, "arrive")
            elif kind == "done":
                self._complete(*args)
            elif kind == "ready":
                self._dispatch(args[0].model)
            elif kind == "report":
                self._report()
                if t < horizon:
                    self._at(t + self.report_s, "report")
            elif kind == "plan":
                plan = self.opt.plan(self.get_active_instances())
                if plan is not None:
                    self.apply_deployment(plan.target_state)
                if t < horizon:
                    self._at(t + self.interval_s, "plan")
        total = sum(len(v) for v in self.latency.values())
        return {
            "scaler": self.opt.scaler,
            "tasks": total,
            "slo_miss_rate": sum(self.misses.values()) / total,
            "miss_rate": {
                m: self.misses[m] / max(1, len(v))
                for m, v in self.latency.items()
            },
            "p99_s": {
                m: float(np.percentile(v, 99)) for m, v in self.latency.items()
            },
            "moves": self.moves,
        }


def simulate(scaler: str, seed: int, scenario: str = "short_steps") -> dict:
    """Run one simulation and return its SLO summary."""
    logger.disable("swarmpilot")
    try:
        return {
            "scenario": scenario,
            "seed": seed,
            **Sim(scaler, seed, **SCENARIOS[scenario]).run(),
        }
    finally:
        logger.enable("swarmpilot")


def main() -> None:
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for scenario in SCENARIOS:
        for seed in range(seeds):
            for scaler in ("count", "tail"):
                print(json.dumps(simulate(scaler, seed, scenario)))


if __name__ == "__main__":
    main()
