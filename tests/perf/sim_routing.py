"""Discrete-event routing sim driving the real SwarmPilot strategy classes.

Each instance is a FIFO single server. Arrivals are Poisson; each task has a
prompt-dependent mean m (heavy-tailed lognormal across prompts) and actual
service time m * lognormal(cv=0.3). The predictor runs in experiment mode
with exp_runtime=m, exp_cv=0.3, i.e. a well-calibrated per-request predictor.

Every routed task is charged to its instance's queue sketch and released
when it finishes, as the Scheduler's result callback does. ``residual_ms``
is the largest sketch value left once every task has finished; it must be 0.

Strategies get a fixed seed so runs repeat. The baseline file from commit
8a9cee4 also has "fixed" and "noseed" modes that patched the sketch and
reseed bugs from outside; plan items 2.2 and 2.3 made both the real code.

Usage:
    uv run python tests/perf/sim_routing.py [n_tasks=3000] [load=0.8]
"""

import asyncio
import heapq
import json
import sys
import tempfile

import numpy as np

from swarmpilot.scheduler.algorithms.factory import get_strategy
from swarmpilot.scheduler.clients.predictor_library_client import (
    PredictorClient,
)
from swarmpilot.scheduler.models import (
    Instance,
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)
from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry

N_INST = 8
N_TASKS = 3000
LOAD = 0.8
PLAT = {
    "software_name": "exp",
    "software_version": "exp",
    "hardware_name": "exp",
}


def workload(seed=0, n_tasks=None, load=None):
    n_tasks = N_TASKS if n_tasks is None else n_tasks
    load = LOAD if load is None else load
    rng = np.random.default_rng(seed)
    means = rng.lognormal(mean=np.log(1000), sigma=1.0, size=n_tasks)
    sig = np.sqrt(np.log(1 + 0.3**2))
    actual = means * rng.lognormal(-(sig**2) / 2, sig, size=n_tasks)
    rate = load * N_INST / actual.mean()
    arrivals = np.cumsum(rng.exponential(1 / rate, size=n_tasks))
    return means, actual, arrivals


async def run(
    strategy,
    mode,
    means,
    actual,
    arrivals,
    chosen_out=None,
    predictor_client=None,
    platform=None,
    metadata_fn=None,
):
    """Route every task; ``chosen_out`` receives each task's instance.

    By default the predictor runs in experiment mode. A test can pass a
    ``predictor_client`` holding trained models for ``platform``, with
    ``metadata_fn(t)`` giving task ``t``'s features.
    """
    n_tasks = len(arrivals)
    qtype = (
        "probabilistic"
        if strategy in ("probabilistic", "adaptive_bootstrap")
        else "expect_error"
    )
    reg = InstanceRegistry(queue_info_type=qtype)
    insts = [
        Instance(
            instance_id=f"i{k}",
            model_id="m",
            endpoint=f"http://x{k}",
            platform_info=platform or PLAT,
        )
        for k in range(N_INST)
    ]
    for i in insts:
        await reg.register(i)
    pc = predictor_client or PredictorClient(
        storage_dir=tempfile.mkdtemp(prefix="sim-models-")
    )
    strat = get_strategy(strategy, pc, reg, seed=1)
    free_at = [0.0] * N_INST
    pending = []  # heap of (finish_time, task index)
    lat, chosen = [], []
    for t in range(n_tasks):
        now = arrivals[t]
        while pending and pending[0][0] <= now:
            _, done = heapq.heappop(pending)
            await reg.release_charge(f"t{done}")
        md = (
            metadata_fn(t)
            if metadata_fn
            else {"exp_runtime": float(means[t]), "exp_cv": 0.3}
        )
        res = await strat.schedule_task("m", md, insts, task_id=f"t{t}")
        k = int(res.selected_instance_id[1:])
        start = max(now, free_at[k])
        free_at[k] = start + actual[t]
        heapq.heappush(pending, (free_at[k], t))
        lat.append(free_at[k] - now)
        chosen.append(k)
    while pending:
        _, done = heapq.heappop(pending)
        await reg.release_charge(f"t{done}")
    residual = 0.0
    for q in (await reg.get_all_queue_info()).values():
        if isinstance(q, InstanceQueueProbabilistic):
            residual = max(residual, *q.values)
        elif isinstance(q, InstanceQueueExpectError):
            residual = max(residual, q.expected_time_ms)
    if chosen_out is not None:
        chosen_out.extend(chosen)
    lat = np.array(lat) / 1000
    share = np.bincount(chosen, minlength=N_INST) / n_tasks
    return {
        "strategy": strategy,
        "mode": mode,
        "p50_s": round(float(np.percentile(lat, 50)), 2),
        "p95_s": round(float(np.percentile(lat, 95)), 2),
        "p99_s": round(float(np.percentile(lat, 99)), 2),
        "max_share": round(float(share.max()), 3),
        "residual_ms": residual,
    }


def oracle_jsq(means, actual, arrivals):
    """Reference: send each task to the instance with least true backlog."""
    free_at = [0.0] * N_INST
    lat = []
    for t in range(len(arrivals)):
        now = arrivals[t]
        k = int(np.argmin([max(f, now) for f in free_at]))
        start = max(now, free_at[k])
        free_at[k] = start + actual[t]
        lat.append(free_at[k] - now)
    lat = np.array(lat) / 1000
    return {
        "strategy": "oracle_least_work",
        "mode": "ref",
        "p50_s": round(float(np.percentile(lat, 50)), 2),
        "p95_s": round(float(np.percentile(lat, 95)), 2),
        "p99_s": round(float(np.percentile(lat, 99)), 2),
        "max_share": None,
    }


async def main():
    from loguru import logger

    logger.remove()
    means, actual, arrivals = workload()
    out = [oracle_jsq(means, actual, arrivals)]
    for s in [
        "round_robin",
        "random",
        "po2",
        "min_time",
        "probabilistic",
        "adaptive_bootstrap",
    ]:
        out.append(await run(s, "as_is", means, actual, arrivals))
    for r in out:
        print(json.dumps(r))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        N_TASKS = int(sys.argv[1])
    if len(sys.argv) > 2:
        LOAD = float(sys.argv[2])
    asyncio.run(main())
