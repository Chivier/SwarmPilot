"""Discrete-event routing sim driving the real SwarmPilot strategy classes.

Each instance is a FIFO single server. Arrivals are Poisson; each task has a
prompt-dependent mean m (heavy-tailed lognormal across prompts) and actual
service time m * lognormal(cv=0.3). The predictor runs in experiment mode
with exp_runtime=m, exp_cv=0.3, i.e. a well-calibrated per-request predictor.

Every routed task is charged to its instance's queue sketch and released
when it finishes, as the Scheduler's result callback does. ``residual_ms``
is the largest sketch value left once every task has finished; it must be 0.

Modes:
  as_is   - the repo's behaviour.
  noseed  - as_is, but with the per-predict random.seed(42)/np.random.seed(42)
            in PredictorClient neutralised (until plan item 2.3 removes it).

The baseline file from commit 8a9cee4 also has a "fixed" mode that patched
sketch subtraction in from outside; plan item 2.2 made that the real code.

Usage:
    uv run python tests/perf/sim_routing.py [n_tasks=3000] [load=0.8]
"""

import asyncio
import heapq
import json
import random
import sys
import tempfile

import numpy as np

from swarmpilot.scheduler.algorithms.factory import get_strategy
from swarmpilot.scheduler.clients import predictor_library_client as plc
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
N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
LOAD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
PLAT = {
    "software_name": "exp",
    "software_version": "exp",
    "hardware_name": "exp",
}


def workload(seed=0):
    rng = np.random.default_rng(seed)
    means = rng.lognormal(mean=np.log(1000), sigma=1.0, size=N_TASKS)
    sig = np.sqrt(np.log(1 + 0.3**2))
    actual = means * rng.lognormal(-(sig**2) / 2, sig, size=N_TASKS)
    rate = LOAD * N_INST / actual.mean()
    arrivals = np.cumsum(rng.exponential(1 / rate, size=N_TASKS))
    return means, actual, arrivals


async def run(strategy, mode, means, actual, arrivals):
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
            platform_info=PLAT,
        )
        for k in range(N_INST)
    ]
    for i in insts:
        await reg.register(i)
    pc = PredictorClient(storage_dir=tempfile.mkdtemp(prefix="sim-models-"))
    strat = get_strategy(strategy, pc, reg)
    free_at = [0.0] * N_INST
    pending = []  # heap of (finish_time, task index)
    lat, chosen = [], []
    for t in range(N_TASKS):
        now = arrivals[t]
        while pending and pending[0][0] <= now:
            _, done = heapq.heappop(pending)
            await reg.release_charge(f"t{done}")
        md = {"exp_runtime": float(means[t]), "exp_cv": 0.3}
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
    lat = np.array(lat) / 1000
    share = np.bincount(chosen, minlength=N_INST) / N_TASKS
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
    for t in range(N_TASKS):
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
    # neutralise the per-call reseed
    plc.random.seed = lambda *a, **k: None
    plc.np.random.seed = lambda *a, **k: None
    random.seed(1)
    np.random.seed(1)
    for s in ["random", "po2", "probabilistic"]:
        out.append(await run(s, "noseed", means, actual, arrivals))
    for r in out:
        print(json.dumps(r))


asyncio.run(main())
