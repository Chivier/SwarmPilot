"""Discrete-event routing sim driving the real SwarmPilot strategy classes.

Each instance is a FIFO single server. Arrivals are Poisson; each task has a
prompt-dependent mean m (heavy-tailed lognormal across prompts) and actual
service time m * lognormal(cv=0.3). The predictor runs in experiment mode
with exp_runtime=m, exp_cv=0.3, i.e. a well-calibrated per-request predictor.

Modes:
  as_is   - exactly the repo's behaviour: queue sketches are only ever added
            to (nothing subtracts them when a task completes).
  fixed   - same strategies, but when a task completes its predicted
            quantiles are subtracted from its instance's queue sketch.
  noseed  - as_is, but with the per-predict random.seed(42)/np.random.seed(42)
            in PredictorClient neutralised.

The "fixed" and "noseed" modes patch around the bugs from outside, so they
show what plan items 2.2 and 2.3 should achieve. Once those land, "as_is"
should match them.

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


async def subtract(reg, iid, pred):
    q = await reg.get_queue_info(iid)
    if isinstance(q, InstanceQueueProbabilistic) and pred.quantiles:
        ks = sorted(pred.quantiles)
        tv = np.interp(q.quantiles, ks, [pred.quantiles[k] for k in ks])
        vals = [max(0.0, v - t) for v, t in zip(q.values, tv)]
        await reg.update_queue_info(
            iid,
            InstanceQueueProbabilistic(
                instance_id=iid, quantiles=q.quantiles, values=vals
            ),
        )
    elif isinstance(q, InstanceQueueExpectError):
        await reg.update_queue_info(
            iid,
            InstanceQueueExpectError(
                instance_id=iid,
                expected_time_ms=max(
                    0.0, q.expected_time_ms - pred.predicted_time_ms
                ),
                error_margin_ms=q.error_margin_ms,
            ),
        )


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
    pending = []  # heap of (finish_time, inst_idx, prediction)
    lat, chosen = [], []
    for t in range(N_TASKS):
        now = arrivals[t]
        while pending and pending[0][0] <= now:
            _, k, _, pred = heapq.heappop(pending)
            if mode == "fixed" and pred is not None:
                await subtract(reg, f"i{k}", pred)
        md = {"exp_runtime": float(means[t]), "exp_cv": 0.3}
        res = await strat.schedule_task("m", md, insts)
        k = int(res.selected_instance_id[1:])
        start = max(now, free_at[k])
        free_at[k] = start + actual[t]
        heapq.heappush(pending, (free_at[k], k, t, res.selected_prediction))
        lat.append(free_at[k] - now)
        chosen.append(k)
    lat = np.array(lat) / 1000
    share = np.bincount(chosen, minlength=N_INST) / N_TASKS
    return {
        "strategy": strategy,
        "mode": mode,
        "p50_s": round(float(np.percentile(lat, 50)), 2),
        "p95_s": round(float(np.percentile(lat, 95)), 2),
        "p99_s": round(float(np.percentile(lat, 99)), 2),
        "max_share": round(float(share.max()), 3),
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
    for s in ["round_robin", "random", "po2", "min_time", "probabilistic"]:
        out.append(await run(s, "as_is", means, actual, arrivals))
    for s in ["min_time", "probabilistic"]:
        out.append(await run(s, "fixed", means, actual, arrivals))
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
