"""How fast the predictor recovers from a distribution shift (item 3.1).

A Scheduler-side loop with the real PredictorClient and TrainingClient.
Tasks arrive at ``rate`` per second with LLM features; runtime is
``(50 + 0.05 * prompt_length + 0.8 * max_tokens) * lognormal(0, 0.15)``
times a scale. Each task is predicted with the installed quantile model
and its result fed back with that prediction, as the result callback does.

After the model is warm, the scale steps from 1 to ``shift`` at t = 0. The
clock advances ``1 / rate`` per task plus the wall time of any retrain the
loop waits for. Recovery is the first time the q0.9 coverage of the last
100 post-shift tasks is back to at least 0.8 and stays there for the rest
of the run.

``drift`` mode is the new default; ``batch`` retrains every 100 samples on
the rolling window, as before 3.1.

Usage:
    uv run python tests/perf/drift_recovery.py [shift=2.0]
"""

import asyncio
import json
import sys
import tempfile
import time

import numpy as np

from swarmpilot.scheduler.clients.predictor_library_client import (
    PredictorClient,
)
from swarmpilot.scheduler.clients.training_library_client import (
    TrainingClient,
)
from swarmpilot.scheduler.models import Instance

PLAT = {
    "software_name": "vllm",
    "software_version": "0.11",
    "hardware_name": "sim",
}
SHIFT = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
POST_TASKS = 2000


def _task(rng: np.random.Generator, scale: float) -> tuple[dict, float]:
    plen = float(rng.integers(50, 4000))
    maxt = float(rng.choice([128, 256, 512, 1024]))
    base = 50 + 0.05 * plen + 0.8 * maxt
    return (
        {"prompt_length": plen, "max_tokens": maxt},
        base * float(rng.lognormal(0, 0.15)) * scale,
    )


async def run(mode: str, rate: float, seed: int = 0) -> dict:
    """Run one shift and return its recovery figures."""
    rng = np.random.default_rng(seed)
    predictor = PredictorClient(storage_dir=tempfile.mkdtemp())
    trainer = TrainingClient(
        storage=predictor._low_level._storage,
        cache=predictor._low_level._cache,
        batch_size=100,
        prediction_types=["quantile"],
        retrain_mode=mode,
        seed=seed,
    )
    inst = Instance(
        instance_id="i0",
        model_id="m",
        endpoint="http://sim",
        platform_info=PLAT,
    )

    async def serve(scale: float) -> tuple[bool, float]:
        feats, y = _task(rng, scale)
        q = None
        if trainer._is_warm("m", PLAT):
            pred = (await predictor.predict("m", feats, [inst]))[0]
            q = pred.quantiles
        trainer.add_sample("m", PLAT, feats, y, predicted_quantiles=q)
        fit_s = 0.0
        if trainer.start_flush_if_ready():
            t0 = time.perf_counter()
            await trainer._background
            fit_s = time.perf_counter() - t0
        hit = q is not None and y <= q[0.9]
        return hit, fit_s

    for _ in range(40):  # warm up on the old distribution
        for _ in range(100):
            await serve(1.0)
        if trainer._is_warm("m", PLAT):
            break
    for _ in range(300):  # settle
        await serve(1.0)

    clock, hits, times, fits = 0.0, [], [], 0.0
    for _ in range(POST_TASKS):
        hit, fit_s = await serve(SHIFT)
        clock += 1.0 / rate + fit_s
        fits += fit_s
        hits.append(hit)
        times.append(clock)
    cover = np.convolve(hits, np.ones(100) / 100, mode="valid")
    ok = cover >= 0.8
    recovered = None
    for k in range(len(ok)):
        if ok[k:].all():
            recovered = times[k + 99]
            break
    await trainer.close()
    return {
        "mode": mode,
        "rate_per_s": rate,
        "shift": SHIFT,
        "recovery_s": None if recovered is None else round(recovered, 1),
        "final_q90_coverage": round(float(np.mean(hits[-200:])), 3),
        "retrain_wall_s": round(fits, 2),
    }


async def main() -> None:
    """Print one JSON line per (mode, rate)."""
    for mode in ("drift", "batch"):
        for rate in (2.0, 5.0, 20.0):
            print(json.dumps(await run(mode, rate)), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
