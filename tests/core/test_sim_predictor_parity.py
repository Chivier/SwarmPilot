"""Routing simulation with trained predictors on both cores.

The predictor's network now runs in Rust on the Rust core, and its
outputs differ from torch in the last float32 bits, so a decision that
was a near tie can flip. The bar is behavioural: almost every task goes
to the same instance and the latency percentiles stay within noise.
"""

import asyncio
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from swarmpilot.predictor.models import PlatformInfo

SIM = Path(__file__).parents[1] / "perf" / "sim_routing.py"
PLATFORM = {
    "software_name": "vllm",
    "software_version": "0.11",
    "hardware_name": "sim-gpu",
}


def _sim():
    spec = importlib.util.spec_from_file_location("sim_routing", SIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trained_client():
    """A PredictorClient with both model kinds trained on sim tasks."""
    from swarmpilot.scheduler.clients.predictor_library_client import (
        PredictorClient,
    )

    sim = _sim()
    torch.manual_seed(0)
    np.random.seed(0)
    means, actual, _ = sim.workload(seed=11, n_tasks=400)
    samples = [
        {"size": float(m), "runtime_ms": float(a)}
        for m, a in zip(means, actual, strict=True)
    ]
    pc = PredictorClient(storage_dir=tempfile.mkdtemp(prefix="sim-trained-"))
    low = pc._low_level
    for kind, config in (
        ("quantile", {"epochs": 150, "log_transform": {"enabled": True}}),
        ("expect_error", {"epochs": 150}),
    ):
        predictor = low.train_predictor(samples, kind, config)
        low.save_model("m", PlatformInfo(**PLATFORM), kind, predictor)
    return pc


@pytest.mark.parametrize(
    "strategy", ["probabilistic", "adaptive_bootstrap", "min_time", "po2"]
)
def test_trained_routing_matches_across_cores(
    use_core, trained_client, strategy
):
    from loguru import logger

    sim = _sim()
    means, actual, arrivals = sim.workload(seed=5, n_tasks=600, load=0.85)
    results, chosen = {}, {}
    logger.disable("swarmpilot")
    try:
        for name in ("python", "rust"):
            use_core(name)
            # Fresh predictor objects, so each core loads and caches its own.
            trained_client._low_level.clear_cache()
            chosen[name] = []
            results[name] = asyncio.run(
                sim.run(
                    strategy,
                    name,
                    means,
                    actual,
                    arrivals,
                    chosen_out=chosen[name],
                    predictor_client=trained_client,
                    platform=PLATFORM,
                    metadata_fn=lambda t: {"size": float(means[t])},
                )
            )
    finally:
        logger.enable("swarmpilot")
    same = np.mean(np.array(chosen["python"]) == np.array(chosen["rust"]))
    assert same >= 0.98, same
    for key in ("p50_s", "p95_s", "p99_s"):
        py, rs = results["python"][key], results["rust"][key]
        assert abs(py - rs) <= 0.05 * max(py, 0.1), (key, py, rs)
    assert results["rust"]["residual_ms"] == 0
