"""End-to-end A/B: the routing simulation on the Python and the Rust core.

``tests/perf/sim_routing.py`` drives the real strategy classes, registry
and predictor (experiment mode) through a discrete-event queueing model.
With the same seeds both cores must route every task to the same
instance, so latency percentiles and the final sketches match exactly.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

SIM = Path(__file__).parents[1] / "perf" / "sim_routing.py"
STRATEGIES = (
    "round_robin",
    "random",
    "po2",
    "min_time",
    "serverless",
    "probabilistic",
    "adaptive_bootstrap",
)


def _sim():
    spec = importlib.util.spec_from_file_location("sim_routing", SIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_routing_sim_identical_on_both_cores(use_core, strategy):
    from loguru import logger

    sim = _sim()
    means, actual, arrivals = sim.workload(seed=3, n_tasks=600, load=0.85)
    results, chosen = {}, {}
    logger.disable("swarmpilot")
    try:
        for core_name in ("python", "rust"):
            use_core(core_name)
            chosen[core_name] = []
            results[core_name] = asyncio.run(
                sim.run(
                    strategy,
                    core_name,
                    means,
                    actual,
                    arrivals,
                    chosen_out=chosen[core_name],
                )
            )
    finally:
        logger.enable("swarmpilot")
    assert chosen["python"] == chosen["rust"]
    for key in ("p50_s", "p95_s", "p99_s", "max_share", "residual_ms"):
        assert results["python"][key] == results["rust"][key], key
    assert results["rust"]["residual_ms"] == 0
