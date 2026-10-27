"""Every strategy gets the queue sketch kind it reads, however it starts."""

import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from swarmpilot.scheduler.algorithms import queue_info_type_for
from swarmpilot.scheduler.api import app, instance_registry

EXPECTED = {
    "min_time": "expect_error",
    "po2": "expect_error",
    "serverless": "expect_error",
    "probabilistic": "probabilistic",
    "adaptive_bootstrap": "probabilistic",
    "round_robin": "probabilistic",
    "random": "probabilistic",
}


@pytest.mark.parametrize(("strategy", "kind"), sorted(EXPECTED.items()))
def test_queue_info_type_for(strategy: str, kind: str) -> None:
    assert queue_info_type_for(strategy) == kind


@pytest.mark.parametrize(("strategy", "kind"), sorted(EXPECTED.items()))
def test_strategy_set_switches_queue_kind_and_reports_name(
    strategy: str, kind: str
) -> None:
    import swarmpilot.scheduler.api as api_module

    saved = (api_module.scheduling_strategy, instance_registry._queue_info_type)
    try:
        client = TestClient(app)
        r = client.post("/v1/strategy/set", json={"strategy_name": strategy})
        assert r.status_code == 200, r.text
        assert instance_registry._queue_info_type == kind
        got = client.get("/v1/strategy/get").json()
        assert got["strategy_info"]["strategy_name"] == strategy
    finally:
        api_module.scheduling_strategy, queue_kind = saved
        instance_registry._queue_info_type = queue_kind


@pytest.mark.parametrize("strategy", ["po2", "serverless"])
def test_startup_strategy_gets_its_queue_kind(strategy: str) -> None:
    # Before the fix only min_time started with expect-error sketches, so
    # po2 started as random and serverless ignored queues entirely.
    code = (
        "from swarmpilot.scheduler.api import instance_registry;"
        "print(instance_registry._queue_info_type)"
    )
    env = dict(os.environ, SCHEDULING_STRATEGY=strategy)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "expect_error"
