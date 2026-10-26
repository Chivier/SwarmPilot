"""Reconciles run one at a time (auto-optimizer thread vs. SDK calls)."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pylet")

from swarmpilot.planner.pylet.deployment_service import (
    PyLetDeploymentService,
)


@patch("swarmpilot.planner.pylet.deployment_service.InstanceManager")
def test_concurrent_reconciles_do_not_overlap(_manager):
    service = PyLetDeploymentService(
        pylet_head_url="http://localhost:8000",
        scheduler_url="http://localhost:8001",
    )
    service._initialized = True
    running = 0
    peak = 0
    guard = threading.Lock()

    def work(**_):
        nonlocal running, peak
        with guard:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with guard:
            running -= 1
        return MagicMock(success=True)

    executor = MagicMock()
    executor.reconcile.side_effect = work
    executor.scale_model.side_effect = work
    service._deployment_executor = executor

    calls = [
        lambda: service.apply_deployment({"a": 2}),
        lambda: service.scale_model("a", 3),
        lambda: service.apply_deployment({"a": 1, "b": 1}),
    ]
    threads = [threading.Thread(target=c) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert executor.reconcile.call_count == 2
    assert executor.scale_model.call_count == 1
    assert peak == 1
