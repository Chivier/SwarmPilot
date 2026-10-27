"""Free GPUs an elastic auto-optimizer can grow into."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from swarmpilot.planner.pylet.deployment_service import (
    PyLetDeploymentService,
)
from swarmpilot.planner.pylet.instance_manager import InstanceManager


def _worker(free: int, status: str = "ONLINE"):
    return SimpleNamespace(
        id="w", host="h", status=status, gpu_indices_available=list(range(free))
    )


def test_free_gpus_counts_online_workers_only():
    manager = InstanceManager.__new__(InstanceManager)
    manager.pylet_client = MagicMock()
    manager.pylet_client.list_workers.return_value = [
        _worker(3),
        _worker(1),
        _worker(4, status="OFFLINE"),
    ]
    assert manager.free_gpus() == 4


def test_spare_capacity_is_in_default_gpu_count_units():
    with patch("swarmpilot.planner.pylet.deployment_service.InstanceManager"):
        service = PyLetDeploymentService(
            "http://pylet:1", "http://sched:1", default_gpu_count=2
        )
    service._initialized = True
    service._instance_manager.free_gpus.return_value = 5
    assert service.spare_capacity() == 2
