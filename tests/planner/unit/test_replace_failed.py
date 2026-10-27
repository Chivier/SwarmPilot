"""Failed instances are replaced, within the last requested replica count."""

import time
from unittest.mock import MagicMock, patch

from swarmpilot.planner.pylet.deployment_service import (
    PyLetDeploymentService,
)
from swarmpilot.planner.pylet.instance_manager import (
    ManagedInstance,
    ManagedInstanceStatus,
)


def _service(live: int, lost: int):
    with patch("swarmpilot.planner.pylet.deployment_service.InstanceManager"):
        service = PyLetDeploymentService("http://pylet:1", "http://sched:1")
    service._initialized = True
    manager = service._instance_manager
    manager.get_instances_by_model.return_value = [
        ManagedInstance(
            pylet_id=f"p{i}",
            instance_id=f"p{i}",
            model_id="m",
            status=ManagedInstanceStatus.ACTIVE,
        )
        for i in range(live)
    ]
    manager.reap_failed.return_value = [
        ManagedInstance(pylet_id=f"x{i}", instance_id=f"x{i}", model_id="m")
        for i in range(lost)
    ]
    service._deployment_executor = MagicMock()
    service._deployment_executor.reconcile.return_value = MagicMock(
        success=True
    )
    return service


def _wait_replaced(service, timeout=2.0):
    deadline = time.monotonic() + timeout
    executor = service._deployment_executor
    while not executor.execute.called and time.monotonic() < deadline:
        time.sleep(0.01)
    return executor.execute


def test_replaces_lost_instances():
    service = _service(live=3, lost=1)
    service.apply_deployment({"m": 4})
    service.check_instances()
    execute = _wait_replaced(service)
    assert execute.call_args.args == ({"m": 3}, {"m": 4})


def test_never_beyond_requested_count():
    # Scaled down to 3 after the failure: nothing to add.
    service = _service(live=3, lost=1)
    service.scale_model("m", 3)
    service.check_instances()
    assert not _wait_replaced(service, timeout=0.3).called


def test_unknown_model_gets_what_it_lost():
    service = _service(live=1, lost=2)
    service.check_instances()
    execute = _wait_replaced(service)
    assert execute.call_args.args == ({"m": 1}, {"m": 3})


def test_terminated_model_is_not_replaced():
    service = _service(live=0, lost=1)
    service.apply_deployment({"m": 2})
    service._instance_manager.terminate_instances.return_value = {}
    service.terminate_model("m")
    service.check_instances()
    assert not _wait_replaced(service, timeout=0.3).called


def test_replacement_can_be_turned_off():
    service = _service(live=3, lost=1)
    service.apply_deployment({"m": 4})
    service._replace_failed = False
    assert len(service.check_instances()) == 1
    assert not _wait_replaced(service, timeout=0.3).called
