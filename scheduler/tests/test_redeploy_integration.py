"""Integration tests for instance redeployment functionality.

Tests the complete redeployment workflow including task extraction,
redistribution, and instance status transitions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from swarmpilot.scheduler.api import app, instance_registry, task_registry
from swarmpilot.scheduler.models import InstanceStatus

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset registries before each test."""
    import swarmpilot.scheduler.api as api_module
    from swarmpilot.scheduler.algorithms import get_strategy
    from swarmpilot.scheduler.api import predictor_client

    # Clear registries
    instance_registry._instances.clear()
    instance_registry._queue_info.clear()
    instance_registry._stats.clear()
    task_registry._tasks.clear()

    # Reset instance registry queue type to probabilistic (default)
    instance_registry._queue_info_type = "probabilistic"

    # Reset scheduling strategy to default (probabilistic)
    api_module.scheduling_strategy = get_strategy(
        strategy_name="probabilistic",
        predictor_client=predictor_client,
        instance_registry=instance_registry,
        target_quantile=0.9,
    )

    yield


@pytest.fixture
def client():
    """Test client for API endpoints."""
    return TestClient(app)


# ============================================================================
# Tests
# ============================================================================


def test_redeploy_start_endpoint_success(client):
    """Test successful redeployment initiation."""
    # Register two instances (so tasks can be redistributed)
    instance1_data = {
        "instance_id": "instance-1",
        "model_id": "model-a",
        "endpoint": "http://instance1:8001",
        "platform_info": {
            "software_name": "pytorch",
            "software_version": "2.0",
            "hardware_name": "gpu-v100",
        },
    }
    instance2_data = {
        "instance_id": "instance-2",
        "model_id": "model-a",
        "endpoint": "http://instance2:8001",
        "platform_info": {
            "software_name": "pytorch",
            "software_version": "2.0",
            "hardware_name": "gpu-v100",
        },
    }

    # Register both instances
    response1 = client.post("/v1/instance/register", json=instance1_data)
    assert response1.status_code == 200
    response2 = client.post("/v1/instance/register", json=instance2_data)
    assert response2.status_code == 200

    # Mock the HTTP call to instance's /redeploy/start endpoint
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Mock successful response from instance
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "returned_tasks": [
                {
                    "task_id": "task-1",
                    "model_id": "model-a",
                    "task_input": {"data": "test"},
                    "enqueue_time": 1234567890.0,
                    "metadata": {},
                }
            ],
            "current_task": None,
            "estimated_completion_ms": None,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Mock scheduling_strategy.schedule_task + worker_queue_manager
        from swarmpilot.scheduler.algorithms import ScheduleResult

        mock_result = ScheduleResult(
            selected_instance_id="instance-2",
            selected_prediction=None,
        )
        with (
            patch(
                "swarmpilot.scheduler.api.scheduling_strategy.schedule_task",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "swarmpilot.scheduler.api.worker_queue_manager.enqueue_task",
                return_value=1,
            ),
        ):
            # Start redeployment on instance-1
            redeploy_request = {
                "instance_id": "instance-1",
                "redeploy_reason": "Testing redeployment",
                "target_model_id": "model-b",
            }
            response = client.post("/v1/instance/redeploy/start", json=redeploy_request)

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "redeploying" in data["message"].lower()
            assert len(data["returned_tasks"]) == 1
            assert len(data["redistributed_tasks"]) == 1
            assert len(data["failed_redistributions"]) == 0

            # Verify instance status changed to REDEPLOYING
            instance = asyncio.run(instance_registry.get("instance-1"))
            assert instance.status == InstanceStatus.REDEPLOYING


def test_redeploy_start_instance_not_found(client):
    """Test redeployment fails when instance doesn't exist."""
    redeploy_request = {
        "instance_id": "nonexistent-instance",
        "redeploy_reason": "Testing",
    }
    response = client.post("/v1/instance/redeploy/start", json=redeploy_request)

    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["success"] is False
    assert "not found" in data["detail"]["error"].lower()


def test_redeploy_start_instance_not_active(client):
    """Test redeployment fails when instance is not ACTIVE."""
    # Register instance
    instance_data = {
        "instance_id": "instance-1",
        "model_id": "model-a",
        "endpoint": "http://instance1:8001",
        "platform_info": {
            "software_name": "pytorch",
            "software_version": "2.0",
            "hardware_name": "gpu-v100",
        },
    }
    client.post("/v1/instance/register", json=instance_data)

    # Set instance to DRAINING status
    asyncio.run(instance_registry.update_status("instance-1", InstanceStatus.DRAINING))

    # Try to start redeployment
    redeploy_request = {
        "instance_id": "instance-1",
        "redeploy_reason": "Testing",
    }
    response = client.post("/v1/instance/redeploy/start", json=redeploy_request)

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["success"] is False
    assert "must be in active state" in data["detail"]["error"].lower()


def test_redeploy_complete_endpoint_success(client):
    """Test successful redeployment completion."""
    # Register instance
    instance_data = {
        "instance_id": "instance-1",
        "model_id": "model-a",
        "endpoint": "http://instance1:8001",
        "platform_info": {
            "software_name": "pytorch",
            "software_version": "2.0",
            "hardware_name": "gpu-v100",
        },
    }
    client.post("/v1/instance/register", json=instance_data)

    # Set instance to REDEPLOYING status
    asyncio.run(
        instance_registry.update_status("instance-1", InstanceStatus.REDEPLOYING)
    )

    # Complete redeployment with updated configuration
    complete_request = {
        "instance_id": "instance-1",
        "model_id": "model-b",  # Changed model
        "endpoint": "http://instance1:8001",
        "platform_info": {
            "software_name": "pytorch",
            "software_version": "2.1",  # Updated version
            "hardware_name": "gpu-v100",
        },
    }
    response = client.post("/v1/instance/redeploy/complete", json=complete_request)

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "completed successfully" in data["message"].lower()
    assert data["instance"]["status"] == "active"
    assert data["instance"]["model_id"] == "model-b"

    # Verify instance status changed to ACTIVE
    instance = asyncio.run(instance_registry.get("instance-1"))
    assert instance.status == InstanceStatus.ACTIVE
    assert instance.model_id == "model-b"


def test_redeploy_complete_instance_not_redeploying(client):
    """Test completion fails when instance is not REDEPLOYING."""
    # Register instance
    instance_data = {
        "instance_id": "instance-1",
        "model_id": "model-a",
        "endpoint": "http://instance1:8001",
        "platform_info": {
            "software_name": "pytorch",
            "software_version": "2.0",
            "hardware_name": "gpu-v100",
        },
    }
    client.post("/v1/instance/register", json=instance_data)

    # Instance is ACTIVE, not REDEPLOYING
    complete_request = {
        "instance_id": "instance-1",
        "model_id": "model-a",
        "endpoint": "http://instance1:8001",
        "platform_info": {
            "software_name": "pytorch",
            "software_version": "2.0",
            "hardware_name": "gpu-v100",
        },
    }
    response = client.post("/v1/instance/redeploy/complete", json=complete_request)

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["success"] is False
    assert "must be in redeploying state" in data["detail"]["error"].lower()


def test_redeploy_task_redistribution_preserves_priority(client):
    """Test that task redistribution preserves enqueue_time for priority ordering."""
    # Register two instances
    for i in range(1, 3):
        instance_data = {
            "instance_id": f"instance-{i}",
            "model_id": "model-a",
            "endpoint": f"http://instance{i}:8001",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "gpu-v100",
            },
        }
        client.post("/v1/instance/register", json=instance_data)

    # Mock instance response with tasks that have different enqueue_times
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "returned_tasks": [
                {
                    "task_id": "task-1",
                    "model_id": "model-a",
                    "task_input": {"data": "test1"},
                    "enqueue_time": 1000.0,  # Earlier
                    "metadata": {},
                },
                {
                    "task_id": "task-2",
                    "model_id": "model-a",
                    "task_input": {"data": "test2"},
                    "enqueue_time": 2000.0,  # Later
                    "metadata": {},
                },
            ],
            "current_task": None,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Track the tasks enqueued to worker_queue_manager
        enqueued_tasks = []

        def mock_enqueue_task(worker_id, task):
            enqueued_tasks.append(task)
            return 1

        from swarmpilot.scheduler.algorithms import ScheduleResult

        mock_result = ScheduleResult(
            selected_instance_id="instance-2",
            selected_prediction=None,
        )

        with (
            patch(
                "swarmpilot.scheduler.api.scheduling_strategy.schedule_task",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "swarmpilot.scheduler.api.worker_queue_manager.enqueue_task",
                side_effect=mock_enqueue_task,
            ),
        ):
            # Start redeployment
            redeploy_request = {
                "instance_id": "instance-1",
                "redeploy_reason": "Testing priority preservation",
            }
            response = client.post("/v1/instance/redeploy/start", json=redeploy_request)

            assert response.status_code == 200
            # Verify enqueue_times were preserved
            assert len(enqueued_tasks) == 2
            assert enqueued_tasks[0].enqueue_time == 1000.0
            assert enqueued_tasks[1].enqueue_time == 2000.0
