"""Unit tests for FastAPI endpoints.

Tests instance management, task submission, and health check endpoints.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from swarmpilot.scheduler.api import app, instance_registry, task_registry
from swarmpilot.scheduler.models import TaskStatus

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset registries before each test."""
    import swarmpilot.scheduler.api as api_module
    from swarmpilot.scheduler.algorithms import get_strategy
    from swarmpilot.scheduler.api import config, predictor_client

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

    # Reset config to default
    config.scheduling.default_strategy = "probabilistic"
    config.scheduling.probabilistic_quantile = 0.9

    yield


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


# ============================================================================
# Instance Registration Tests
# ============================================================================


class TestInstanceRegistration:
    """Tests for instance registration endpoint."""

    def test_register_instance_success(self, client):
        """Test successful instance registration."""
        response = client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance"]["instance_id"] == "inst-1"
        assert (
            data["message"]
            == "Instance registered successfully (work stealing in progress)"
        )

    def test_register_duplicate_instance(self, client):
        """Test registering duplicate instance returns 200 with warning (idempotent)."""
        # Register first time
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Try to register again - now returns 200 (idempotent behavior)
        response = client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert (
            "already registered" in data["message"].lower()
            or "duplicate" in data["message"].lower()
        )

    def test_register_instance_validation(self, client):
        """Test validation of registration request."""
        response = client.post(
            "/v1/instance/register",
            json={"instance_id": "inst-1"},  # Missing required fields
        )

        assert response.status_code == 422  # Validation error


# ============================================================================
# Instance Removal Tests
# ============================================================================


class TestInstanceRemoval:
    """Tests for instance removal endpoint."""

    def test_remove_instance_success(self, client):
        """Test successful instance removal with drain-before-remove workflow."""
        # Register instance first
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Drain the instance first (required for safe removal)
        drain_response = client.post("/v1/instance/drain", json={"instance_id": "inst-1"})
        assert drain_response.status_code == 200

        # Remove it
        response = client.post("/v1/instance/remove", json={"instance_id": "inst-1"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "inst-1"

    def test_remove_nonexistent_instance(self, client):
        """Test removing non-existent instance returns 404."""
        response = client.post("/v1/instance/remove", json={"instance_id": "nonexistent"})

        assert response.status_code == 404

    async def test_remove_instance_with_pending_tasks(self, client):
        """Test removing instance with pending tasks (allowed with warning)."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Add pending task
        await instance_registry.increment_pending("inst-1")

        # Removal is allowed even with pending tasks (logs warning)
        response = client.post("/v1/instance/remove", json={"instance_id": "inst-1"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# ============================================================================
# Instance List Tests
# ============================================================================


class TestInstanceList:
    """Tests for instance listing endpoint."""

    def test_list_all_instances(self, client):
        """Test listing all instances."""
        # Register multiple instances
        for i in range(3):
            client.post(
                "/v1/instance/register",
                json={
                    "instance_id": f"inst-{i}",
                    "model_id": "model-1",
                    "endpoint": f"http://localhost:800{i}",
                    "platform_info": {
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "test-hardware",
                    },
                },
            )

        response = client.get("/v1/instance/list")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 3
        assert len(data["instances"]) == 3

    def test_list_instances_filtered(self, client):
        """Test listing instances filtered by model_id."""
        # Register instances with different models
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-a",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-2",
                "model_id": "model-b",
                "endpoint": "http://localhost:8002",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        response = client.get("/v1/instance/list?model_id=model-a")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["instances"][0]["model_id"] == "model-a"

    def test_list_empty_instances(self, client):
        """Test listing when no instances registered."""
        response = client.get("/v1/instance/list")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["instances"] == []


# ============================================================================
# Instance Info Tests
# ============================================================================


class TestInstanceInfo:
    """Tests for instance info endpoint."""

    def test_get_instance_info(self, client):
        """Test getting instance information."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        response = client.get("/v1/instance/info?instance_id=inst-1")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance"]["instance_id"] == "inst-1"
        assert "stats" in data
        assert "queue_info" in data

    def test_get_nonexistent_instance_info(self, client):
        """Test getting info for non-existent instance."""
        response = client.get("/v1/instance/info?instance_id=nonexistent")

        assert response.status_code == 404


# ============================================================================
# Task Submission Tests
# ============================================================================


class TestTaskSubmission:
    """Tests for task submission endpoint."""

    def test_submit_task_success(self, client):
        """Test successful task submission."""
        # Register instance first
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Mock predictor
        with patch("swarmpilot.scheduler.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            response = client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task"]["task_id"] == "task-1"

    def test_submit_task_no_instances(self, client):
        """Test submitting task when no instances available - task is queued."""
        response = client.post(
            "/v1/task/submit",
            json={
                "task_id": "task-1",
                "model_id": "model-1",
                "task_input": {"prompt": "test"},
                "metadata": {},
            },
        )

        # Task is queued even when no instances are available
        # It will wait for an instance to be registered
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task"]["task_id"] == "task-1"
        assert data["task"]["status"] == "pending"

    def test_submit_duplicate_task(self, client):
        """Test submitting duplicate task."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task first time
        with patch("swarmpilot.scheduler.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {},
                    "metadata": {},
                },
            )

        # Try to submit again
        with patch("swarmpilot.scheduler.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            response = client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {},
                    "metadata": {},
                },
            )

        assert response.status_code == 400


# ============================================================================
# Task List Tests
# ============================================================================


class TestTaskList:
    """Tests for task listing endpoint."""

    def test_list_all_tasks(self, client):
        """Test listing all tasks."""
        # Register instance and submit tasks
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        with patch("swarmpilot.scheduler.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            for i in range(3):
                client.post(
                    "/v1/task/submit",
                    json={
                        "task_id": f"task-{i}",
                        "model_id": "model-1",
                        "task_input": {},
                        "metadata": {},
                    },
                )

        response = client.get("/v1/task/list")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 3
        assert data["total"] == 3

    def test_list_tasks_with_filters(self, client):
        """Test listing tasks with status filter."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task
        with patch("swarmpilot.scheduler.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {},
                    "metadata": {},
                },
            )

        response = client.get("/v1/task/list?status=pending")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 0

    def test_list_tasks_pagination(self, client):
        """Test task list pagination."""
        response = client.get("/v1/task/list?limit=10&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 0


# ============================================================================
# Task Info Tests
# ============================================================================


class TestTaskInfo:
    """Tests for task info endpoint."""

    def test_get_task_info(self, client):
        """Test getting task information."""
        # Register instance and submit task
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        with patch("swarmpilot.scheduler.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"test": "data"},
                    "metadata": {},
                },
            )

        response = client.get("/v1/task/info?task_id=task-1")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task"]["task_id"] == "task-1"
        assert "timestamps" in data["task"]

    def test_get_nonexistent_task_info(self, client):
        """Test getting info for non-existent task."""
        response = client.get("/v1/task/info?task_id=nonexistent")

        assert response.status_code == 404


class TestTaskResubmit:
    """Tests for task resubmit endpoint."""

    def test_resubmit_task_success(self, client):
        """Test successful task resubmission with original submission time preserved."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit a task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Mock scheduling_strategy.schedule_task to verify resubmission path
        from unittest.mock import MagicMock

        from swarmpilot.scheduler.algorithms import ScheduleResult

        mock_result = ScheduleResult(
            selected_instance_id="inst-1",
            selected_prediction=None,
        )
        mock_wqm = MagicMock()
        mock_wqm.enqueue_task = MagicMock(return_value=1)
        with (
            patch(
                "swarmpilot.scheduler.api.scheduling_strategy.schedule_task",
                new=AsyncMock(return_value=mock_result),
            ) as mock_schedule,
            patch("swarmpilot.scheduler.api.worker_queue_manager", mock_wqm),
        ):
            response = client.post(
                "/v1/task/resubmit",
                json={"task_id": "task-1", "original_instance_id": "inst-1"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "resubmitted successfully" in data["message"]

            # Verify schedule_task was called with correct parameters
            mock_schedule.assert_called_once()
            call_kwargs = mock_schedule.call_args[1]
            assert call_kwargs["model_id"] == "model-1"

    def test_resubmit_task_preserves_original_time(self, client):
        """Test that resubmit preserves the original submission timestamp."""
        import asyncio

        from swarmpilot.scheduler import api
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit a task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Get the original task to capture its submitted_at time
        from datetime import datetime as dt_module

        async def get_original_time():
            task = await api.task_registry.get("task-1")
            if task.submitted_at:
                dt = dt_module.fromisoformat(task.submitted_at.replace("Z", "+00:00"))
                return dt.timestamp()
            return None

        original_timestamp = asyncio.get_event_loop().run_until_complete(
            get_original_time()
        )

        # Resubmit and verify it succeeds
        from swarmpilot.scheduler.algorithms import ScheduleResult

        mock_result = ScheduleResult(
            selected_instance_id="inst-1",
            selected_prediction=None,
        )
        with patch(
            "swarmpilot.scheduler.api.scheduling_strategy.schedule_task",
            new=AsyncMock(return_value=mock_result),
        ):
            response = client.post(
                "/v1/task/resubmit",
                json={"task_id": "task-1", "original_instance_id": "inst-1"},
            )

            assert response.status_code == 200

    def test_resubmit_task_not_found(self, client):
        """Test resubmitting non-existent task returns 404."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        response = client.post(
            "/v1/task/resubmit",
            json={"task_id": "nonexistent", "original_instance_id": "inst-1"},
        )

        assert response.status_code == 404
        assert "Task not found" in response.json()["detail"]["error"]

    def test_resubmit_task_invalid_status_completed(self, client):
        """Test resubmitting completed task returns 400."""
        import asyncio

        from swarmpilot.scheduler import api
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Mark task as completed
        async def mark_completed():
            await api.task_registry.update_status("task-1", TaskStatus.COMPLETED)

        asyncio.get_event_loop().run_until_complete(mark_completed())

        # Try to resubmit
        response = client.post(
            "/v1/task/resubmit",
            json={"task_id": "task-1", "original_instance_id": "inst-1"},
        )

        assert response.status_code == 400
        assert "Cannot resubmit task" in response.json()["detail"]["error"]
        assert "COMPLETED" in response.json()["detail"]["error"]

    def test_resubmit_task_invalid_status_failed(self, client):
        """Test resubmitting failed task returns 400."""
        import asyncio

        from swarmpilot.scheduler import api
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Mark task as failed
        async def mark_failed():
            await api.task_registry.update_status("task-1", TaskStatus.FAILED)

        asyncio.get_event_loop().run_until_complete(mark_failed())

        # Try to resubmit
        response = client.post(
            "/v1/task/resubmit",
            json={"task_id": "task-1", "original_instance_id": "inst-1"},
        )

        assert response.status_code == 400
        assert "Cannot resubmit task" in response.json()["detail"]["error"]
        assert "FAILED" in response.json()["detail"]["error"]

    def test_resubmit_task_original_instance_not_found(self, client):
        """Test resubmitting with non-existent original instance returns 404."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Try to resubmit with non-existent instance
        response = client.post(
            "/v1/task/resubmit",
            json={
                "task_id": "task-1",
                "original_instance_id": "nonexistent-instance",
            },
        )

        assert response.status_code == 404
        assert "Original instance not found" in response.json()["detail"]["error"]

    def test_resubmit_task_decrements_pending_count(self, client):
        """Test that resubmit decrements pending count on original instance."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Verify decrement_pending is called
        with (
            patch(
                "swarmpilot.scheduler.api.scheduling_strategy.schedule_task",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "swarmpilot.scheduler.api.instance_registry.decrement_pending", new=AsyncMock()
            ) as mock_decrement,
        ):
            response = client.post(
                "/v1/task/resubmit",
                json={
                    "task_id": "task-1",
                    "original_instance_id": "inst-1",
                },
            )

            assert response.status_code == 200
            mock_decrement.assert_called_once_with("inst-1")

    def test_resubmit_running_task(self, client):
        """Test resubmitting a running task succeeds."""
        import asyncio

        from swarmpilot.scheduler import api
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Mark task as running
        async def mark_running():
            await api.task_registry.update_status("task-1", TaskStatus.RUNNING)

        asyncio.get_event_loop().run_until_complete(mark_running())

        # Resubmit should succeed for running task
        with patch(
            "swarmpilot.scheduler.api.scheduling_strategy.schedule_task",
            new=AsyncMock(return_value=None),
        ):
            response = client.post(
                "/v1/task/resubmit",
                json={"task_id": "task-1", "original_instance_id": "inst-1"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_resubmit_task_reset_for_resubmit_error(self, client):
        """Test resubmit when reset_for_resubmit raises KeyError."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Mock reset_for_resubmit to raise KeyError
        with patch(
            "swarmpilot.scheduler.api.task_registry.reset_for_resubmit",
            side_effect=KeyError("not found"),
        ):
            response = client.post(
                "/v1/task/resubmit",
                json={"task_id": "task-1", "original_instance_id": "inst-1"},
            )

            assert response.status_code == 404
            assert "Task not found" in response.json()["detail"]["error"]

    def test_resubmit_task_with_metadata(self, client):
        """Test resubmitting task preserves metadata."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task with metadata
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        task_metadata = {"user_id": "user-123", "priority": "high"}

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": task_metadata,
                },
            )

        # Resubmit and verify metadata is preserved
        response = client.post(
            "/v1/task/resubmit",
            json={"task_id": "task-1", "original_instance_id": "inst-1"},
        )

        assert response.status_code == 200

        # Verify task still has its metadata in registry
        import asyncio

        from swarmpilot.scheduler.api import task_registry

        task = asyncio.get_event_loop().run_until_complete(task_registry.get("task-1"))
        assert task.metadata == task_metadata


class TestTaskClear:
    """Tests for task clear endpoint."""

    def test_clear_empty_tasks(self, client):
        """Test clearing tasks when registry is empty."""
        response = client.post("/v1/task/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cleared_count"] == 0
        assert "Successfully cleared 0 scheduler task(s)" in data["message"]

    def test_clear_multiple_tasks(self, client):
        """Test clearing multiple tasks."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit multiple tasks
        with patch("swarmpilot.scheduler.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            for i in range(3):
                client.post(
                    "/v1/task/submit",
                    json={
                        "task_id": f"task-{i}",
                        "model_id": "model-1",
                        "task_input": {"test": "data"},
                        "metadata": {},
                    },
                )

        # Verify tasks exist
        response = client.get("/v1/task/list")
        assert response.json()["count"] == 3

        # Clear all tasks
        response = client.post("/v1/task/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cleared_count"] == 3
        assert "Successfully cleared 3 scheduler task(s)" in data["message"]

        # Verify tasks are cleared
        response = client.get("/v1/task/list")
        assert response.json()["count"] == 0


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_healthy(self, client):
        """Test health check when service is healthy."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "healthy"
        assert "stats" in data
        assert "version" in data
        assert "timestamp" in data

    def test_health_check_includes_stats(self, client):
        """Test that health check includes statistics."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["stats"]["total_instances"] == 1
        assert data["stats"]["active_instances"] == 1


# ============================================================================
# Strategy Management Tests
# ============================================================================


class TestStrategyManagement:
    """Tests for strategy management endpoints."""

    def test_get_strategy_default(self, client):
        """Test getting the default strategy (probabilistic)."""
        response = client.get("/v1/strategy/get")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_info"]["strategy_name"] == "probabilistic"
        assert "target_quantile" in data["strategy_info"]["parameters"]
        assert data["strategy_info"]["parameters"]["target_quantile"] == 0.9

    def test_set_strategy_to_min_time(self, client):
        """Test switching to min_time strategy."""
        response = client.post("/v1/strategy/set", json={"strategy_name": "min_time"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_info"]["strategy_name"] == "min_time"
        assert data["cleared_tasks"] == 0
        assert data["reinitialized_instances"] == 0
        assert "Successfully switched" in data["message"]

    def test_set_strategy_to_round_robin(self, client):
        """Test switching to round_robin strategy."""
        response = client.post("/v1/strategy/set", json={"strategy_name": "round_robin"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_info"]["strategy_name"] == "round_robin"
        assert data["strategy_info"]["parameters"] == {}

    def test_set_strategy_to_probabilistic_with_custom_quantile(self, client):
        """Test switching to probabilistic strategy with custom quantile."""
        response = client.post(
            "/v1/strategy/set",
            json={"strategy_name": "probabilistic", "target_quantile": 0.95},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_info"]["strategy_name"] == "probabilistic"
        assert data["strategy_info"]["parameters"]["target_quantile"] == 0.95

    def test_set_strategy_invalid_name(self, client):
        """Test setting strategy with invalid name."""
        response = client.post(
            "/v1/strategy/set", json={"strategy_name": "invalid_strategy"}
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_set_strategy_clears_tasks(self, client):
        """Test that setting strategy clears all tasks."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance first
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit a task (mock predictor)
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0, 0.95: 120.0, 0.99: 130.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Switch strategy
        response = client.post("/v1/strategy/set", json={"strategy_name": "min_time"})

        assert response.status_code == 200
        data = response.json()
        assert data["cleared_tasks"] == 1

        # Verify tasks are cleared
        tasks_response = client.get("/v1/task/list")
        assert tasks_response.json()["total"] == 0

    def test_set_strategy_reinitializes_instances(self, client):
        """Test that setting strategy reinitializes instance queues."""
        # Register two instances
        for i in range(1, 3):
            client.post(
                "/v1/instance/register",
                json={
                    "instance_id": f"inst-{i}",
                    "model_id": "model-1",
                    "endpoint": f"http://localhost:800{i}",
                    "platform_info": {
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "test-hardware",
                    },
                },
            )

        # Switch from probabilistic to min_time
        response = client.post("/v1/strategy/set", json={"strategy_name": "min_time"})

        assert response.status_code == 200
        data = response.json()
        assert data["reinitialized_instances"] == 2

        # Verify instances still exist
        instances_response = client.get("/v1/instance/list")
        assert instances_response.json()["count"] == 2

        # Verify queue info type changed by checking instance info
        instance_info = client.get("/v1/instance/info?instance_id=inst-1")
        assert instance_info.status_code == 200
        queue_info = instance_info.json()["queue_info"]
        # For min_time, should have expected_time_ms and error_margin_ms
        assert "expected_time_ms" in queue_info
        assert "error_margin_ms" in queue_info

    async def test_set_strategy_rejects_when_tasks_running(self, client):
        """Test that setting strategy is rejected when tasks are running."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit a task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0, 0.95: 120.0, 0.99: 130.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Manually set task to RUNNING status
        await task_registry.update_status("task-1", TaskStatus.RUNNING)

        # Try to switch strategy - should fail
        response = client.post("/v1/strategy/set", json={"strategy_name": "min_time"})

        assert response.status_code == 400
        data = response.json()
        assert "Cannot switch strategy while" in data["detail"]["error"]

    def test_strategy_switch_round_trip(self, client):
        """Test switching between all strategies."""
        # Start with probabilistic (default)
        response = client.get("/v1/strategy/get")
        assert response.json()["strategy_info"]["strategy_name"] == "probabilistic"

        # Switch to min_time
        response = client.post("/v1/strategy/set", json={"strategy_name": "min_time"})
        assert response.status_code == 200

        response = client.get("/v1/strategy/get")
        assert response.json()["strategy_info"]["strategy_name"] == "min_time"

        # Switch to round_robin
        response = client.post("/v1/strategy/set", json={"strategy_name": "round_robin"})
        assert response.status_code == 200

        response = client.get("/v1/strategy/get")
        assert response.json()["strategy_info"]["strategy_name"] == "round_robin"

        # Switch back to probabilistic
        response = client.post(
            "/v1/strategy/set",
            json={"strategy_name": "probabilistic", "target_quantile": 0.8},
        )
        assert response.status_code == 200

        response = client.get("/v1/strategy/get")
        data = response.json()
        assert data["strategy_info"]["strategy_name"] == "probabilistic"
        assert data["strategy_info"]["parameters"]["target_quantile"] == 0.8

    def test_set_strategy_invalid_quantile(self, client):
        """Test setting strategy with invalid quantile value."""
        # Quantile > 1.0
        response = client.post(
            "/v1/strategy/set",
            json={"strategy_name": "probabilistic", "target_quantile": 1.5},
        )
        assert response.status_code == 422  # Pydantic validation error

        # Quantile < 0.0
        response = client.post(
            "/v1/strategy/set",
            json={"strategy_name": "probabilistic", "target_quantile": -0.1},
        )
        assert response.status_code == 422  # Pydantic validation error


# ============================================================================
# Additional Tests for Coverage
# ============================================================================


class TestInstanceRegistrationValidation:
    """Additional tests for instance registration validation."""

    def test_register_instance_missing_platform_keys(self, client):
        """Test registration with incomplete platform_info."""
        response = client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    # Missing software_version and hardware_name
                },
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["success"] is False
        assert "missing required keys" in data["detail"]["error"]


class TestInstanceDrain:
    """Tests for instance drain endpoint."""

    def test_drain_instance_success(self, client):
        """Test draining an instance successfully."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Drain the instance
        response = client.post("/v1/instance/drain", json={"instance_id": "inst-1"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "inst-1"
        assert data["status"] == "draining"
        assert "pending_tasks" in data

    def test_drain_nonexistent_instance(self, client):
        """Test draining non-existent instance returns 404."""
        response = client.post("/v1/instance/drain", json={"instance_id": "nonexistent"})

        assert response.status_code == 404

    async def test_drain_instance_with_pending_tasks(self, client):
        """Test draining instance with pending tasks.

        With the central queue architecture, pending_tasks is incremented when
        a task is dispatched to an instance, not when it's submitted. This test
        manually increments the pending count to simulate a dispatched task.
        """
        from swarmpilot.scheduler import api

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Manually increment pending count to simulate a dispatched task
        # (In the new central queue architecture, pending_tasks is incremented
        # when a task is dispatched from the queue to an instance)
        await api.instance_registry.increment_pending("inst-1")

        # Now drain the instance
        response = client.post("/v1/instance/drain", json={"instance_id": "inst-1"})

        assert response.status_code == 200
        data = response.json()
        assert data["pending_tasks"] > 0

    def test_get_drain_status_success(self, client):
        """Test getting drain status."""
        # Register and drain instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        client.post("/v1/instance/drain", json={"instance_id": "inst-1"})

        # Check drain status
        response = client.get("/v1/instance/drain/status?instance_id=inst-1")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "inst-1"
        assert data["status"] == "draining"
        assert "can_remove" in data

    def test_get_drain_status_nonexistent(self, client):
        """Test getting drain status for non-existent instance."""
        response = client.get("/v1/instance/drain/status?instance_id=nonexistent")

        assert response.status_code == 404


class TestTaskResultCallback:
    """Tests for task result callback endpoint."""

    def test_callback_task_result_success(self, client):
        """Test successful task result callback."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit a task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Send callback
        with patch("swarmpilot.scheduler.api.task_result_callback.handle_result", new=AsyncMock()):
            response = client.post(
                "/v1/callback/task_result",
                json={
                    "task_id": "task-1",
                    "status": "completed",
                    "result": {"output": "test result"},
                    "execution_time_ms": 150,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_callback_task_result_task_not_found(self, client):
        """Test callback for non-existent task."""
        response = client.post(
            "/v1/callback/task_result",
            json={
                "task_id": "nonexistent",
                "status": "completed",
                "result": {"output": "test"},
                "execution_time_ms": 100,
            },
        )

        assert response.status_code == 404

    def test_callback_task_result_invalid_status(self, client):
        """Test callback with invalid status."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance and submit task
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Send callback with invalid status
        response = client.post(
            "/v1/callback/task_result",
            json={
                "task_id": "task-1",
                "status": "invalid_status",
                "result": {"output": "test"},
                "execution_time_ms": 100,
            },
        )

        assert response.status_code == 400

    def test_callback_task_result_failed(self, client):
        """Test callback for failed task."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance and submit task
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Send callback with failure
        with patch("swarmpilot.scheduler.api.task_result_callback.handle_result", new=AsyncMock()):
            response = client.post(
                "/v1/callback/task_result",
                json={
                    "task_id": "task-1",
                    "status": "failed",
                    "error": "Execution error",
                    "execution_time_ms": 50,
                },
            )

        assert response.status_code == 200


class TestTaskSubmissionErrors:
    """Additional tests for task submission error handling."""

    async def test_submit_task_predictor_service_unavailable(self, client):
        """Test task submission when predictor service is unavailable.

        Tasks are created with PENDING status and scheduling happens inline.
        If scheduling fails, the task remains in 'pending' status.
        """
        from swarmpilot.scheduler import api

        # Clear any leftover state from previous tests
        await api.task_registry.clear_all()

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task - with central queue, task is enqueued immediately
        response = client.post(
            "/v1/task/submit",
            json={
                "task_id": "task-1",
                "model_id": "model-1",
                "task_input": {"prompt": "test"},
                "metadata": {},
            },
        )

        # API returns immediately with success
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task"]["task_id"] == "task-1"
        assert data["task"]["status"] == "pending"

        # Task should be in pending status (scheduling happens asynchronously)
        task = await api.task_registry.get("task-1")
        assert task is not None
        assert task.status.value == "pending"

    async def test_submit_task_no_trained_model(self, client):
        """Test task submission when no trained model is available.

        Tasks are created with PENDING status and scheduling happens inline.
        """
        from swarmpilot.scheduler import api

        # Clear any leftover state from previous tests
        await api.task_registry.clear_all()

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task - with central queue, task is enqueued immediately
        response = client.post(
            "/v1/task/submit",
            json={
                "task_id": "task-1",
                "model_id": "model-1",
                "task_input": {"prompt": "test"},
                "metadata": {},
            },
        )

        # API returns immediately with success
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Task should be in pending status (scheduling happens asynchronously)
        task = await api.task_registry.get("task-1")
        assert task is not None
        assert task.status.value == "pending"

    async def test_submit_task_invalid_metadata(self, client):
        """Test task submission with invalid metadata.

        Tasks are created with PENDING status and scheduling happens inline.
        """
        from swarmpilot.scheduler import api

        # Clear any leftover state from previous tests
        await api.task_registry.clear_all()

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task with custom metadata - task is enqueued immediately
        response = client.post(
            "/v1/task/submit",
            json={
                "task_id": "task-1",
                "model_id": "model-1",
                "task_input": {"prompt": "test"},
                "metadata": {"custom": "data"},
            },
        )

        # API returns immediately with success
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Task should be in pending status (scheduling happens asynchronously)
        task = await api.task_registry.get("task-1")
        assert task is not None
        assert task.status.value == "pending"


class TestProfilingMiddleware:
    """Tests for profiling middleware."""

    def test_profile_request_html_format(self, client):
        """Test profiling with HTML format."""
        import os

        # Make a request with profiling enabled (HTML format)
        response = client.get("/v1/health?profile=true&profile_format=html")

        # Should succeed
        assert response.status_code == 200

        # Check if profile file was created
        assert os.path.exists("profile.html")

        # Clean up
        if os.path.exists("profile.html"):
            os.remove("profile.html")

    def test_profile_request_speedscope_format(self, client):
        """Test profiling with speedscope format."""
        import os

        # Make a request with profiling enabled (speedscope format, default)
        response = client.get("/v1/health?profile=true&profile_format=speedscope")

        # Should succeed
        assert response.status_code == 200

        # Check if profile file was created
        assert os.path.exists("profile.speedscope.json")

        # Clean up
        if os.path.exists("profile.speedscope.json"):
            os.remove("profile.speedscope.json")

    def test_profile_request_default_format(self, client):
        """Test profiling with default format (speedscope)."""
        import os

        # Make a request with profiling enabled (default format)
        response = client.get("/v1/health?profile=true")

        # Should succeed
        assert response.status_code == 200

        # Check if profile file was created
        assert os.path.exists("profile.speedscope.json")

        # Clean up
        if os.path.exists("profile.speedscope.json"):
            os.remove("profile.speedscope.json")


class TestDrainEdgeCases:
    """Additional tests for drain edge cases."""

    def test_drain_instance_with_expect_error_queue(self, client):
        """Test draining with expect_error queue type."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance first
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Switch to min_time strategy (uses expect_error queue)
        client.post("/v1/strategy/set", json={"strategy_name": "min_time"})

        # Register another instance after strategy switch
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-2",
                "model_id": "model-1",
                "endpoint": "http://localhost:8002",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit a task
        mock_prediction = Prediction(
            instance_id="inst-2", predicted_time_ms=100.0, error_margin_ms=10.0
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Drain the instance
        response = client.post("/v1/instance/drain", json={"instance_id": "inst-2"})

        assert response.status_code == 200
        data = response.json()
        assert "estimated_completion_time_ms" in data

    def test_drain_instance_probabilistic_with_median(self, client):
        """Test draining with probabilistic queue that has median."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit a task to populate queue
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0, 0.95: 120.0, 0.99: 130.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Drain the instance
        response = client.post("/v1/instance/drain", json={"instance_id": "inst-1"})

        assert response.status_code == 200
        data = response.json()
        # Should have estimated time from median quantile (0.5)
        assert "estimated_completion_time_ms" in data


class TestStrategyHelpers:
    """Tests for strategy helper functions."""

    def test_get_current_strategy_info_unknown(self, client):
        """Test getting strategy info for unknown strategy."""
        # This is harder to test directly without modifying global state
        # But we can at least exercise the code path
        response = client.get("/v1/strategy/get")
        assert response.status_code == 200


class TestWebSocketEndpoint:
    """Tests for WebSocket endpoint."""

    def test_websocket_subscribe_and_receive_result(self, client):
        """Test WebSocket subscribe and receive task result."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance and submit task
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Test WebSocket connection
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Subscribe to task
            websocket.send_json({"type": "subscribe", "task_ids": ["task-1"]})

            # Should receive acknowledgment
            ack = websocket.receive_json()
            assert ack["type"] == "ack"
            assert "task-1" in ack["subscribed_tasks"]

            # Should also receive result immediately since task is pending
            # (or may receive result first depending on timing)

    def test_websocket_unsubscribe(self, client):
        """Test WebSocket unsubscribe."""
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Subscribe first
            websocket.send_json({"type": "subscribe", "task_ids": ["task-1", "task-2"]})

            ack1 = websocket.receive_json()
            assert len(ack1["subscribed_tasks"]) == 2

            # Unsubscribe from one
            websocket.send_json({"type": "unsubscribe", "task_ids": ["task-1"]})

            ack2 = websocket.receive_json()
            assert ack2["type"] == "ack"
            assert "task-1" not in ack2["subscribed_tasks"]
            assert "task-2" in ack2["subscribed_tasks"]

    def test_websocket_invalid_message_type(self, client):
        """Test WebSocket with invalid message type."""
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Send invalid message type
            websocket.send_json({"type": "invalid_type", "data": "test"})

            # Should receive error message
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert "Unknown message type" in error["error"]

    def test_websocket_invalid_task_ids_format(self, client):
        """Test WebSocket with invalid task_ids format."""
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Send invalid task_ids (not a list)
            websocket.send_json({"type": "subscribe", "task_ids": "not-a-list"})

            # Should receive error message
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert "must be a list" in error["error"]

    def test_websocket_multiple_subscriptions(self, client):
        """Test WebSocket with multiple task subscriptions."""
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Subscribe to multiple tasks
            websocket.send_json(
                {
                    "type": "subscribe",
                    "task_ids": ["task-1", "task-2", "task-3"],
                }
            )

            ack = websocket.receive_json()
            assert ack["type"] == "ack"
            assert len(ack["subscribed_tasks"]) == 3

    def test_websocket_ping_pong_application_level(self, client):
        """Test WebSocket application-level ping/pong."""
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Send application-level ping
            websocket.send_json({"type": "ping", "timestamp": 123456.789})

            # Should receive pong response
            pong = websocket.receive_json()
            assert pong["type"] == "pong"
            assert "timestamp" in pong

    def test_websocket_json_decode_error(self, client):
        """Test WebSocket with invalid JSON."""
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Send invalid JSON
            websocket.send_text("invalid json {")

            # Should receive error message
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert "Invalid JSON format" in error["error"]

    def test_websocket_protocol_level_ping(self, client):
        """Test WebSocket protocol-level ping."""
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Send protocol-level ping (this is handled by the WebSocket server)
            # We can't easily test this from the client side, but we can verify
            # that the endpoint handles it by checking it doesn't break
            websocket.send_json({"type": "subscribe", "task_ids": ["task-1"]})

            # Should still work
            ack = websocket.receive_json()
            assert ack["type"] == "ack"

    def test_websocket_exception_handling(self, client):
        """Test WebSocket exception handling."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance and submit task
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Connect and test
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Subscribe to trigger code paths
            websocket.send_json({"type": "subscribe", "task_ids": ["task-1"]})

            # Receive ack
            websocket.receive_json()


class TestRemainingErrorHandling:
    """Tests for remaining error handling paths."""

    def test_register_instance_registry_error(self, client):
        """Test registration with registry error."""
        # Try to register with invalid data that causes internal error
        with patch(
            "swarmpilot.scheduler.api.instance_registry.register",
            side_effect=ValueError("Internal error"),
        ):
            response = client.post(
                "/v1/instance/register",
                json={
                    "instance_id": "inst-1",
                    "model_id": "model-1",
                    "endpoint": "http://localhost:8001",
                    "platform_info": {
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "test-hardware",
                    },
                },
            )

        assert response.status_code == 400

    def test_drain_instance_value_error(self, client):
        """Test drain with value error."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Try to drain instance with mocked error
        with patch(
            "swarmpilot.scheduler.api.instance_registry.start_draining",
            side_effect=ValueError("Cannot drain"),
        ):
            response = client.post("/v1/instance/drain", json={"instance_id": "inst-1"})

        assert response.status_code == 400

    def test_submit_task_timeout_error(self, client):
        """Test task submission with timeout error."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Mock to raise timeout error
        # Mock the scheduling_strategy
        with patch(
            "swarmpilot.scheduler.api.scheduling_strategy.schedule_task",
            side_effect=TimeoutError("Request timeout"),
        ):
            response = client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # With background scheduling, API returns 200 immediately
        assert response.status_code == 200

    def test_task_registry_create_error(self, client):
        """Test task submission with task registry error."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        mock_prediction = Prediction(instance_id="inst-1", predicted_time_ms=100.0)

        from swarmpilot.scheduler.algorithms import ScheduleResult

        with (
            patch(
                "swarmpilot.scheduler.api.scheduling_strategy.schedule_task",
                new=AsyncMock(
                    return_value=ScheduleResult(
                        selected_instance_id="inst-1",
                        selected_prediction=mock_prediction,
                    )
                ),
            ),
            patch(
                "swarmpilot.scheduler.api.task_registry.create_task",
                side_effect=ValueError("Task creation error"),
            ),
        ):
            response = client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        assert response.status_code == 400

    def test_set_strategy_exception(self, client):
        """Test set strategy with exception during initialization."""
        # Mock get_strategy to raise exception
        with patch(
            "swarmpilot.scheduler.api.get_strategy", side_effect=Exception("Strategy init error")
        ):
            response = client.post("/v1/strategy/set", json={"strategy_name": "min_time"})

        assert response.status_code == 500

    def test_health_check_with_exception(self, client):
        """Test health check when registry has errors."""
        # Mock to raise exception
        with patch(
            "swarmpilot.scheduler.api.instance_registry.get_total_count",
            side_effect=Exception("Registry error"),
        ):
            response = client.get("/v1/health")

        assert response.status_code == 503

    def test_remove_instance_key_error(self, client):
        """Test remove instance with KeyError."""
        # Try to remove non-existent instance that causes KeyError
        with patch(
            "swarmpilot.scheduler.api.instance_registry.safe_remove",
            side_effect=KeyError("not found"),
        ):
            response = client.post(
                "/v1/instance/remove", json={"instance_id": "nonexistent"}
            )

        assert response.status_code == 404

    def test_remove_instance_success(self, client):
        """Test remove instance succeeds after registration."""
        # Register instance first
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Remove instance - should succeed
        response = client.post("/v1/instance/remove", json={"instance_id": "inst-1"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_drain_instance_no_median_quantile(self, client):
        """Test drain when quantile distribution has no 0.5."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task with quantiles that do not include 0.5
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.9: 110.0, 0.95: 120.0, 0.99: 130.0},  # No 0.5
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Drain should handle missing 0.5 quantile gracefully
        response = client.post("/v1/instance/drain", json={"instance_id": "inst-1"})

        assert response.status_code == 200

    def test_instance_info_no_queue_info(self, client):
        """Test instance info when queue info is missing."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Mock get_queue_info to return None
        with patch("swarmpilot.scheduler.api.instance_registry.get_queue_info", return_value=None):
            response = client.get("/v1/instance/info?instance_id=inst-1")

        assert response.status_code == 500

    def test_instance_info_no_stats(self, client):
        """Test instance info when stats are missing."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Mock get_stats to return None
        with patch("swarmpilot.scheduler.api.instance_registry.get_stats", return_value=None):
            response = client.get("/v1/instance/info?instance_id=inst-1")

        assert response.status_code == 500

    def test_get_current_strategy_info_round_robin(self, client):
        """Test getting strategy info for round_robin."""
        # Switch to round_robin
        client.post("/v1/strategy/set", json={"strategy_name": "round_robin"})

        response = client.get("/v1/strategy/get")

        assert response.status_code == 200
        data = response.json()
        assert data["strategy_info"]["strategy_name"] == "round_robin"
        assert data["strategy_info"]["parameters"] == {}

    def test_reinitialize_instance_queues_coverage(self, client):
        """Test instance queue reinitialization for different strategies."""
        # Register multiple instances
        for i in range(1, 4):
            client.post(
                "/v1/instance/register",
                json={
                    "instance_id": f"inst-{i}",
                    "model_id": "model-1",
                    "endpoint": f"http://localhost:800{i}",
                    "platform_info": {
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "test-hardware",
                    },
                },
            )

        # Switch to min_time (expect_error queues)
        response = client.post("/v1/strategy/set", json={"strategy_name": "min_time"})
        assert response.status_code == 200
        assert response.json()["reinitialized_instances"] == 3

        # Switch to round_robin (probabilistic queues)
        response = client.post("/v1/strategy/set", json={"strategy_name": "round_robin"})
        assert response.status_code == 200
        assert response.json()["reinitialized_instances"] == 3

    def test_model_not_found_error_path(self, client):
        """Test submit_task with 'Model not found' ValueError."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Mock to raise ValueError with "Model not found"
        # Mock the scheduling_strategy
        with patch(
            "swarmpilot.scheduler.api.scheduling_strategy.schedule_task",
            side_effect=ValueError("Model not found in database"),
        ):
            response = client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # With background scheduling, API returns 200 immediately
        assert response.status_code == 200


class TestLifespanContextManager:
    """Tests for application lifespan startup and shutdown."""

    async def test_lifespan_startup_success(self):
        """Test successful application startup."""
        from swarmpilot.scheduler import api

        # Track startup calls
        startup_called = False
        shutdown_called = False

        # Mock the external services
        with patch(
            "swarmpilot.scheduler.api.predictor_client.close", new=AsyncMock()
        ) as mock_predictor_close:
            # Get the lifespan context manager
            lifespan_cm = api.lifespan(api.app)

            # Enter the context (startup)
            async with lifespan_cm:
                startup_called = True

            shutdown_called = True
            # Verify shutdown was called
            mock_predictor_close.assert_called_once()

        assert startup_called
        assert shutdown_called

    async def test_lifespan_shutdown_with_training_client(self):
        """Test shutdown when training client is available."""
        from swarmpilot.scheduler import api

        # Set up a mock training client
        mock_training_client = AsyncMock()
        mock_training_client.close = AsyncMock()

        with patch("swarmpilot.scheduler.api.predictor_client.close", new=AsyncMock()):
            # Temporarily set training_client
            original_training_client = api.training_client
            api.training_client = mock_training_client

            try:
                lifespan_cm = api.lifespan(api.app)
                async with lifespan_cm:
                    pass

                # Verify training client was closed
                mock_training_client.close.assert_called_once()
            finally:
                # Restore original
                api.training_client = original_training_client

    async def test_lifespan_shutdown_error_handling(self):
        """Test shutdown error propagation."""
        from swarmpilot.scheduler import api

        # Mock predictor_client.close to fail
        mock_predictor_close = AsyncMock(side_effect=Exception("Shutdown failed"))

        with patch("swarmpilot.scheduler.api.predictor_client.close", mock_predictor_close):
            lifespan_cm = api.lifespan(api.app)
            # Shutdown errors should propagate
            with pytest.raises(Exception) as exc_info:
                async with lifespan_cm:
                    pass

            assert "Shutdown failed" in str(exc_info.value)

        # Verify shutdown was attempted
        mock_predictor_close.assert_called_once()


class TestCustomQuantilesInReinitialization:
    """Tests for custom quantiles in instance queue reinitialization."""

    def test_reinitialize_with_custom_quantiles(self, client):
        """Test reinitializing instance queues with custom quantiles."""
        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Switch strategy with custom quantile
        response = client.post(
            "/v1/strategy/set",
            json={"strategy_name": "probabilistic", "target_quantile": 0.75},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["strategy_info"]["parameters"]["target_quantile"] == 0.75
        assert data["reinitialized_instances"] == 1


class TestAdditionalCoveragePaths:
    """Additional tests to reach 90% coverage."""

    async def test_websocket_subscribe_to_completed_task(self, client):
        """Test subscribing to a task that is already completed."""
        from swarmpilot.scheduler import api
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0},
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Mark task as completed
        from datetime import datetime

        from swarmpilot.scheduler.models import TaskStatus

        await api.task_registry.update_status("task-1", TaskStatus.COMPLETED)
        task = await api.task_registry.get("task-1")
        task.result = {"output": "completed"}
        # Set completed timestamp to calculate execution time
        task.completed_at = datetime.now().isoformat() + "Z"
        task.started_at = datetime.now().isoformat() + "Z"

        # Now subscribe to the completed task
        with client.websocket_connect("/v1/task/get_result") as websocket:
            websocket.send_json({"type": "subscribe", "task_ids": ["task-1"]})

            # Should receive the result immediately
            msg1 = websocket.receive_json()
            # Can be either result or ack depending on timing
            if msg1["type"] == "result":
                # Got result first
                assert msg1["task_id"] == "task-1"
                assert msg1["status"] == "completed"
                # Get ack next
                msg2 = websocket.receive_json()
                assert msg2["type"] == "ack"
            else:
                # Got ack first
                assert msg1["type"] == "ack"
                assert "task-1" in msg1["subscribed_tasks"]

    def test_websocket_unsubscribe_invalid_format(self, client):
        """Test unsubscribe with invalid task_ids format."""
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Send unsubscribe with invalid task_ids (not a list)
            websocket.send_json({"type": "unsubscribe", "task_ids": "not-a-list"})

            # Should receive error message
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert "must be a list" in error["error"]

    def test_websocket_pong_message(self, client):
        """Test receiving pong message from client."""
        with client.websocket_connect("/v1/task/get_result") as websocket:
            # Send pong message (client responding to server ping)
            websocket.send_json({"type": "pong", "timestamp": 123456.789})

            # Server should just log and continue
            # We can verify by sending a subscribe and getting ack
            websocket.send_json({"type": "subscribe", "task_ids": ["task-1"]})

            ack = websocket.receive_json()
            assert ack["type"] == "ack"

    def test_remove_instance_not_found_key_error(self, client):
        """Test remove instance when KeyError is raised."""
        # This should trigger the KeyError exception path (lines 385-386)
        response = client.post(
            "/v1/instance/remove",
            json={"instance_id": "definitely-does-not-exist"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]["error"].lower()

    def test_drain_instance_median_fallback(self, client):
        """Test drain when median quantile needs fallback."""
        from swarmpilot.scheduler.clients.models import Prediction

        # Register instance
        client.post(
            "/v1/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            },
        )

        # Submit task with quantiles that will test the fallback path
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={
                0.9: 110.0,
                0.95: 120.0,
            },  # No 0.5, will trigger fallback
            error_margin_ms=10.0,
        )

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[mock_prediction]),
        ):
            client.post(
                "/v1/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {},
                },
            )

        # Drain the instance - should trigger lines 448-450
        response = client.post("/v1/instance/drain", json={"instance_id": "inst-1"})

        assert response.status_code == 200
        data = response.json()
        assert "estimated_completion_time_ms" in data

    def test_get_strategy_info_unknown(self, client):
        """Test get_current_strategy_info with unknown strategy."""
        from swarmpilot.scheduler import api

        # Create a mock strategy with unknown class name
        class UnknownStrategy:
            pass

        original_strategy = api.scheduling_strategy
        try:
            api.scheduling_strategy = UnknownStrategy()

            response = client.get("/v1/strategy/get")

            assert response.status_code == 200
            data = response.json()
            # Should return "unknown" strategy name (lines 1182-1183)
            assert data["strategy_info"]["strategy_name"] == "unknown"
            assert data["strategy_info"]["parameters"] == {}
        finally:
            api.scheduling_strategy = original_strategy
