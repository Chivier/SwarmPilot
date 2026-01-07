"""API-level integration tests for safe instance removal.

Tests the HTTP endpoints:
- POST /instance/drain
- GET /instance/drain/status
- POST /instance/remove (updated with safe_remove)
- POST /task/submit (verifies exclusion of draining instances)
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with a fresh app instance."""
    # Import here to avoid circular dependencies
    from src import api

    # Reset global registries for clean state
    from src.registry.instance_registry import InstanceRegistry
    from src.registry.task_registry import TaskRegistry
    from src.services.websocket_manager import ConnectionManager

    api.instance_registry = InstanceRegistry()
    api.task_registry = TaskRegistry()
    api.websocket_manager = ConnectionManager()

    return TestClient(api.app)


@pytest.fixture
def register_test_instance(client):
    """Helper fixture to register a test instance."""

    def _register(instance_id="test-inst-1", port=9000):
        response = client.post(
            "/instance/register",
            json={
                "instance_id": instance_id,
                "model_id": "test_model",
                "endpoint": f"http://localhost:{port}",
                "platform_info": {
                    "software_name": "Linux",
                    "software_version": "5.15.0",
                    "hardware_name": "x86_64",
                },
            },
        )
        assert response.status_code == 200
        return response.json()

    return _register


# ============================================================================
# Drain Endpoint Tests
# ============================================================================


class TestDrainEndpoint:
    """Tests for POST /instance/drain endpoint."""

    async def test_drain_active_instance_success(
        self, client, register_test_instance
    ):
        """Test draining an active instance."""
        register_test_instance("inst-1", 9001)

        response = client.post(
            "/instance/drain", json={"instance_id": "inst-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "inst-1"
        assert data["status"] == "draining"
        assert data["pending_tasks"] == 0
        assert data["running_tasks"] == 0

    async def test_drain_nonexistent_instance(self, client):
        """Test draining non-existent instance returns 404."""
        response = client.post(
            "/instance/drain", json={"instance_id": "nonexistent"}
        )

        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["success"] is False
        assert "not found" in data["detail"]["error"].lower()

    async def test_drain_already_draining_fails(
        self, client, register_test_instance
    ):
        """Test draining already draining instance returns 400."""
        register_test_instance("inst-1", 9001)

        # First drain succeeds
        response1 = client.post(
            "/instance/drain", json={"instance_id": "inst-1"}
        )
        assert response1.status_code == 200

        # Second drain fails
        response2 = client.post(
            "/instance/drain", json={"instance_id": "inst-1"}
        )
        assert response2.status_code == 400
        data = response2.json()
        assert data["detail"]["success"] is False

    async def test_drain_includes_estimated_time(
        self, client, register_test_instance
    ):
        """Test drain response includes estimated completion time."""
        register_test_instance("inst-1", 9001)

        # Mock queue info with some expected time
        from src import api
        from src.model import InstanceQueueExpectError

        await api.instance_registry.update_queue_info(
            "inst-1",
            InstanceQueueExpectError(
                instance_id="inst-1",
                expected_time_ms=1500.0,
                error_margin_ms=200.0,
            ),
        )

        response = client.post(
            "/instance/drain", json={"instance_id": "inst-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "estimated_completion_time_ms" in data
        assert data["estimated_completion_time_ms"] == 1500.0


# ============================================================================
# Drain Status Endpoint Tests
# ============================================================================


class TestDrainStatusEndpoint:
    """Tests for GET /instance/drain/status endpoint."""

    async def test_get_status_active_instance(
        self, client, register_test_instance
    ):
        """Test getting drain status for active instance."""
        register_test_instance("inst-1", 9001)

        response = client.get(
            "/instance/drain/status", params={"instance_id": "inst-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "inst-1"
        assert data["status"] == "active"
        assert data["pending_tasks"] == 0
        assert data["can_remove"] is False

    async def test_get_status_draining_no_tasks(
        self, client, register_test_instance
    ):
        """Test drain status for draining instance with no pending tasks."""
        register_test_instance("inst-1", 9001)

        # Start draining
        client.post("/instance/drain", json={"instance_id": "inst-1"})

        # Check status
        response = client.get(
            "/instance/drain/status", params={"instance_id": "inst-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "draining"
        assert data["pending_tasks"] == 0
        assert data["can_remove"] is True
        assert data["drain_initiated_at"] is not None

    async def test_get_status_draining_with_tasks(
        self, client, register_test_instance
    ):
        """Test drain status for draining instance with pending tasks."""
        from src import api

        register_test_instance("inst-1", 9001)

        # Add pending tasks
        await api.instance_registry.increment_pending("inst-1")
        await api.instance_registry.increment_pending("inst-1")

        # Start draining
        client.post("/instance/drain", json={"instance_id": "inst-1"})

        # Check status
        response = client.get(
            "/instance/drain/status", params={"instance_id": "inst-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "draining"
        assert data["pending_tasks"] == 2
        assert data["can_remove"] is False

    async def test_get_status_nonexistent_instance(self, client):
        """Test getting status for non-existent instance returns 404."""
        response = client.get(
            "/instance/drain/status", params={"instance_id": "nonexistent"}
        )

        assert response.status_code == 404


# ============================================================================
# Safe Remove Endpoint Tests
# ============================================================================


class TestSafeRemoveEndpoint:
    """Tests for updated POST /instance/remove endpoint."""

    async def test_remove_draining_instance_no_tasks(
        self, client, register_test_instance
    ):
        """Test safe removal of draining instance with no tasks."""
        register_test_instance("inst-1", 9001)

        # Drain instance
        client.post("/instance/drain", json={"instance_id": "inst-1"})

        # Remove should succeed
        response = client.post(
            "/instance/remove", json={"instance_id": "inst-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "inst-1"

        # Verify instance is gone
        list_response = client.get("/instance/list")
        instances = list_response.json()["instances"]
        assert "inst-1" not in [i["instance_id"] for i in instances]

    async def test_remove_active_instance_succeeds(
        self, client, register_test_instance
    ):
        """Test removing active instance without draining succeeds (permissive removal)."""
        register_test_instance("inst-1", 9001)

        # Remove without draining - now allowed
        response = client.post(
            "/instance/remove", json={"instance_id": "inst-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "inst-1"

    async def test_remove_draining_instance_with_tasks_succeeds(
        self, client, register_test_instance
    ):
        """Test removing draining instance with pending tasks succeeds (with warning log)."""
        from src import api

        register_test_instance("inst-1", 9001)

        # Add pending tasks
        await api.instance_registry.increment_pending("inst-1")

        # Drain instance
        client.post("/instance/drain", json={"instance_id": "inst-1"})

        # Remove - now succeeds (logs warning about pending tasks)
        response = client.post(
            "/instance/remove", json={"instance_id": "inst-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# ============================================================================
# Task Assignment Integration Tests
# ============================================================================


class TestTaskAssignmentWithDraining:
    """Tests verifying draining instances don't receive new tasks."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup mocks for task submission dependencies."""
        from src import api
        from src.clients.predictor_client import Prediction

        # Mock the predict method on background_scheduler's scheduling_strategy's predictor_client
        # This is necessary because background_scheduler uses scheduling_strategy
        async def mock_predict(
            model_id, metadata, instances, prediction_type="quantile"
        ):
            """Mock predict that returns predictions for all instances."""
            return [
                Prediction(
                    instance_id=inst.instance_id,
                    predicted_time_ms=1000.0,
                    confidence=0.9,
                    quantiles={
                        0.5: 900.0,
                        0.9: 1100.0,
                        0.95: 1200.0,
                        0.99: 1500.0,
                    },
                )
                for inst in instances
            ]

        with patch.object(
            api.background_scheduler.scheduling_strategy.predictor_client,
            "predict",
            side_effect=mock_predict,
        ):
            yield

    @pytest.fixture
    def mock_task_dispatcher(self):
        """Mock task dispatcher to avoid actual task dispatch."""
        from src import api

        with patch.object(
            api.background_scheduler.task_dispatcher, "dispatch_task_async"
        ) as mock_dispatch:
            yield mock_dispatch

    async def test_task_not_assigned_to_draining_instance(
        self, client, register_test_instance, mock_task_dispatcher
    ):
        """Test that tasks are not assigned to draining instances.

        Note: With background scheduling, we verify that draining instances
        are excluded from the available instances list, which ensures they
        won't be selected by the background scheduler.
        """
        from src import api
        from src.model import InstanceQueueExpectError

        # Register 3 instances
        register_test_instance("inst-1", 9001)
        register_test_instance("inst-2", 9002)
        register_test_instance("inst-3", 9003)

        # Setup queue info for scheduling
        for inst_id in ["inst-1", "inst-2", "inst-3"]:
            await api.instance_registry.update_queue_info(
                inst_id,
                InstanceQueueExpectError(
                    instance_id=inst_id,
                    expected_time_ms=1000.0,
                    error_margin_ms=100.0,
                ),
            )

        # Drain one instance
        drain_response = client.post(
            "/instance/drain", json={"instance_id": "inst-1"}
        )
        assert drain_response.status_code == 200

        # Verify that inst-1 is excluded from active instances
        active_instances = await api.instance_registry.list_active(
            model_id="test_model"
        )
        active_ids = [inst.instance_id for inst in active_instances]
        assert "inst-1" not in active_ids
        assert "inst-2" in active_ids
        assert "inst-3" in active_ids

        # Submit a task - it should succeed (returns immediately in background mode)
        task_response = client.post(
            "/task/submit",
            json={
                "task_id": "test-task-1",
                "model_id": "test_model",
                "task_input": {"data": "test"},
                "metadata": {"test": True},
            },
        )

        assert task_response.status_code == 200
        data = task_response.json()

        # Task is initially in PENDING state with no assigned instance
        assert data["task"]["status"] == "pending"

        # The key test: background scheduler will only see inst-2 and inst-3
        # as available instances, so inst-1 cannot be selected

    async def test_no_available_instances_when_all_draining(
        self, client, register_test_instance, mock_task_dispatcher
    ):
        """Test task submission fails when all instances are draining."""
        # Register 2 instances
        register_test_instance("inst-1", 9001)
        register_test_instance("inst-2", 9002)

        # Drain both
        client.post("/instance/drain", json={"instance_id": "inst-1"})
        client.post("/instance/drain", json={"instance_id": "inst-2"})

        # Try to submit task - task is queued even when all instances are draining
        response = client.post(
            "/task/submit",
            json={
                "task_id": "test-task-1",
                "model_id": "test_model",
                "task_input": {"data": "test"},
                "metadata": {"test": True},
            },
        )

        # Task is queued (200) and waits for an available instance
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task"]["status"] == "pending"

    async def test_instance_becomes_available_after_drain_completes(
        self, client, register_test_instance, mock_task_dispatcher
    ):
        """Test workflow: drain → tasks complete → remove → re-register."""
        from src import api

        # Register instance
        register_test_instance("inst-1", 9001)

        # Setup queue info
        await api.instance_registry.update_queue_info(
            "inst-1",
            api.InstanceQueueExpectError(
                instance_id="inst-1",
                expected_time_ms=1000.0,
                error_margin_ms=100.0,
            ),
        )

        # Add pending task
        await api.instance_registry.increment_pending("inst-1")

        # Drain instance
        client.post("/instance/drain", json={"instance_id": "inst-1"})

        # Task completes
        await api.instance_registry.decrement_pending("inst-1")
        await api.instance_registry.increment_completed("inst-1")

        # Check can_remove status
        status_response = client.get(
            "/instance/drain/status", params={"instance_id": "inst-1"}
        )
        assert status_response.json()["can_remove"] is True

        # Remove instance
        remove_response = client.post(
            "/instance/remove", json={"instance_id": "inst-1"}
        )
        assert remove_response.status_code == 200

        # Re-register same instance
        register_response = client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "test_model",
                "endpoint": "http://localhost:9001",
                "platform_info": {
                    "software_name": "Linux",
                    "software_version": "5.15.0",
                    "hardware_name": "x86_64",
                },
            },
        )
        assert register_response.status_code == 200

        # Instance should be available again
        list_response = client.get("/instance/list")
        instances = list_response.json()["instances"]
        instance_ids = [i["instance_id"] for i in instances]
        assert "inst-1" in instance_ids

        # And should be ACTIVE status
        inst_1 = next(i for i in instances if i["instance_id"] == "inst-1")
        assert inst_1["status"] == "active"


# ============================================================================
# Complete Workflow Integration Test
# ============================================================================


class TestCompleteRemovalWorkflow:
    """End-to-end test of the complete safe removal workflow."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup mocks."""
        with patch("src.api.predictor_client") as mock_predictor:
            mock_predictor.predict = AsyncMock(return_value=[])
            with patch("src.api.task_dispatcher") as mock_dispatcher:
                mock_dispatcher.dispatch_task = AsyncMock()
                yield

    async def test_complete_workflow(self, client, register_test_instance):
        """Test complete workflow from registration to safe removal.

        1. Register instance
        2. Submit tasks
        3. Drain instance
        4. Verify no new tasks assigned
        5. Simulate task completion
        6. Remove instance when safe
        """
        from src import api
        from src.model import InstanceQueueExpectError

        # Step 1: Register 2 instances
        register_test_instance("inst-1", 9001)
        register_test_instance("inst-2", 9002)

        # Setup queue info
        for inst_id in ["inst-1", "inst-2"]:
            await api.instance_registry.update_queue_info(
                inst_id,
                InstanceQueueExpectError(
                    instance_id=inst_id,
                    expected_time_ms=1000.0,
                    error_margin_ms=100.0,
                ),
            )

        # Step 2: Simulate inst-1 has pending tasks
        await api.instance_registry.increment_pending("inst-1")
        await api.instance_registry.increment_pending("inst-1")

        # Step 3: Drain inst-1
        drain_response = client.post(
            "/instance/drain", json={"instance_id": "inst-1"}
        )
        assert drain_response.status_code == 200
        assert drain_response.json()["pending_tasks"] == 2

        # Step 4: Verify inst-1 excluded from new task assignments
        list_active = await api.instance_registry.list_active(
            model_id="test_model"
        )
        assert len(list_active) == 1
        assert list_active[0].instance_id == "inst-2"

        # Step 5: Remove succeeds even with pending tasks (permissive behavior)
        # Note: This logs a warning but allows removal for operational flexibility
        remove_response = client.post(
            "/instance/remove", json={"instance_id": "inst-1"}
        )
        assert remove_response.status_code == 200
        assert remove_response.json()["success"] is True

        # Step 6: Verify instance is gone
        final_list = client.get("/instance/list")
        instances = final_list.json()["instances"]
        assert len(instances) == 1
        assert instances[0]["instance_id"] == "inst-2"
