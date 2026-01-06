"""
Unit tests for model restart API endpoints.
Tests POST /model/restart and GET /model/restart/status endpoints.

NOTE: Tests in this file are temporarily disabled because WebSocket
communication with scheduler has been disabled. All Instance-Scheduler
communication now uses HTTP API only.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi import status
from src.models import RestartStatus, ModelInfo

# Skip all tests since WebSocket restart functionality is temporarily disabled
pytestmark = pytest.mark.skip(reason="WebSocket restart functionality temporarily disabled")


@pytest.mark.unit
class TestRestartModelEndpoint:
    """Unit tests for POST /model/restart endpoint"""

    def test_restart_model_success_minimal_params(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
    ):
        """Test POST /model/restart - successful restart with minimal parameters"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_model_registry.model_exists.return_value = True

        # Make request with only required field
        response = api_client.post(
            "/model/restart",
            json={"model_id": "new-model"}
        )

        # Verify response structure
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "operation_id" in data
        assert isinstance(data["operation_id"], str)
        assert len(data["operation_id"]) > 0
        assert data["status"] == "draining"
        assert "Model restart operation initiated" in data["message"]

    def test_restart_model_success_with_all_params(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
    ):
        """Test POST /model/restart - successful restart with all parameters"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={"old_param": "old_value"}
        )
        mock_model_registry.model_exists.return_value = True

        # Make request with all parameters
        response = api_client.post(
            "/model/restart",
            json={
                "model_id": "new-model",
                "parameters": {"gpu": True, "batch_size": 32},
                "scheduler_url": "http://scheduler-2.example.com:8000"
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "operation_id" in data
        assert data["status"] == "draining"

    def test_restart_model_no_model_running(
        self,
        api_client,
        mock_docker_manager,
    ):
        """Test POST /model/restart - fails when no model is running"""
        # Setup mock - no model running
        mock_docker_manager.is_model_running.return_value = False

        # Make request
        response = api_client.post(
            "/model/restart",
            json={"model_id": "new-model"}
        )

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "No model is currently running" in data["detail"]

    def test_restart_model_invalid_model_id(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
    ):
        """Test POST /model/restart - fails when new model not in registry"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_model_registry.model_exists.return_value = False

        # Make request
        response = api_client.post(
            "/model/restart",
            json={"model_id": "nonexistent-model"}
        )

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "not found in registry" in data["detail"]
        assert "nonexistent-model" in data["detail"]

    def test_restart_model_already_in_progress(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test POST /model/restart - fails when restart already in progress"""
        # Setup mocks for first request
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_model_registry.model_exists.return_value = True

        # Make task queue return pending tasks to prevent immediate completion
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 100,  # Many pending tasks to block completion
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 100,
        }

        # Patch asyncio.create_task to prevent background execution
        original_create_task = asyncio.create_task
        created_tasks = []
        def mock_create_task(coro):
            task = original_create_task(coro)
            created_tasks.append(task)
            return task
        monkeypatch.setattr("asyncio.create_task", mock_create_task)

        # First request - should succeed
        response1 = api_client.post(
            "/model/restart",
            json={"model_id": "new-model"}
        )
        assert response1.status_code == status.HTTP_200_OK
        operation_id = response1.json()["operation_id"]

        # Second request - should fail with conflict (operation still in progress)
        response2 = api_client.post(
            "/model/restart",
            json={"model_id": "another-model"}
        )
        assert response2.status_code == status.HTTP_409_CONFLICT
        data = response2.json()
        assert "detail" in data
        assert "already in progress" in data["detail"]

    def test_restart_model_same_model_id(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
    ):
        """Test POST /model/restart - can restart with same model ID (parameter change)"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={"version": "1.0"}
        )
        mock_model_registry.model_exists.return_value = True

        # Restart with same model but different parameters
        response = api_client.post(
            "/model/restart",
            json={
                "model_id": "test-model",
                "parameters": {"version": "2.0"}
            }
        )

        # Should succeed (parameter change)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "operation_id" in data

    def test_restart_model_empty_parameters(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
    ):
        """Test POST /model/restart - handles empty parameters dict"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_model_registry.model_exists.return_value = True

        # Make request with empty parameters
        response = api_client.post(
            "/model/restart",
            json={
                "model_id": "new-model",
                "parameters": {}
            }
        )

        # Should succeed
        assert response.status_code == status.HTTP_200_OK

    def test_restart_model_missing_required_field(
        self,
        api_client,
    ):
        """Test POST /model/restart - fails when model_id is missing"""
        # Make request without required model_id
        response = api_client.post(
            "/model/restart",
            json={"parameters": {}}
        )

        # Should fail with validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_restart_model_invalid_scheduler_url(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
    ):
        """Test POST /model/restart - accepts any scheduler_url format (validation happens later)"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_model_registry.model_exists.return_value = True

        # Make request with invalid URL format
        response = api_client.post(
            "/model/restart",
            json={
                "model_id": "new-model",
                "scheduler_url": "not-a-valid-url"
            }
        )

        # Should succeed at request validation level (URL validation happens during operation)
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
class TestRestartStatusEndpoint:
    """Unit tests for GET /model/restart/status endpoint"""

    def test_get_restart_status_pending(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
    ):
        """Test GET /model/restart/status - pending operation status"""
        # Setup mocks and initiate restart
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_model_registry.model_exists.return_value = True

        # Initiate restart
        restart_response = api_client.post(
            "/model/restart",
            json={"model_id": "new-model", "parameters": {"key": "value"}}
        )
        operation_id = restart_response.json()["operation_id"]

        # Query status immediately
        status_response = api_client.get(
            f"/model/restart/status?operation_id={operation_id}"
        )

        # Verify response structure
        assert status_response.status_code == status.HTTP_200_OK
        data = status_response.json()
        assert data["success"] is True
        assert data["operation_id"] == operation_id
        assert data["status"] in [
            "pending", "draining", "waiting_tasks", "stopping_model",
            "deregistering", "starting_model", "registering", "completed", "failed"
        ]
        assert data["new_model_id"] == "new-model"
        assert "initiated_at" in data
        assert data["pending_tasks_at_start"] == 0
        assert data["pending_tasks_completed"] == 0

    def test_get_restart_status_with_old_model(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test GET /model/restart/status - includes old_model_id"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_docker_manager.stop_model.return_value = "old-model"
        mock_model_registry.model_exists.return_value = True

        # Make task queue return no pending tasks
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        # Prevent background task from running to check initial state
        def no_op_create_task(coro):
            # Don't actually run the task, just consume the coroutine
            coro.close()
            return Mock()
        monkeypatch.setattr("asyncio.create_task", no_op_create_task)

        # Initiate restart
        restart_response = api_client.post(
            "/model/restart",
            json={"model_id": "new-model"}
        )
        operation_id = restart_response.json()["operation_id"]

        # Query status immediately (before background task runs)
        status_response = api_client.get(
            f"/model/restart/status?operation_id={operation_id}"
        )

        # Verify old_model_id is present
        assert status_response.status_code == status.HTTP_200_OK
        data = status_response.json()
        assert data["old_model_id"] == "old-model"

    def test_get_restart_status_not_found(self, api_client):
        """Test GET /model/restart/status - operation not found"""
        response = api_client.get(
            "/model/restart/status?operation_id=nonexistent-id"
        )

        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"]

    def test_get_restart_status_missing_operation_id(self, api_client):
        """Test GET /model/restart/status - missing operation_id parameter"""
        response = api_client.get("/model/restart/status")

        # Should fail with validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_get_restart_status_completed_at_null_when_pending(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
    ):
        """Test GET /model/restart/status - completed_at is null when pending"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_model_registry.model_exists.return_value = True

        # Initiate restart
        restart_response = api_client.post(
            "/model/restart",
            json={"model_id": "new-model"}
        )
        operation_id = restart_response.json()["operation_id"]

        # Query status
        status_response = api_client.get(
            f"/model/restart/status?operation_id={operation_id}"
        )

        # Verify completed_at is null for non-terminal states
        data = status_response.json()
        if data["status"] not in ["completed", "failed"]:
            assert data["completed_at"] is None

    def test_get_restart_status_error_null_when_no_error(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
    ):
        """Test GET /model/restart/status - error is null when operation successful"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_model_registry.model_exists.return_value = True

        # Initiate restart
        restart_response = api_client.post(
            "/model/restart",
            json={"model_id": "new-model"}
        )
        operation_id = restart_response.json()["operation_id"]

        # Query status
        status_response = api_client.get(
            f"/model/restart/status?operation_id={operation_id}"
        )

        # Verify error is null
        data = status_response.json()
        if data["status"] != "failed":
            assert data["error"] is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestPerformRestartOperation:
    """Unit tests for the background _perform_restart_operation function"""

    async def test_perform_restart_complete_flow(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - complete successful flow"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 0
        })
        mock_scheduler_client.deregister_instance = AsyncMock(return_value=True)
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "old-model"
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="new-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock - no pending tasks
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-complete-flow"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={"test": "params"},
            new_scheduler_url=None,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation
        await _perform_restart_operation(operation_id)

        # Verify operation completed
        assert operation.status == RestartStatus.COMPLETED
        assert operation.completed_at is not None
        assert operation.error is None

        # Verify all steps were called in correct order
        mock_scheduler_client.drain_instance.assert_called_once()
        mock_task_queue.get_queue_stats.assert_called()
        mock_docker_manager.stop_model.assert_called_once()
        mock_scheduler_client.deregister_instance.assert_called_once()
        mock_docker_manager.start_model.assert_called_once_with(
            "new-model", {"test": "params"}
        )
        mock_scheduler_client.register_instance.assert_called_once()

    async def test_perform_restart_with_pending_tasks(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - waits for pending tasks"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 3
        })
        mock_scheduler_client.deregister_instance = AsyncMock(return_value=True)
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "old-model"
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="new-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock - tasks complete gradually
        call_count = [0]
        def get_queue_stats_side_effect():
            call_count[0] += 1
            if call_count[0] <= 2:
                # First two calls: still have pending tasks
                return {
                    "queued": 3 - call_count[0],
                    "running": 0,
                    "completed": call_count[0] - 1,
                    "failed": 0,
                    "total": 3,
                }
            else:
                # After that: all tasks completed
                return {
                    "queued": 0,
                    "running": 0,
                    "completed": 3,
                    "failed": 0,
                    "total": 3,
                }

        mock_task_queue.get_queue_stats.side_effect = get_queue_stats_side_effect

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-pending-tasks"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={},
            new_scheduler_url=None,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation
        await _perform_restart_operation(operation_id)

        # Verify operation completed
        assert operation.status == RestartStatus.COMPLETED
        assert operation.pending_tasks_at_start > 0
        assert operation.pending_tasks_completed == operation.pending_tasks_at_start
        # Verify we polled multiple times
        assert mock_task_queue.get_queue_stats.call_count >= 3

    async def test_perform_restart_model_not_found(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - fails when model not in registry"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 0
        })
        mock_scheduler_client.deregister_instance = AsyncMock(return_value=True)

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "old-model"

        # Setup registry mock - model doesn't exist
        mock_model_registry.model_exists.return_value = False

        # Setup task queue mock
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-not-found"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="nonexistent-model",
            new_parameters={},
            new_scheduler_url=None,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation
        await _perform_restart_operation(operation_id)

        # Verify operation failed
        assert operation.status == RestartStatus.FAILED
        assert operation.error is not None
        assert "not found in registry" in operation.error.lower()
        assert operation.completed_at is not None

    async def test_perform_restart_timeout(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - fails on timeout waiting for tasks"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 100
        })

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock - tasks never complete
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 100,  # Always have pending tasks
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 100,
        }

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-timeout"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={},
            new_scheduler_url=None,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation with short timeout by patching time.time
        with patch('src.server.time.time') as mock_time:
            # Simulate immediate timeout
            mock_time.side_effect = [0, 0, 301]  # Start, poll check, timeout check
            await _perform_restart_operation(operation_id)

        # Verify operation failed due to timeout
        assert operation.status == RestartStatus.FAILED
        assert operation.error is not None
        assert "timeout" in operation.error.lower() or "Timeout" in operation.error

    async def test_perform_restart_no_scheduler(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - works without scheduler"""
        # Create mock scheduler client with scheduler disabled
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = False

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "old-model"
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="new-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock - no pending tasks
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-no-scheduler"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={},
            new_scheduler_url=None,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation
        await _perform_restart_operation(operation_id)

        # Verify operation completed (scheduler steps should be skipped)
        assert operation.status == RestartStatus.COMPLETED
        assert operation.error is None

        # Verify scheduler methods were not called
        assert not hasattr(mock_scheduler_client, 'drain_instance') or \
               not mock_scheduler_client.drain_instance.called

    async def test_perform_restart_updates_scheduler_url(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - updates scheduler URL when provided"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.scheduler_url = "http://old-scheduler:8000"
        mock_scheduler_client._registered = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 0
        })
        mock_scheduler_client.deregister_instance = AsyncMock(return_value=True)
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "old-model"
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="new-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation with new scheduler URL
        operation_id = "test-op-new-scheduler"
        new_scheduler_url = "http://new-scheduler:9000"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={},
            new_scheduler_url=new_scheduler_url,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation
        await _perform_restart_operation(operation_id)

        # Verify scheduler URL was updated
        assert mock_scheduler_client.scheduler_url == new_scheduler_url
        assert mock_scheduler_client._registered is False  # Registration reset
        assert operation.status == RestartStatus.COMPLETED

    async def test_perform_restart_drain_failure_continues(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - continues even if drain fails"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(side_effect=Exception("Drain failed"))
        mock_scheduler_client.deregister_instance = AsyncMock(return_value=True)
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "old-model"
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="new-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-drain-fail"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={},
            new_scheduler_url=None,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation
        await _perform_restart_operation(operation_id)

        # Should complete successfully despite drain failure
        assert operation.status == RestartStatus.COMPLETED
        assert operation.error is None

    async def test_perform_restart_deregister_failure_continues(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - continues even if deregister fails"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 0
        })
        mock_scheduler_client.deregister_instance = AsyncMock(side_effect=Exception("Deregister failed"))
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "old-model"
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="new-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-dereg-fail"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={},
            new_scheduler_url=None,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation
        await _perform_restart_operation(operation_id)

        # Should complete successfully despite deregister failure
        assert operation.status == RestartStatus.COMPLETED
        assert operation.error is None

    async def test_perform_restart_register_failure_completes(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - completes even if final register fails"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 0
        })
        mock_scheduler_client.deregister_instance = AsyncMock(return_value=True)
        mock_scheduler_client.register_instance = AsyncMock(side_effect=Exception("Register failed"))

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "old-model"
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="new-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-reg-fail"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={},
            new_scheduler_url=None,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation
        await _perform_restart_operation(operation_id)

        # Should complete successfully despite register failure (only logged as warning)
        assert operation.status == RestartStatus.COMPLETED
        assert operation.error is None

    async def test_perform_restart_model_start_failure(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - fails when model start fails"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 0
        })
        mock_scheduler_client.deregister_instance = AsyncMock(return_value=True)

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "old-model"
        mock_docker_manager.start_model.side_effect = Exception("Failed to start model container")

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        # Patch global functions
        monkeypatch.setattr("src.server.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.server.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.server.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.server.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.server import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-start-fail"
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={},
            new_scheduler_url=None,
        )
        _restart_operations[operation_id] = operation

        # Execute restart operation
        await _perform_restart_operation(operation_id)

        # Should fail
        assert operation.status == RestartStatus.FAILED
        assert operation.error is not None
        assert "Failed to start model container" in operation.error
