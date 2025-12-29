"""
Integration tests for model restart API endpoints.
Tests POST /model/restart and GET /model/restart/status endpoints.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi import status
from src.models import RestartStatus, ModelInfo


@pytest.mark.integration
class TestRestartAPIEndpoints:
    """Test suite for model restart API endpoints"""

    def test_restart_model_success(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
    ):
        """Test POST /model/restart - successful restart initiation"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="old-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_model_registry.model_exists.return_value = True

        # Make request
        response = api_client.post(
            "/model/restart",
            json={
                "model_id": "new-model",
                "parameters": {"key": "value"},
                "scheduler_url": "http://scheduler:8000"
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "operation_id" in data
        assert data["status"] == "draining"
        assert "Model restart operation initiated" in data["message"]

    def test_restart_model_no_model_running(
        self,
        api_client,
        mock_docker_manager,
    ):
        """Test POST /model/restart - fails when no model is running"""
        # Setup mock
        mock_docker_manager.is_model_running.return_value = False

        # Make request
        response = api_client.post(
            "/model/restart",
            json={
                "model_id": "new-model",
                "scheduler_url": "http://scheduler:8000"
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No model is currently running" in response.json()["detail"]

    def test_restart_model_invalid_model(
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
            json={
                "model_id": "invalid-model",
                "scheduler_url": "http://scheduler:8000"
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not found in registry" in response.json()["detail"]

    def test_restart_model_already_in_progress(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
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

        # Patch _perform_restart_operation to do nothing
        # This prevents the operation from completing and keeps it in "in progress" state
        async def mock_perform_restart(operation_id):
            # Do nothing - operation will stay in DRAINING state
            pass

        monkeypatch.setattr("src.api._perform_restart_operation", mock_perform_restart)

        # First request - should succeed
        response1 = api_client.post(
            "/model/restart",
            json={
                "model_id": "new-model",
                "scheduler_url": "http://scheduler:8000"
            }
        )
        assert response1.status_code == status.HTTP_200_OK

        # Second request - should fail with conflict
        response2 = api_client.post(
            "/model/restart",
            json={
                "model_id": "another-model",
                "scheduler_url": "http://scheduler:8000"
            }
        )
        assert response2.status_code == status.HTTP_409_CONFLICT
        assert "already in progress" in response2.json()["detail"]

    def test_get_restart_status_success(
        self,
        api_client,
        mock_docker_manager,
        mock_model_registry,
    ):
        """Test GET /model/restart/status - successful status query"""
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
            json={
                "model_id": "new-model",
                "parameters": {},
                "scheduler_url": "http://scheduler:8000"
            }
        )
        operation_id = restart_response.json()["operation_id"]

        # Query status
        status_response = api_client.get(
            f"/model/restart/status?operation_id={operation_id}"
        )

        # Verify response
        assert status_response.status_code == status.HTTP_200_OK
        data = status_response.json()
        assert data["success"] is True
        assert data["operation_id"] == operation_id
        assert data["status"] in [
            "pending", "draining", "waiting_tasks", "stopping_model",
            "deregistering", "starting_model", "registering", "completed", "failed"
        ]
        assert data["new_model_id"] == "new-model"

    def test_get_restart_status_not_found(self, api_client):
        """Test GET /model/restart/status - operation not found"""
        response = api_client.get(
            "/model/restart/status?operation_id=nonexistent-id"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestRestartBackgroundOperation:
    """Test suite for the background restart operation"""

    async def test_perform_restart_operation_complete_flow(
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
        monkeypatch.setattr("src.api.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.api.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.api.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.api.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.api import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-123"
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
        assert operation.completed_at is not None
        assert operation.error is None

        # Verify all steps were called
        mock_scheduler_client.drain_instance.assert_called_once()
        mock_task_queue.peek_pending_tasks.assert_called()  # Updated: now uses peek_pending_tasks
        mock_docker_manager.stop_model.assert_called_once()
        mock_scheduler_client.deregister_instance.assert_called_once()
        mock_docker_manager.start_model.assert_called_once_with(
            "new-model", {}
        )
        mock_scheduler_client.register_instance.assert_called_once()

    async def test_perform_restart_operation_with_pending_tasks(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - extracts and redistributes pending tasks"""
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
        mock_scheduler_client.resubmit_task = AsyncMock(return_value=True)

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

        # Setup task queue mock - peek pending tasks returns 3 tasks
        pending_tasks_to_return = [
            {
                "task_id": f"task-{i}",
                "model_id": "old-model",
                "task_input": {"data": f"test-{i}"},
                "enqueue_time": 1234567890.0 + i,
                "submitted_at": "2024-01-01T00:00:00Z",
                "callback_url": None,
                "metadata": {}
            }
            for i in range(3)
        ]
        mock_task_queue.peek_pending_tasks = AsyncMock(return_value=pending_tasks_to_return)
        mock_task_queue.remove_task = AsyncMock(return_value=True)  # For task redistribution
        mock_task_queue.current_task_id = None  # No running tasks

        # Patch global functions
        monkeypatch.setattr("src.api.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.api.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.api.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.api.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.api import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-456"
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
        assert operation.redistributed_tasks_count == 3
        # Verify peek_pending_tasks was called
        mock_task_queue.peek_pending_tasks.assert_called_once()
        # Verify resubmit_task was called 3 times
        assert mock_scheduler_client.resubmit_task.call_count == 3
        # Verify remove_task was called 3 times (after successful resubmit)
        assert mock_task_queue.remove_task.call_count == 3

    async def test_perform_restart_operation_model_not_found(
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
        monkeypatch.setattr("src.api.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.api.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.api.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.api.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.api import _perform_restart_operation, _restart_operations, RestartOperation
        from src.models import RestartStatus

        # Create operation
        operation_id = "test-op-789"
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

    async def test_perform_restart_operation_timeout(
        self,
        mock_docker_manager,
        mock_model_registry,
        mock_task_queue,
        monkeypatch,
    ):
        """Test _perform_restart_operation - fails on timeout waiting for running task"""
        # Create mock scheduler client
        mock_scheduler_client = Mock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 0
        })

        # Setup docker manager mocks
        mock_docker_manager.is_model_running.return_value = True

        # Setup registry mock
        mock_model_registry.model_exists.return_value = True

        # Setup task queue mock - no queued tasks, but running task never completes
        mock_task_queue.peek_pending_tasks = AsyncMock(return_value=[])
        # Simulate a running task by setting current_task_id to a non-None value
        # This will cause the while loop to wait for it to complete
        mock_task_queue.current_task_id = "running-task-id"

        # Patch global functions
        monkeypatch.setattr("src.api.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.api.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.api.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.api.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.api import _perform_restart_operation, _restart_operations, RestartOperation
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

        # Execute restart operation with short timeout by patching max_wait_time
        with patch('src.api.time.time') as mock_time:
            # Simulate immediate timeout
            mock_time.side_effect = [0, 0, 301]  # Start, poll check, timeout check
            await _perform_restart_operation(operation_id)

        # Verify operation failed due to timeout
        assert operation.status == RestartStatus.FAILED
        assert operation.error is not None
        assert "timeout" in operation.error.lower() or "Timeout" in operation.error

    async def test_perform_restart_operation_no_scheduler(
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
        monkeypatch.setattr("src.api.get_docker_manager", lambda: mock_docker_manager)
        monkeypatch.setattr("src.api.get_task_queue", lambda: mock_task_queue)
        monkeypatch.setattr("src.api.get_registry", lambda: mock_model_registry)
        monkeypatch.setattr("src.api.get_scheduler_client", lambda: mock_scheduler_client)

        # Import after patching
        from src.api import _perform_restart_operation, _restart_operations, RestartOperation
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
