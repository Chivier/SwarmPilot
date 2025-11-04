"""
Integration tests for API endpoints (src/api.py)
Tests all API endpoints with various scenarios
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
from src.models import Task, TaskStatus, ModelInfo, InstanceStatus


@pytest.mark.integration
class TestModelManagementEndpoints:
    """Test suite for model management endpoints"""

    def test_start_model_success(self, api_client, mock_docker_manager, mock_model_registry):
        """Test POST /model/start - successful model start"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={"temp": 0.7}
        )

        # Make request
        response = api_client.post(
            "/model/start",
            json={"model_id": "test-model", "parameters": {"temp": 0.7}}
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["model_id"] == "test-model"
        assert data["status"] == "running"

    def test_start_model_already_running(self, api_client, mock_docker_manager):
        """Test POST /model/start - fails when model already running"""
        # Setup mock
        mock_docker_manager.is_model_running.return_value = True

        # Make request
        response = api_client.post(
            "/model/start",
            json={"model_id": "test-model"}
        )

        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already running" in response.json()["detail"]

    def test_start_model_not_found(self, api_client, mock_docker_manager, mock_model_registry):
        """Test POST /model/start - model not in registry"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = False

        # Make request
        response = api_client.post(
            "/model/start",
            json={"model_id": "non-existent-model"}
        )

        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not found in registry" in response.json()["detail"]

    def test_start_model_failure(self, api_client, mock_docker_manager, mock_model_registry):
        """Test POST /model/start - container fails to start"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        mock_docker_manager.start_model.side_effect = RuntimeError("Container failed")

        # Make request
        response = api_client.post(
            "/model/start",
            json={"model_id": "test-model"}
        )

        # Verify response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to start model" in response.json()["detail"]

    def test_stop_model_success(self, api_client, mock_docker_manager):
        """Test GET /model/stop - successful model stop"""
        # Setup mock
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "test-model"

        # Make request
        response = api_client.get("/model/stop")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["model_id"] == "test-model"

    def test_stop_model_not_running(self, api_client, mock_docker_manager):
        """Test GET /model/stop - no model running"""
        # Setup mock
        mock_docker_manager.is_model_running.return_value = False

        # Make request
        response = api_client.get("/model/stop")

        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No model is currently running" in response.json()["detail"]


@pytest.mark.integration
class TestTaskManagementEndpoints:
    """Test suite for task management endpoints"""

    def test_submit_task_success(self, api_client, mock_docker_manager, mock_task_queue):
        """Test POST /task/submit - successful task submission"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = Mock(model_id="test-model")
        mock_task_queue.submit_task.return_value = 1

        # Make request
        response = api_client.post(
            "/task/submit",
            json={
                "task_id": "task-1",
                "model_id": "test-model",
                "task_input": {"prompt": "Hello"}
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["task_id"] == "task-1"
        assert data["position"] == 1
        assert data["status"] == "queued"

    def test_submit_task_no_model(self, api_client, mock_docker_manager):
        """Test POST /task/submit - no model running"""
        # Setup mock
        mock_docker_manager.is_model_running.return_value = False

        # Make request
        response = api_client.post(
            "/task/submit",
            json={
                "task_id": "task-1",
                "model_id": "test-model",
                "task_input": {"prompt": "Hello"}
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert "No model is currently running" in response.json()["detail"]

    def test_submit_task_model_mismatch(self, api_client, mock_docker_manager):
        """Test POST /task/submit - model ID mismatch"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = Mock(model_id="different-model")

        # Make request
        response = api_client.post(
            "/task/submit",
            json={
                "task_id": "task-1",
                "model_id": "test-model",
                "task_input": {"prompt": "Hello"}
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert "does not match" in response.json()["detail"]

    def test_submit_task_duplicate(self, api_client, mock_docker_manager, mock_task_queue):
        """Test POST /task/submit - duplicate task ID"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = Mock(model_id="test-model")
        mock_task_queue.submit_task.side_effect = ValueError("Task already exists")

        # Make request
        response = api_client.post(
            "/task/submit",
            json={
                "task_id": "task-1",
                "model_id": "test-model",
                "task_input": {"prompt": "Hello"}
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already exists" in response.json()["detail"]

    def test_list_tasks_no_filter(self, api_client, mock_task_queue):
        """Test GET /task/list - list all tasks"""
        # Setup mock
        task1 = Task(task_id="task-1", model_id="model", task_input={"prompt": "test1"})
        task2 = Task(task_id="task-2", model_id="model", task_input={"prompt": "test2"})
        mock_task_queue.list_tasks.return_value = [task1, task2]

        # Make request
        response = api_client.get("/task/list")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 2
        assert len(data["tasks"]) == 2

    def test_list_tasks_with_status_filter(self, api_client, mock_task_queue):
        """Test GET /task/list?status=completed"""
        # Setup mock
        task = Task(task_id="task-1", model_id="model", task_input={"prompt": "test"})
        task.mark_completed({"output": "result"})
        mock_task_queue.list_tasks.return_value = [task]

        # Make request
        response = api_client.get("/task/list?status=completed")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["tasks"][0]["status"] == "completed"

    def test_list_tasks_invalid_status(self, api_client):
        """Test GET /task/list?status=invalid"""
        response = api_client.get("/task/list?status=invalid")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid status" in response.json()["detail"]

    def test_list_tasks_with_limit(self, api_client, mock_task_queue):
        """Test GET /task/list?limit=10"""
        # Setup mock
        tasks = [
            Task(task_id=f"task-{i}", model_id="model", task_input={})
            for i in range(10)
        ]
        mock_task_queue.list_tasks.return_value = tasks

        # Make request
        response = api_client.get("/task/list?limit=10")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 10

    def test_get_task_exists(self, api_client, mock_task_queue):
        """Test GET /task/{task_id} - task exists"""
        # Setup mock
        task = Task(task_id="task-1", model_id="model", task_input={"prompt": "test"})
        mock_task_queue.get_task.return_value = task

        # Make request
        response = api_client.get("/task/task-1")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["task"]["task_id"] == "task-1"

    def test_get_task_not_found(self, api_client, mock_task_queue):
        """Test GET /task/{task_id} - task not found"""
        # Setup mock
        mock_task_queue.get_task.return_value = None

        # Make request
        response = api_client.get("/task/non-existent-task")

        # Verify response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Task not found" in response.json()["detail"]

    def test_delete_task_success(self, api_client, mock_task_queue):
        """Test DELETE /task/{task_id} - successful deletion"""
        # Setup mock
        mock_task_queue.delete_task.return_value = True

        # Make request
        response = api_client.delete("/task/task-1")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["task_id"] == "task-1"

    def test_delete_task_running(self, api_client, mock_task_queue):
        """Test DELETE /task/{task_id} - cannot delete running task"""
        # Setup mock
        mock_task_queue.delete_task.side_effect = ValueError("Cannot delete a running task")

        # Make request
        response = api_client.delete("/task/task-1")

        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "running" in response.json()["detail"]

    def test_delete_task_not_found(self, api_client, mock_task_queue):
        """Test DELETE /task/{task_id} - task not found"""
        # Setup mock
        mock_task_queue.delete_task.return_value = False

        # Make request
        response = api_client.delete("/task/non-existent-task")

        # Verify response
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_clear_tasks_success_empty(self, api_client, mock_task_queue, mock_docker_manager):
        """Test POST /task/clear - successfully clear empty queue"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_task_queue.clear_all_tasks.return_value = {
            "queued": 0,
            "completed": 0,
            "failed": 0,
            "total": 0
        }

        # Make request
        response = api_client.post("/task/clear")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Successfully cleared 0 task(s)"
        assert data["cleared_count"]["total"] == 0
        # Verify Docker restart was not called (no model running)
        mock_docker_manager.restart_model.assert_not_called()

    def test_clear_tasks_success_with_tasks(self, api_client, mock_task_queue, mock_docker_manager):
        """Test POST /task/clear - successfully clear queue with tasks and restart Docker"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.restart_model.return_value = "test-model"
        mock_task_queue.clear_all_tasks.return_value = {
            "queued": 5,
            "completed": 10,
            "failed": 2,
            "total": 17
        }

        # Make request
        response = api_client.post("/task/clear")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Successfully cleared 17 task(s)"
        assert data["cleared_count"]["queued"] == 5
        assert data["cleared_count"]["completed"] == 10
        assert data["cleared_count"]["failed"] == 2
        assert data["cleared_count"]["total"] == 17
        # Verify Docker restart was called
        mock_docker_manager.restart_model.assert_called_once()

    def test_clear_tasks_docker_restart_failure(self, api_client, mock_task_queue, mock_docker_manager):
        """Test POST /task/clear - fails when Docker restart fails"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.restart_model.side_effect = RuntimeError("Failed to restart container")

        # Make request
        response = api_client.post("/task/clear")

        # Verify response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to restart Docker container" in response.json()["detail"]
        # Verify clear_all_tasks was not called
        mock_task_queue.clear_all_tasks.assert_not_called()

    def test_clear_tasks_with_running_task(self, api_client, mock_task_queue, mock_docker_manager):
        """Test POST /task/clear - fails when tasks are running"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.restart_model.return_value = "test-model"
        mock_task_queue.clear_all_tasks.side_effect = RuntimeError(
            "Cannot clear tasks while 1 task(s) are running. "
            "Wait for running tasks to complete or stop processing first."
        )

        # Make request
        response = api_client.post("/task/clear")

        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot clear tasks while 1 task(s) are running" in response.json()["detail"]

    def test_clear_tasks_with_multiple_running_tasks(self, api_client, mock_task_queue, mock_docker_manager):
        """Test POST /task/clear - fails when multiple tasks are running"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.restart_model.return_value = "test-model"
        mock_task_queue.clear_all_tasks.side_effect = RuntimeError(
            "Cannot clear tasks while 3 task(s) are running. "
            "Wait for running tasks to complete or stop processing first."
        )

        # Make request
        response = api_client.post("/task/clear")

        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot clear tasks while 3 task(s) are running" in response.json()["detail"]


@pytest.mark.integration
class TestManagementEndpoints:
    """Test suite for management endpoints"""

    def test_get_info_idle(self, api_client, mock_docker_manager, mock_task_queue):
        """Test GET /info - instance is idle (no model running)"""
        # Setup mocks
        mock_docker_manager.get_current_model.return_value = None
        mock_task_queue.get_queue_stats.return_value = {
            "total": 0,
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0
        }

        # Make request
        response = api_client.get("/info")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["instance"]["status"] == "idle"
        assert data["instance"]["current_model"] is None
        assert data["instance"]["task_queue"]["total"] == 0

    def test_get_info_running(self, api_client, mock_docker_manager, mock_task_queue):
        """Test GET /info - instance is running (model running, no tasks)"""
        # Setup mocks
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_task_queue.get_queue_stats.return_value = {
            "total": 5,
            "queued": 2,
            "running": 0,
            "completed": 3,
            "failed": 0
        }

        # Make request
        response = api_client.get("/info")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["instance"]["status"] == "running"
        assert data["instance"]["current_model"]["model_id"] == "test-model"

    def test_get_info_busy(self, api_client, mock_docker_manager, mock_task_queue):
        """Test GET /info - instance is busy (task processing)"""
        # Setup mocks
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_task_queue.get_queue_stats.return_value = {
            "total": 5,
            "queued": 2,
            "running": 1,
            "completed": 2,
            "failed": 0
        }

        # Make request
        response = api_client.get("/info")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["instance"]["status"] == "busy"
        assert data["instance"]["task_queue"]["running"] == 1

    def test_health_check_healthy(self, api_client, mock_docker_manager):
        """Test GET /health - instance is healthy"""
        # Setup mock
        mock_docker_manager.is_model_running.return_value = False

        # Make request
        response = api_client.get("/health")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_healthy_with_model(self, api_client, mock_docker_manager):
        """Test GET /health - instance with healthy model"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.check_model_health.return_value = True

        # Make request
        response = api_client.get("/health")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_unhealthy(self, api_client, mock_docker_manager):
        """Test GET /health - model is unhealthy"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.check_model_health.return_value = False

        # Make request
        response = api_client.get("/health")

        # Verify response
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data


@pytest.mark.integration
class TestRequestValidation:
    """Test suite for request validation"""

    def test_start_model_missing_model_id(self, api_client):
        """Test POST /model/start without model_id"""
        response = api_client.post("/model/start", json={})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_submit_task_missing_fields(self, api_client):
        """Test POST /task/submit with missing required fields"""
        # Missing task_input
        response = api_client.post(
            "/task/submit",
            json={"task_id": "task-1", "model_id": "model"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_list_tasks_invalid_limit(self, api_client):
        """Test GET /task/list with invalid limit"""
        # Limit too large
        response = api_client.get("/task/list?limit=2000")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.integration
class TestSchedulerIntegration:
    """Test suite for scheduler integration in API endpoints"""

    def test_start_model_with_scheduler_registration_success(
        self, api_client, mock_docker_manager, mock_model_registry
    ):
        """Test POST /model/start with successful scheduler registration"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock scheduler client
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request
            response = api_client.post(
                "/model/start",
                json={"model_id": "test-model"}
            )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        mock_scheduler_client.register_instance.assert_called_once_with(model_id="test-model")

    def test_start_model_with_scheduler_registration_failure(
        self, api_client, mock_docker_manager, mock_model_registry
    ):
        """Test POST /model/start when scheduler registration fails (should not fail request)"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock scheduler client to raise exception
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.register_instance = AsyncMock(side_effect=Exception("Network error"))

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request - should still succeed even if scheduler fails
            response = api_client.post(
                "/model/start",
                json={"model_id": "test-model"}
            )

        # Verify response still succeeds
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True

    def test_start_model_with_scheduler_disabled(
        self, api_client, mock_docker_manager, mock_model_registry
    ):
        """Test POST /model/start when scheduler is disabled"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock scheduler client as disabled
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.is_enabled = False

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request
            response = api_client.post(
                "/model/start",
                json={"model_id": "test-model"}
            )

        # Verify response succeeds and scheduler not called
        assert response.status_code == status.HTTP_200_OK
        mock_scheduler_client.register_instance.assert_not_called()

    def test_stop_model_with_scheduler_deregistration(
        self, api_client, mock_docker_manager
    ):
        """Test GET /model/stop with scheduler deregistration"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "test-model"

        # Mock scheduler client
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.deregister_instance = AsyncMock(return_value=True)

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request
            response = api_client.get("/model/stop")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        mock_scheduler_client.deregister_instance.assert_called_once()

    def test_stop_model_with_scheduler_deregistration_failure(
        self, api_client, mock_docker_manager
    ):
        """Test GET /model/stop when scheduler deregistration fails (should not fail request)"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.stop_model.return_value = "test-model"

        # Mock scheduler client to raise exception
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.deregister_instance = AsyncMock(side_effect=Exception("Network error"))

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request - should still succeed even if scheduler fails
            response = api_client.get("/model/stop")

        # Verify response still succeeds
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True

    def test_start_model_with_scheduler_url_parameter(
        self, api_client, mock_docker_manager, mock_model_registry
    ):
        """Test POST /model/start with scheduler_url parameter updates scheduler configuration"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock scheduler client
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.scheduler_url = "http://old-scheduler:8000"
        mock_scheduler_client._registered = True
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request with new scheduler_url
            new_scheduler_url = "http://new-scheduler:8100"
            response = api_client.post(
                "/model/start",
                json={
                    "model_id": "test-model",
                    "scheduler_url": new_scheduler_url
                }
            )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True

        # Verify scheduler_url was updated
        assert mock_scheduler_client.scheduler_url == new_scheduler_url

        # Verify registration status was reset
        assert mock_scheduler_client._registered is False

        # Verify register_instance was called with the new scheduler
        mock_scheduler_client.register_instance.assert_called_once_with(model_id="test-model")

    def test_start_model_with_scheduler_url_no_previous_scheduler(
        self, api_client, mock_docker_manager, mock_model_registry
    ):
        """Test POST /model/start with scheduler_url when no previous scheduler configured"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock scheduler client with no scheduler configured (None)
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.scheduler_url = None
        mock_scheduler_client.is_enabled = False  # Will become True after setting URL
        mock_scheduler_client._registered = False
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request with scheduler_url
            new_scheduler_url = "http://new-scheduler:8100"
            response = api_client.post(
                "/model/start",
                json={
                    "model_id": "test-model",
                    "scheduler_url": new_scheduler_url
                }
            )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True

        # Verify scheduler_url was set
        assert mock_scheduler_client.scheduler_url == new_scheduler_url

    def test_start_model_without_scheduler_url_uses_existing(
        self, api_client, mock_docker_manager, mock_model_registry
    ):
        """Test POST /model/start without scheduler_url uses existing scheduler configuration"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock scheduler client with existing scheduler
        existing_scheduler_url = "http://existing-scheduler:8000"
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.scheduler_url = existing_scheduler_url
        mock_scheduler_client._registered = False
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request WITHOUT scheduler_url parameter
            response = api_client.post(
                "/model/start",
                json={"model_id": "test-model"}
            )

        # Verify response
        assert response.status_code == status.HTTP_200_OK

        # Verify scheduler_url remains unchanged
        assert mock_scheduler_client.scheduler_url == existing_scheduler_url

        # Verify register_instance was called
        mock_scheduler_client.register_instance.assert_called_once_with(model_id="test-model")

    def test_start_model_with_scheduler_url_and_parameters(
        self, api_client, mock_docker_manager, mock_model_registry
    ):
        """Test POST /model/start with both scheduler_url and model parameters"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        model_params = {"temp": 0.7, "max_tokens": 100}
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters=model_params
        )

        # Mock scheduler client
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.scheduler_url = "http://old-scheduler:8000"
        mock_scheduler_client._registered = True
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request with both scheduler_url and parameters
            new_scheduler_url = "http://new-scheduler:8100"
            response = api_client.post(
                "/model/start",
                json={
                    "model_id": "test-model",
                    "parameters": model_params,
                    "scheduler_url": new_scheduler_url
                }
            )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True

        # Verify scheduler_url was updated
        assert mock_scheduler_client.scheduler_url == new_scheduler_url

        # Verify both model parameters and scheduler update worked
        mock_docker_manager.start_model.assert_called_once_with("test-model", model_params)
        mock_scheduler_client.register_instance.assert_called_once_with(model_id="test-model")

    def test_start_model_with_scheduler_url_registration_fails(
        self, api_client, mock_docker_manager, mock_model_registry
    ):
        """Test POST /model/start with scheduler_url when registration fails (should not fail request)"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = False
        mock_model_registry.model_exists.return_value = True
        mock_docker_manager.start_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock scheduler client with registration failure
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.scheduler_url = "http://old-scheduler:8000"
        mock_scheduler_client._registered = True
        mock_scheduler_client.register_instance = AsyncMock(
            side_effect=Exception("Network error - new scheduler unreachable")
        )

        with patch("src.api.get_scheduler_client", return_value=mock_scheduler_client):
            # Make request with new scheduler_url
            new_scheduler_url = "http://new-scheduler:8100"
            response = api_client.post(
                "/model/start",
                json={
                    "model_id": "test-model",
                    "scheduler_url": new_scheduler_url
                }
            )

        # Verify response still succeeds even though registration failed
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True

        # Verify scheduler_url was updated (even though registration failed)
        assert mock_scheduler_client.scheduler_url == new_scheduler_url

        # Verify registration was attempted
        mock_scheduler_client.register_instance.assert_called_once()
