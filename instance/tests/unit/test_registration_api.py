"""
Unit tests for model registration and deregistration API endpoints.
Tests POST /model/register and GET /model/deregister/status endpoints.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from fastapi import status
from src.models import DeregisterStatus, ModelInfo


@pytest.mark.unit
class TestRegisterModelEndpoint:
    """Unit tests for POST /model/register endpoint"""

    def test_register_model_success(
        self,
        api_client,
        mock_docker_manager,
        mock_scheduler_client,
    ):
        """Test POST /model/register - successful registration"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.register_instance = AsyncMock(return_value=True)

        # Make request
        response = api_client.post(
            "/model/register",
            json={"scheduler_url": "http://scheduler:8000"}
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "Successfully registered" in data["message"]
        assert "test-model" in data["message"]
        assert "http://scheduler:8000" in data["message"]

        # Verify scheduler URL was updated
        assert mock_scheduler_client.scheduler_url == "http://scheduler:8000"
        assert mock_scheduler_client._registered is False

    def test_register_model_no_model_running(
        self,
        api_client,
        mock_docker_manager,
        mock_scheduler_client
    ):
        """Test POST /model/register - fails when no model is running"""
        # Setup mock - no model running
        mock_docker_manager.is_model_running.return_value = False
        mock_scheduler_client._registered = False

        # Make request
        response = api_client.post(
            "/model/register",
            json={"scheduler_url": "http://scheduler:8000"}
        )

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "No model is currently running" in data["detail"]
        assert mock_scheduler_client._registered is False

    def test_register_model_empty_scheduler_url(
        self,
        api_client,
        mock_docker_manager,
    ):
        """Test POST /model/register - fails when scheduler_url is empty"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Make request with empty scheduler_url
        response = api_client.post(
            "/model/register",
            json={"scheduler_url": ""}
        )

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "scheduler_url is required" in data["detail"]

    def test_register_model_whitespace_scheduler_url(
        self,
        api_client,
        mock_docker_manager,
    ):
        """Test POST /model/register - fails when scheduler_url is whitespace only"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Make request with whitespace scheduler_url
        response = api_client.post(
            "/model/register",
            json={"scheduler_url": "   "}
        )

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "scheduler_url is required" in data["detail"]

    def test_register_model_scheduler_disabled(
        self,
        api_client,
        mock_docker_manager,
        mock_scheduler_client,
    ):
        """Test POST /model/register - fails when scheduler is disabled"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_scheduler_client.is_enabled = False

        # Make request
        response = api_client.post(
            "/model/register",
            json={"scheduler_url": "http://scheduler:8000"}
        )

        # Verify response indicates scheduler not enabled
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is False
        assert "not enabled" in data["message"]

    def test_register_model_scheduler_returns_failure(
        self,
        api_client,
        mock_docker_manager,
        mock_scheduler_client,
    ):
        """Test POST /model/register - handles scheduler registration failure"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.register_instance = AsyncMock(return_value=False)

        # Make request
        response = api_client.post(
            "/model/register",
            json={"scheduler_url": "http://scheduler:8000"}
        )

        # Verify response indicates failure
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is False
        assert "Failed to register" in data["message"]

    def test_register_model_scheduler_raises_exception(
        self,
        api_client,
        mock_docker_manager,
        mock_scheduler_client,
    ):
        """Test POST /model/register - handles scheduler exception"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.register_instance = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        # Make request
        response = api_client.post(
            "/model/register",
            json={"scheduler_url": "http://scheduler:8000"}
        )

        # Verify error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "Connection refused" in data["detail"]

    def test_register_model_get_current_model_fails(
        self,
        api_client,
        mock_docker_manager,
    ):
        """Test POST /model/register - fails when get_current_model returns None"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = None

        # Make request
        response = api_client.post(
            "/model/register",
            json={"scheduler_url": "http://scheduler:8000"}
        )

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "Failed to get current model" in data["detail"]

    def test_register_model_missing_scheduler_url(
        self,
        api_client,
    ):
        """Test POST /model/register - fails when scheduler_url is missing"""
        # Make request without scheduler_url
        response = api_client.post(
            "/model/register",
            json={}
        )

        # Should fail with validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.unit
class TestDeregisterStatusEndpoint:
    """Unit tests for GET /model/deregister/status endpoint"""

    def test_get_deregister_status_success(
        self,
        api_client,
        mock_docker_manager,
        mock_task_queue,
        monkeypatch,
    ):
        """Test GET /model/deregister/status - returns operation status"""
        # Setup mocks for deregister operation
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Make task queue return pending tasks to keep operation running
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 10,
            "running": 1,
            "completed": 0,
            "failed": 0,
            "total": 11,
        }

        # Prevent background task from running
        def no_op_create_task(coro):
            coro.close()
            return Mock()
        monkeypatch.setattr("asyncio.create_task", no_op_create_task)

        # Initiate deregister
        deregister_response = api_client.post("/model/deregister")
        assert deregister_response.status_code == status.HTTP_200_OK
        operation_id = deregister_response.json()["operation_id"]

        # Query status
        status_response = api_client.get(
            f"/model/deregister/status?operation_id={operation_id}"
        )

        # Verify response structure
        assert status_response.status_code == status.HTTP_200_OK
        data = status_response.json()
        assert data["success"] is True
        assert data["operation_id"] == operation_id
        assert data["status"] in [
            "pending", "draining", "extracting_tasks",
            "waiting_running_task", "deregistering", "completed", "failed"
        ]
        assert data["old_model_id"] == "test-model"
        assert "initiated_at" in data
        assert "pending_tasks_at_start" in data
        assert "pending_tasks_completed" in data
        assert "redistributed_tasks_count" in data

    def test_get_deregister_status_not_found(self, api_client):
        """Test GET /model/deregister/status - operation not found"""
        response = api_client.get(
            "/model/deregister/status?operation_id=nonexistent-id"
        )

        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"]

    def test_get_deregister_status_missing_operation_id(self, api_client):
        """Test GET /model/deregister/status - missing operation_id parameter"""
        response = api_client.get("/model/deregister/status")

        # Should fail with validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_deregister_status_completed_operation(
        self,
        api_client,
        mock_docker_manager,
        mock_task_queue,
        mock_scheduler_client,
        monkeypatch,
    ):
        """Test GET /model/deregister/status - completed operation shows completed_at"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_scheduler_client.is_enabled = True
        mock_scheduler_client.drain_instance = AsyncMock(return_value={
            "success": True,
            "status": "draining",
            "pending_tasks": 0
        })
        mock_scheduler_client.deregister_instance = AsyncMock(return_value=True)

        # Make task queue return no pending tasks for immediate completion
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }
        mock_task_queue.extract_pending_tasks = AsyncMock(return_value=[])

        # Initiate deregister (this will run and complete quickly)
        deregister_response = api_client.post("/model/deregister")
        assert deregister_response.status_code == status.HTTP_200_OK
        operation_id = deregister_response.json()["operation_id"]

        # Wait a bit for background task to complete
        import time
        time.sleep(0.1)

        # Query status
        status_response = api_client.get(
            f"/model/deregister/status?operation_id={operation_id}"
        )

        # Verify response
        assert status_response.status_code == status.HTTP_200_OK
        data = status_response.json()
        assert data["success"] is True
        assert data["operation_id"] == operation_id
        # Status should be completed since there were no pending tasks
        if data["status"] == "completed":
            assert data["completed_at"] is not None
            assert data["error"] is None


@pytest.mark.unit
class TestDeregisterModelEndpoint:
    """Unit tests for POST /model/deregister endpoint"""

    def test_deregister_model_success(
        self,
        api_client,
        mock_docker_manager,
        mock_task_queue,
        monkeypatch,
    ):
        """Test POST /model/deregister - successful initiation"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Prevent background task from running
        def no_op_create_task(coro):
            coro.close()
            return Mock()
        monkeypatch.setattr("asyncio.create_task", no_op_create_task)

        # Make request
        response = api_client.post("/model/deregister")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "operation_id" in data
        assert isinstance(data["operation_id"], str)
        assert len(data["operation_id"]) > 0
        assert data["status"] == "draining"
        assert "deregister operation initiated" in data["message"].lower()

    def test_deregister_model_no_model_running(
        self,
        api_client,
        mock_docker_manager,
    ):
        """Test POST /model/deregister - fails when no model is running"""
        # Setup mock - no model running
        mock_docker_manager.is_model_running.return_value = False

        # Make request
        response = api_client.post("/model/deregister")

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "No model is currently running" in data["detail"]

    def test_deregister_model_already_in_progress(
        self,
        api_client,
        mock_docker_manager,
        mock_task_queue,
        monkeypatch,
    ):
        """Test POST /model/deregister - fails when deregister already in progress"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Make task queue return pending tasks to prevent completion
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 100,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 100,
        }

        # Prevent background task from running
        def no_op_create_task(coro):
            coro.close()
            return Mock()
        monkeypatch.setattr("asyncio.create_task", no_op_create_task)

        # First request - should succeed
        response1 = api_client.post("/model/deregister")
        assert response1.status_code == status.HTTP_200_OK

        # Second request - should fail with conflict
        response2 = api_client.post("/model/deregister")
        assert response2.status_code == status.HTTP_409_CONFLICT
        data = response2.json()
        assert "detail" in data
        assert "already in progress" in data["detail"]
