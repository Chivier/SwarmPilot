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
        mock_scheduler_client,
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
        mock_docker_manager.stop_model = AsyncMock(return_value="test-model")

        # Make task queue return pending tasks
        mock_task_queue.get_queue_stats.return_value = {
            "queued": 10,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 10,
        }
        mock_task_queue.current_task_id = None  # No running task to skip wait

        mock_scheduler_client.is_enabled = False

        # Initiate deregister - now completes synchronously
        deregister_response = api_client.post("/model/deregister")
        assert deregister_response.status_code == status.HTTP_200_OK

        # Verify response structure - deregister completes synchronously now
        response_data = deregister_response.json()
        assert response_data["success"] is True
        assert response_data["model_id"] == "test-model"
        assert "redistributed_tasks_count" in response_data

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
    ):
        """Test POST /model/deregister - completes synchronously with no running tasks"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_docker_manager.stop_model = AsyncMock(return_value="test-model")
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
        mock_task_queue.current_task_id = None  # No running task
        mock_task_queue.extract_pending_tasks = AsyncMock(return_value=[])

        # Deregister completes synchronously
        deregister_response = api_client.post("/model/deregister")
        assert deregister_response.status_code == status.HTTP_200_OK

        # Verify response
        data = deregister_response.json()
        assert data["success"] is True
        assert data["model_id"] == "test-model"


@pytest.mark.unit
class TestDeregisterModelEndpoint:
    """Unit tests for POST /model/deregister endpoint"""

    def test_deregister_model_success(
        self,
        api_client,
        mock_docker_manager,
        mock_task_queue,
        mock_scheduler_client,
    ):
        """Test POST /model/deregister - successful completion"""
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_docker_manager.stop_model = AsyncMock(return_value="test-model")
        mock_task_queue.current_task_id = None  # No running task
        mock_scheduler_client.is_enabled = False

        # Make request
        response = api_client.post("/model/deregister")

        # Verify response - deregister completes synchronously now
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["model_id"] == "test-model"
        assert "redistributed_tasks_count" in data

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

    def test_deregister_model_sequential_requests(
        self,
        api_client,
        mock_docker_manager,
        mock_task_queue,
        mock_scheduler_client,
    ):
        """Test POST /model/deregister - sequential requests with synchronous operation.

        With synchronous deregister (blocking until completion), the first request
        completes before the second one starts. The second request should fail
        because no model is running after the first deregister.
        """
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_docker_manager.stop_model = AsyncMock(return_value="test-model")
        mock_task_queue.current_task_id = None  # No running task
        mock_scheduler_client.is_enabled = False

        # First request - should succeed
        response1 = api_client.post("/model/deregister")
        assert response1.status_code == status.HTTP_200_OK
        data1 = response1.json()
        assert data1["success"] is True

        # After first deregister, no model is running
        mock_docker_manager.is_model_running.return_value = False

        # Second request - should fail because no model is running
        response2 = api_client.post("/model/deregister")
        assert response2.status_code == status.HTTP_400_BAD_REQUEST
        data2 = response2.json()
        assert "detail" in data2
        assert "No model is currently running" in data2["detail"]


@pytest.mark.unit
class TestDeregisterTimeoutBehavior:
    """Tests for deregister endpoint timeout behavior.

    These tests verify the 10-second timeout for waiting on running tasks:
    - If task completes within 10s: normal flow continues
    - If task exceeds 10s: task is detached and deregister proceeds
    """

    def test_deregister_no_running_task_proceeds_immediately(
        self,
        api_client,
        mock_docker_manager,
        mock_task_queue,
        mock_scheduler_client,
    ):
        """Test deregister proceeds immediately when no task is running.

        When current_task_id is None, the endpoint should skip the wait
        loop entirely and proceed with deregistration.
        """
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_docker_manager.stop_model = AsyncMock(return_value="test-model")
        mock_task_queue.current_task_id = None  # No running task
        mock_task_queue.extract_pending_tasks = AsyncMock(return_value=[])
        mock_scheduler_client.is_enabled = False

        # Make request
        response = api_client.post("/model/deregister")

        # Verify success
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True

    def test_deregister_detaches_task_after_timeout(
        self,
        api_client,
        mock_docker_manager,
        mock_task_queue,
        mock_scheduler_client,
        monkeypatch,
    ):
        """Test deregister detaches task when 10s timeout is reached.

        When a task is running and doesn't complete within 10s,
        detach_current_task() should be called to allow deregister to proceed.
        """
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_docker_manager.stop_model = AsyncMock(return_value="test-model")

        # Task that never completes (always returns same task_id)
        mock_task_queue.current_task_id = "long-running-task"
        mock_task_queue.extract_pending_tasks = AsyncMock(return_value=[])
        mock_task_queue.detach_current_task = AsyncMock(return_value="long-running-task")

        mock_scheduler_client.is_enabled = False

        # Mock time to simulate timeout quickly
        import time
        start_time = time.time()
        call_count = [0]

        def mock_time():
            call_count[0] += 1
            # After first few calls, simulate timeout by returning start + 11 seconds
            if call_count[0] > 2:
                return start_time + 11
            return start_time

        monkeypatch.setattr("time.time", mock_time)

        # Mock asyncio.sleep to not actually sleep
        async def mock_sleep(seconds):
            pass

        monkeypatch.setattr("asyncio.sleep", mock_sleep)

        # Make request
        response = api_client.post("/model/deregister")

        # Verify detach was called
        assert mock_task_queue.detach_current_task.called
        assert response.status_code == status.HTTP_200_OK

    def test_deregister_waits_for_task_completion_within_timeout(
        self,
        api_client,
        mock_docker_manager,
        mock_task_queue,
        mock_scheduler_client,
        monkeypatch,
    ):
        """Test deregister waits and task completes within 10s timeout.

        When a task completes before the timeout, detach_current_task()
        should NOT be called.
        """
        # Setup mocks
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        mock_docker_manager.stop_model = AsyncMock(return_value="test-model")

        # Task that completes after first check
        check_count = [0]

        @property
        def get_current_task_id(self):
            check_count[0] += 1
            if check_count[0] > 1:
                return None  # Task completed
            return "quick-task"

        # Use a mock that simulates task completing
        mock_task_queue.extract_pending_tasks = AsyncMock(return_value=[])
        mock_task_queue.detach_current_task = AsyncMock(return_value=None)

        # First access returns task_id, subsequent returns None
        type(mock_task_queue).current_task_id = property(
            lambda self: "quick-task" if check_count[0] == 0 else None
        )

        mock_scheduler_client.is_enabled = False

        # Track if detach was called
        detach_called = [False]
        original_detach = mock_task_queue.detach_current_task

        async def tracking_detach():
            detach_called[0] = True
            return await original_detach()

        mock_task_queue.detach_current_task = tracking_detach

        # Mock asyncio.sleep to increment check_count
        async def mock_sleep(seconds):
            check_count[0] += 1

        monkeypatch.setattr("asyncio.sleep", mock_sleep)

        # Make request
        response = api_client.post("/model/deregister")

        # Verify success without needing to detach
        assert response.status_code == status.HTTP_200_OK
