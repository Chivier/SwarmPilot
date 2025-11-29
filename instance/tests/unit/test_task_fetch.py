"""
Unit tests for GET /task/fetch endpoint
"""

import pytest
from unittest.mock import AsyncMock
from fastapi import status

from src.models import Task, TaskStatus


@pytest.mark.unit
class TestTaskFetchEndpoint:
    """Test suite for GET /task/fetch endpoint"""

    def test_fetch_task_success(self, api_client, mock_task_queue):
        """Test successful task fetch returns exist=True and task_id"""
        # Setup mock - return a task
        task = Task(
            task_id="task-123",
            model_id="test-model",
            task_input={"prompt": "test"},
            status=TaskStatus.FETCHED,
        )
        mock_task_queue.fetch_task = AsyncMock(return_value=task)

        # Make request
        response = api_client.get("/task/fetch")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["exist"] is True
        assert data["task_id"] == "task-123"

        # Verify fetch_task was called
        mock_task_queue.fetch_task.assert_called_once()

    def test_fetch_task_empty_queue(self, api_client, mock_task_queue):
        """Test fetch on empty queue returns exist=False"""
        # Setup mock - return None (no task available)
        mock_task_queue.fetch_task = AsyncMock(return_value=None)

        # Make request
        response = api_client.get("/task/fetch")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["exist"] is False
        assert data["task_id"] == ""

    def test_fetch_task_all_running(self, api_client, mock_task_queue):
        """Test fetch when all tasks are running returns exist=False"""
        # Setup mock - return None (all tasks are running, none available)
        mock_task_queue.fetch_task = AsyncMock(return_value=None)

        # Make request
        response = api_client.get("/task/fetch")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["exist"] is False
        assert data["task_id"] == ""

    def test_fetch_task_no_body_required(self, api_client, mock_task_queue):
        """Test that fetch endpoint is a GET with no body required"""
        # Setup mock
        mock_task_queue.fetch_task = AsyncMock(return_value=None)

        # GET request with no body should work
        response = api_client.get("/task/fetch")

        assert response.status_code == status.HTTP_200_OK

    def test_fetch_task_response_format(self, api_client, mock_task_queue):
        """Test that response format matches specification exactly"""
        # Setup mock
        task = Task(
            task_id="test-task-abc",
            model_id="model-1",
            task_input={"prompt": "test"},
            status=TaskStatus.FETCHED,
        )
        task.enqueue_time = 1700000000.0
        task.submitted_at = "2023-11-14T22:13:20Z"
        mock_task_queue.fetch_task = AsyncMock(return_value=task)

        # Make request
        response = api_client.get("/task/fetch")

        # Verify exact response structure (includes full task details for redistribution)
        data = response.json()
        expected_keys = {"exist", "task_id", "model_id", "task_input", "enqueue_time", "submitted_at"}
        assert set(data.keys()) == expected_keys
        assert isinstance(data["exist"], bool)
        assert isinstance(data["task_id"], str)
        assert data["model_id"] == "model-1"
        assert data["task_input"] == {"prompt": "test"}
        assert data["enqueue_time"] == 1700000000.0
        assert data["submitted_at"] == "2023-11-14T22:13:20Z"
