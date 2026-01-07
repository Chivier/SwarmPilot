"""Unit tests for task clear functionality.

Tests the fix for the central queue not being cleared when /task/clear is called.
This includes:
- CentralTaskQueue.clear() method
- Clearing flag behavior during clear operation
- Submit task rejection during clear
"""

from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.central_queue import CentralTaskQueue

# ============================================================================
# CentralTaskQueue.clear() Tests
# ============================================================================


class TestCentralQueueClear:
    """Tests for CentralTaskQueue.clear() method."""

    @pytest.fixture
    def central_queue(self, task_registry, instance_registry):
        """Create a CentralTaskQueue for testing."""
        return CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

    @pytest.mark.asyncio
    async def test_clear_empty_queue(self, central_queue):
        """Test clearing an empty queue returns 0."""
        count = await central_queue.clear()
        assert count == 0
        assert await central_queue.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_clear_queue_with_single_task(self, central_queue):
        """Test clearing a queue with one task."""
        await central_queue.enqueue(
            task_id="task-1",
            model_id="model-1",
            task_input={"data": "test"},
            metadata={},
        )

        assert await central_queue.get_queue_size() == 1

        count = await central_queue.clear()

        assert count == 1
        assert await central_queue.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_clear_queue_with_multiple_tasks(self, central_queue):
        """Test clearing a queue with multiple tasks."""
        for i in range(5):
            await central_queue.enqueue(
                task_id=f"task-{i}",
                model_id="model-1",
                task_input={"index": i},
                metadata={},
            )

        assert await central_queue.get_queue_size() == 5

        count = await central_queue.clear()

        assert count == 5
        assert await central_queue.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_clear_then_enqueue(self, central_queue):
        """Test that enqueue works correctly after clear."""
        # Add and clear
        await central_queue.enqueue("task-1", "model-1", {}, {})
        await central_queue.enqueue("task-2", "model-1", {}, {})
        await central_queue.clear()

        assert await central_queue.get_queue_size() == 0

        # Enqueue new task after clear
        await central_queue.enqueue("task-3", "model-1", {}, {})

        assert await central_queue.get_queue_size() == 1

        # Verify it's the new task
        info = await central_queue.get_queue_info()
        assert info["total_size"] == 1

    @pytest.mark.asyncio
    async def test_clear_multiple_times(self, central_queue):
        """Test clearing multiple times is safe."""
        await central_queue.enqueue("task-1", "model-1", {}, {})

        count1 = await central_queue.clear()
        assert count1 == 1

        count2 = await central_queue.clear()
        assert count2 == 0

        count3 = await central_queue.clear()
        assert count3 == 0

    @pytest.mark.asyncio
    async def test_clear_different_models(self, central_queue):
        """Test clearing tasks from different models."""
        await central_queue.enqueue("task-a", "model-a", {}, {})
        await central_queue.enqueue("task-b", "model-b", {}, {})
        await central_queue.enqueue("task-c", "model-a", {}, {})

        info = await central_queue.get_queue_info()
        assert info["by_model"]["model-a"] == 2
        assert info["by_model"]["model-b"] == 1

        count = await central_queue.clear()

        assert count == 3
        assert await central_queue.get_queue_size() == 0


# ============================================================================
# Task Clear API Tests
# ============================================================================


class TestTaskClearAPI:
    """Tests for /task/clear endpoint with clearing flag."""

    @pytest.fixture
    def mock_instance_response(self):
        """Create mock instance clear response."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "cleared_count": {
                "total": 3,
                "queued": 2,
                "completed": 1,
                "failed": 0,
            },
        }
        return mock_response

    @pytest.mark.asyncio
    async def test_clear_tasks_clears_central_queue(self, test_client):
        """Test that /task/clear clears the central queue."""
        from src.api import central_queue

        # First clear any existing tasks from previous tests
        await central_queue.clear()
        assert await central_queue.get_queue_size() == 0

        # Setup: Add tasks to central queue
        await central_queue.enqueue("test-task-1", "test-model", {}, {})
        await central_queue.enqueue("test-task-2", "test-model", {}, {})

        initial_size = await central_queue.get_queue_size()
        assert initial_size == 2

        # Call clear endpoint
        with patch("src.api.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    raise_for_status=MagicMock(),
                    json=MagicMock(
                        return_value={
                            "success": True,
                            "cleared_count": {"total": 0},
                        }
                    ),
                )
            )
            mock_client_class.return_value = mock_client

            response = test_client.post("/task/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "queued task(s)" in data["message"]

        # Verify central queue is cleared
        final_size = await central_queue.get_queue_size()
        assert final_size == 0

    @pytest.mark.asyncio
    async def test_submit_task_blocked_during_clear(self, test_client):
        """Test that task submission is blocked during clear operation."""
        import src.api as api_module

        # Set clearing flag
        api_module._clearing_in_progress = True

        try:
            response = test_client.post(
                "/task/submit",
                json={
                    "task_id": "blocked-task",
                    "model_id": "test-model",
                    "task_input": {"data": "test"},
                    "metadata": {},
                },
            )

            assert response.status_code == 503
            data = response.json()
            assert (
                "clear operation in progress" in data["detail"]["error"].lower()
            )

        finally:
            # Reset flag
            api_module._clearing_in_progress = False

    @pytest.mark.asyncio
    async def test_clearing_flag_reset_after_success(self, test_client):
        """Test that clearing flag is reset after successful clear."""
        import src.api as api_module

        # Ensure flag starts as False
        assert api_module._clearing_in_progress is False

        with patch("src.api.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    raise_for_status=MagicMock(),
                    json=MagicMock(
                        return_value={
                            "success": True,
                            "cleared_count": {"total": 0},
                        }
                    ),
                )
            )
            mock_client_class.return_value = mock_client

            response = test_client.post("/task/clear")

        assert response.status_code == 200

        # Flag should be reset
        assert api_module._clearing_in_progress is False

    @pytest.mark.asyncio
    async def test_clearing_flag_reset_after_error(self, test_client):
        """Test that clearing flag is reset even if clear fails."""
        import src.api as api_module

        with patch(
            "src.api.task_registry.clear_all", new_callable=AsyncMock
        ) as mock_clear:
            mock_clear.side_effect = Exception("Test error")

            # This should fail but flag should still be reset
            with suppress(Exception):
                _response = test_client.post("/task/clear")

        # Flag should be reset due to finally block
        assert api_module._clearing_in_progress is False

    @pytest.mark.asyncio
    async def test_concurrent_clear_rejected(self, test_client):
        """Test that concurrent clear operations are rejected."""
        import src.api as api_module

        # Simulate a clear in progress
        api_module._clearing_in_progress = True

        try:
            response = test_client.post("/task/clear")

            assert response.status_code == 409
            data = response.json()
            assert "already in progress" in data["detail"]["error"].lower()

        finally:
            api_module._clearing_in_progress = False


# ============================================================================
# Integration Tests
# ============================================================================


class TestTaskClearIntegration:
    """Integration tests for task clear workflow."""

    @pytest.mark.asyncio
    async def test_clear_submit_clear_cycle(self, test_client):
        """Test clear -> submit -> clear cycle works correctly."""
        import src.api as api_module
        from src.api import central_queue

        # Ensure clean state
        api_module._clearing_in_progress = False
        await central_queue.clear()  # Clear any leftover from previous tests

        with patch("src.api.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    raise_for_status=MagicMock(),
                    json=MagicMock(
                        return_value={
                            "success": True,
                            "cleared_count": {"total": 0},
                        }
                    ),
                )
            )
            mock_client_class.return_value = mock_client

            # First clear
            response1 = test_client.post("/task/clear")
            assert response1.status_code == 200

        # Verify queue is empty
        assert await central_queue.get_queue_size() == 0

        # Add tasks to queue (simulating submit path)
        await central_queue.enqueue("new-task-1", "model-1", {}, {})
        await central_queue.enqueue("new-task-2", "model-1", {}, {})
        assert await central_queue.get_queue_size() == 2

        with patch("src.api.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    raise_for_status=MagicMock(),
                    json=MagicMock(
                        return_value={
                            "success": True,
                            "cleared_count": {"total": 0},
                        }
                    ),
                )
            )
            mock_client_class.return_value = mock_client

            # Second clear
            response2 = test_client.post("/task/clear")
            assert response2.status_code == 200

        # Verify queue is empty again
        assert await central_queue.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_clear_response_includes_queue_count(self, test_client):
        """Test that clear response includes queue cleared count in message."""
        from src.api import central_queue

        # Clear any leftover from previous tests
        await central_queue.clear()

        # Add tasks to queue
        await central_queue.enqueue("task-1", "model-1", {}, {})
        await central_queue.enqueue("task-2", "model-1", {}, {})

        with patch("src.api.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    raise_for_status=MagicMock(),
                    json=MagicMock(
                        return_value={
                            "success": True,
                            "cleared_count": {"total": 0},
                        }
                    ),
                )
            )
            mock_client_class.return_value = mock_client

            response = test_client.post("/task/clear")

        assert response.status_code == 200
        data = response.json()
        # Message should mention queued tasks
        assert "queued task(s)" in data["message"]
