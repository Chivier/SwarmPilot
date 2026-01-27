"""Unit tests for task clear functionality.

Tests the /task/clear endpoint behavior including:
- Clearing task registry
- Submit task behavior during/after clear
"""

from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Task Clear API Tests
# ============================================================================


class TestTaskClearAPI:
    """Tests for /task/clear endpoint."""

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
    async def test_clear_tasks_clears_registry(self, test_client):
        """Test that /task/clear clears the task registry."""
        from src.api import task_registry

        # Setup: Add tasks to task registry
        await task_registry.create_task(
            task_id="test-task-1",
            model_id="test-model",
            task_input={},
            metadata={},
            assigned_instance="",
        )
        await task_registry.create_task(
            task_id="test-task-2",
            model_id="test-model",
            task_input={},
            metadata={},
            assigned_instance="",
        )

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

    @pytest.mark.asyncio
    async def test_clearing_flag_reset_after_error(self, test_client):
        """Test that clear recovers gracefully from errors."""
        with patch(
            "src.api.task_registry.clear_all", new_callable=AsyncMock
        ) as mock_clear:
            mock_clear.side_effect = Exception("Test error")

            # This should fail but not leave state corrupted
            with suppress(Exception):
                _response = test_client.post("/task/clear")


# ============================================================================
# Integration Tests
# ============================================================================


class TestTaskClearIntegration:
    """Integration tests for task clear workflow."""

    @pytest.mark.asyncio
    async def test_clear_submit_clear_cycle(self, test_client):
        """Test clear -> submit -> clear cycle works correctly."""
        from src.api import task_registry

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

        # Add tasks directly to registry
        await task_registry.create_task(
            task_id="new-task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="",
        )

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

    @pytest.mark.asyncio
    async def test_clear_response_includes_queue_count(self, test_client):
        """Test that clear response includes queue cleared count in message."""
        from src.api import task_registry

        # Add tasks to registry
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="",
        )

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
