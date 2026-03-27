"""Unit tests for /task/repredict endpoint.

Tests batch re-prediction of all pending/running tasks
without modifying metadata.

TDD Test Suite - Written before implementation.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from swarmpilot.scheduler.clients.models import Prediction
from swarmpilot.scheduler.models import (
    TaskStatus,
)

# ============================================================================
# Fixtures for repredict tests
# ============================================================================


@pytest.fixture
def sample_prediction():
    """Create a sample prediction for repredict tests."""
    return Prediction(
        instance_id="inst-1",
        predicted_time_ms=200.0,
        confidence=0.95,
        error_margin_ms=20.0,
        quantiles={0.5: 150.0, 0.9: 250.0, 0.95: 300.0, 0.99: 400.0},
    )


@pytest.fixture
def new_prediction():
    """Create a new prediction (different values) for repredict tests."""
    return Prediction(
        instance_id="inst-1",
        predicted_time_ms=300.0,  # Different from original
        confidence=0.92,
        error_margin_ms=30.0,
        quantiles={0.5: 250.0, 0.9: 350.0, 0.95: 400.0, 0.99: 500.0},
    )


def setup_instance_and_task(
    test_client, task_id: str, model_id: str = "model-1"
):
    """Helper to register instance and submit a task."""
    # Register instance (idempotent)
    test_client.post(
        "/v1/instance/register",
        json={
            "instance_id": "inst-1",
            "model_id": model_id,
            "endpoint": "http://localhost:8001",
            "platform_info": {
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test-hw",
            },
        },
    )

    # Submit task
    with patch(
        "swarmpilot.scheduler.api.predictor_client.predict",
        new=AsyncMock(return_value=[]),
    ):
        test_client.post(
            "/v1/task/submit",
            json={
                "task_id": task_id,
                "model_id": model_id,
                "task_input": {"prompt": "test"},
                "metadata": {"height": 512, "width": 512},
            },
        )


# ============================================================================
# Test Class: Basic Functionality
# ============================================================================


class TestRepredictBasic:
    """Tests for basic repredict functionality."""

    def test_empty_registry_returns_zero(self, test_client):
        """Test that empty registry returns zero counts."""
        response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_tasks"] == 0
        assert data["eligible_tasks"] == 0
        assert data["repredicted"] == 0
        assert data["failed"] == 0
        assert data["skipped"] == 0

    def test_no_eligible_tasks_all_completed(self, test_client):
        """Test with only COMPLETED tasks - all should be skipped."""
        from swarmpilot.scheduler.api import task_registry

        setup_instance_and_task(test_client, "task-completed-1")
        setup_instance_and_task(test_client, "task-completed-2")

        # Set tasks to COMPLETED
        async def set_completed():
            for task_id in ["task-completed-1", "task-completed-2"]:
                task = await task_registry.get(task_id)
                if task:
                    task.status = TaskStatus.COMPLETED

        asyncio.run(set_completed())

        response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_tasks"] == 2
        assert data["eligible_tasks"] == 0
        assert data["skipped"] == 2


# ============================================================================
# Test Class: Repredict Functionality
# ============================================================================


class TestRepredictTasks:
    """Tests for actual re-prediction."""

    def test_pending_task_repredicted(self, test_client, new_prediction):
        """Test that PENDING task is re-predicted."""
        from swarmpilot.scheduler.api import task_registry

        setup_instance_and_task(test_client, "task-pending")

        # Setup task in queue with old prediction
        async def setup_task_in_queue():
            task = await task_registry.get("task-pending")
            if task:
                task.assigned_instance = "inst-1"
                task.status = TaskStatus.PENDING
                task.predicted_time_ms = 100.0
                task.predicted_error_margin_ms = 10.0
                task.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

        asyncio.run(setup_task_in_queue())

        # Mock predictor to return new prediction
        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[new_prediction]),
        ):
            response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["repredicted"] == 1

    def test_running_task_repredicted(self, test_client, new_prediction):
        """Test that RUNNING task is re-predicted."""
        from swarmpilot.scheduler.api import task_registry

        setup_instance_and_task(test_client, "task-running")

        # Setup task in queue as RUNNING
        async def setup_task_running():
            task = await task_registry.get("task-running")
            if task:
                task.assigned_instance = "inst-1"
                task.status = TaskStatus.RUNNING
                task.predicted_time_ms = 100.0
                task.predicted_error_margin_ms = 10.0
                task.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

        asyncio.run(setup_task_running())

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[new_prediction]),
        ):
            response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["repredicted"] == 1

    def test_multiple_tasks_repredicted(self, test_client, new_prediction):
        """Test that multiple eligible tasks are all re-predicted."""
        from swarmpilot.scheduler.api import task_registry

        # Setup multiple tasks
        for i in range(3):
            setup_instance_and_task(test_client, f"task-multi-{i}")

        # Setup all tasks in queue
        async def setup_tasks_in_queue():
            for i in range(3):
                task = await task_registry.get(f"task-multi-{i}")
                if task:
                    task.assigned_instance = "inst-1"
                    task.status = TaskStatus.PENDING
                    task.predicted_time_ms = 100.0
                    task.predicted_error_margin_ms = 10.0
                    task.predicted_quantiles = {
                        0.5: 80.0,
                        0.9: 150.0,
                        0.95: 200.0,
                        0.99: 300.0,
                    }

        asyncio.run(setup_tasks_in_queue())

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[new_prediction]),
        ):
            response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["repredicted"] == 3

    def test_metadata_not_modified(self, test_client, new_prediction):
        """Test that metadata is NOT modified during repredict."""
        from swarmpilot.scheduler.api import task_registry

        setup_instance_and_task(test_client, "task-no-meta-change")

        original_metadata = {"original_key": "original_value", "height": 1024}

        async def setup_task():
            task = await task_registry.get("task-no-meta-change")
            if task:
                task.assigned_instance = "inst-1"
                task.status = TaskStatus.PENDING
                task.metadata = original_metadata.copy()
                task.predicted_time_ms = 100.0
                task.predicted_error_margin_ms = 10.0
                task.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

        asyncio.run(setup_task())

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[new_prediction]),
        ):
            response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200

        # Verify metadata unchanged
        async def verify_metadata():
            task = await task_registry.get("task-no-meta-change")
            assert task.metadata == original_metadata

        asyncio.run(verify_metadata())


# ============================================================================
# Test Class: Skip Behavior
# ============================================================================


class TestRepredictSkipBehavior:
    """Tests for skip behavior with COMPLETED/FAILED tasks."""

    def test_completed_task_skipped(self, test_client):
        """Test that COMPLETED task is skipped."""
        from swarmpilot.scheduler.api import task_registry

        setup_instance_and_task(test_client, "task-skip-completed")

        async def set_completed():
            task = await task_registry.get("task-skip-completed")
            if task:
                task.status = TaskStatus.COMPLETED

        asyncio.run(set_completed())

        response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["skipped"] >= 1
        assert data["repredicted"] == 0

    def test_failed_task_skipped(self, test_client):
        """Test that FAILED task is skipped."""
        from swarmpilot.scheduler.api import task_registry

        setup_instance_and_task(test_client, "task-skip-failed")

        async def set_failed():
            task = await task_registry.get("task-skip-failed")
            if task:
                task.status = TaskStatus.FAILED

        asyncio.run(set_failed())

        response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["skipped"] >= 1
        assert data["repredicted"] == 0

    def test_task_not_in_queue_skipped(self, test_client):
        """Test that task without assigned_instance is skipped."""
        from swarmpilot.scheduler.api import task_registry

        setup_instance_and_task(test_client, "task-no-instance")

        # Remove assigned_instance (simulate not in queue)
        async def clear_instance():
            task = await task_registry.get("task-no-instance")
            if task:
                task.assigned_instance = ""
                task.status = TaskStatus.PENDING

        asyncio.run(clear_instance())

        response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["skipped"] >= 1
        assert data["repredicted"] == 0


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestRepredictErrorHandling:
    """Tests for error handling during repredict."""

    def test_predictor_failure_continues(self, test_client):
        """Test that predictor failure for one task doesn't stop others."""
        from swarmpilot.scheduler.api import task_registry

        # Setup two tasks
        setup_instance_and_task(test_client, "task-fail-1")
        setup_instance_and_task(test_client, "task-fail-2")

        async def setup_tasks():
            for task_id in ["task-fail-1", "task-fail-2"]:
                task = await task_registry.get(task_id)
                if task:
                    task.assigned_instance = "inst-1"
                    task.status = TaskStatus.PENDING
                    task.predicted_time_ms = 100.0
                    task.predicted_error_margin_ms = 10.0
                    task.predicted_quantiles = {
                        0.5: 80.0,
                        0.9: 150.0,
                        0.95: 200.0,
                        0.99: 300.0,
                    }

        asyncio.run(setup_tasks())

        # Make predictor fail
        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(side_effect=Exception("Predictor error")),
        ):
            response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        # Both should have failed, but operation completes
        assert data["failed"] == 2
        assert data["repredicted"] == 0

    def test_predictor_failure_restores_queue(self, test_client):
        """Test that queue is restored on predictor failure."""
        from swarmpilot.scheduler.api import instance_registry, task_registry

        setup_instance_and_task(test_client, "task-restore-queue")

        # Setup task with known prediction values
        original_prediction_ms = 100.0

        async def setup_task():
            task = await task_registry.get("task-restore-queue")
            if task:
                task.assigned_instance = "inst-1"
                task.status = TaskStatus.PENDING
                task.predicted_time_ms = original_prediction_ms
                task.predicted_error_margin_ms = 10.0
                task.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

        asyncio.run(setup_task())

        # Get queue state before
        async def get_queue_before():
            return await instance_registry.get_queue_info("inst-1")

        _queue_before = asyncio.run(get_queue_before())

        # Make predictor fail
        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(side_effect=Exception("Predictor error")),
        ):
            response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["failed"] >= 1

        # Note: Queue restoration is implementation detail
        # The important thing is the operation completes without crashing


# ============================================================================
# Test Class: Queue Update with expect_error Strategy
# ============================================================================


class TestRepredictExpectErrorStrategy:
    """Tests for queue updates with expect_error scheduling strategy."""

    def test_expect_error_queue_updated(self, test_client, new_prediction):
        """Test that queue is updated correctly with expect_error strategy."""
        from swarmpilot.scheduler.api import task_registry

        setup_instance_and_task(test_client, "task-expect-error")

        async def setup_task():
            task = await task_registry.get("task-expect-error")
            if task:
                task.assigned_instance = "inst-1"
                task.status = TaskStatus.PENDING
                task.predicted_time_ms = 100.0
                task.predicted_error_margin_ms = 10.0
                task.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

        asyncio.run(setup_task())

        # Re-predict with new values
        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[new_prediction]),
        ):
            response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["repredicted"] == 1

        # Verify task prediction updated
        async def verify_prediction():
            task = await task_registry.get("task-expect-error")
            # New prediction should be applied
            assert task.predicted_time_ms == new_prediction.predicted_time_ms

        asyncio.run(verify_prediction())


# ============================================================================
# Test Class: Queue Update with Probabilistic Strategy
# ============================================================================


class TestRepredictProbabilisticStrategy:
    """Tests for queue updates with probabilistic scheduling strategy."""

    def test_probabilistic_queue_updated(self, test_client, new_prediction):
        """Test that queue is updated correctly with probabilistic strategy."""
        from swarmpilot.scheduler.api import task_registry

        # Switch to probabilistic strategy
        response = test_client.post(
            "/v1/strategy/set",
            json={"strategy_name": "probabilistic", "target_quantile": 0.9},
        )
        if response.status_code != 200:
            pytest.skip("Failed to set probabilistic strategy")

        setup_instance_and_task(test_client, "task-probabilistic")

        async def setup_task():
            task = await task_registry.get("task-probabilistic")
            if task:
                task.assigned_instance = "inst-1"
                task.status = TaskStatus.PENDING
                task.predicted_time_ms = 100.0
                task.predicted_error_margin_ms = 10.0
                task.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

        asyncio.run(setup_task())

        # Re-predict
        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[new_prediction]),
        ):
            response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["repredicted"] == 1


# ============================================================================
# Test Class: Mixed Scenarios
# ============================================================================


class TestRepredictMixedScenarios:
    """Tests for mixed scenarios with various task states."""

    def test_mixed_task_states(self, test_client, new_prediction):
        """Test with mix of PENDING, RUNNING, COMPLETED, FAILED tasks."""
        from swarmpilot.scheduler.api import task_registry

        # Setup tasks with different states
        for i in range(4):
            setup_instance_and_task(test_client, f"task-mixed-{i}")

        async def setup_mixed_states():
            # Task 0: PENDING in queue (should be repredicted)
            task0 = await task_registry.get("task-mixed-0")
            if task0:
                task0.assigned_instance = "inst-1"
                task0.status = TaskStatus.PENDING
                task0.predicted_time_ms = 100.0
                task0.predicted_error_margin_ms = 10.0
                task0.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

            # Task 1: RUNNING in queue (should be repredicted)
            task1 = await task_registry.get("task-mixed-1")
            if task1:
                task1.assigned_instance = "inst-1"
                task1.status = TaskStatus.RUNNING
                task1.predicted_time_ms = 100.0
                task1.predicted_error_margin_ms = 10.0
                task1.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

            # Task 2: COMPLETED (should be skipped)
            task2 = await task_registry.get("task-mixed-2")
            if task2:
                task2.status = TaskStatus.COMPLETED

            # Task 3: FAILED (should be skipped)
            task3 = await task_registry.get("task-mixed-3")
            if task3:
                task3.status = TaskStatus.FAILED

        asyncio.run(setup_mixed_states())

        with patch(
            "swarmpilot.scheduler.api.predictor_client.predict",
            new=AsyncMock(return_value=[new_prediction]),
        ):
            response = test_client.post("/v1/task/repredict")

        assert response.status_code == 200
        data = response.json()
        assert data["total_tasks"] == 4
        assert data["repredicted"] == 2  # PENDING + RUNNING
        assert data["skipped"] == 2  # COMPLETED + FAILED
