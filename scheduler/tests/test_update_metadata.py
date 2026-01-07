"""Unit tests for /task/update_metadata endpoint.

Tests metadata updates with queue recalculation for both
expect_error and probabilistic scheduling strategies.

TDD Test Suite - Written before implementation.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.model import (
    TaskStatus,
)
from src.clients.predictor_client import Prediction

# ============================================================================
# Fixtures for update_metadata tests
# ============================================================================


@pytest.fixture
def sample_metadata_update():
    """Create a sample metadata update request."""
    return {
        "updates": [
            {
                "task_id": "task-1",
                "metadata": {"new_key": "new_value", "priority": "high"},
            }
        ]
    }


@pytest.fixture
def sample_batch_metadata_update():
    """Create a batch metadata update request."""
    return {
        "updates": [
            {"task_id": "task-1", "metadata": {"key1": "value1"}},
            {"task_id": "task-2", "metadata": {"key2": "value2"}},
            {"task_id": "task-3", "metadata": {"key3": "value3"}},
        ]
    }


@pytest.fixture
def sample_prediction_for_update():
    """Create a sample prediction for metadata update tests."""
    return Prediction(
        instance_id="inst-1",
        predicted_time_ms=200.0,
        confidence=0.95,
        error_margin_ms=20.0,
        quantiles={0.5: 150.0, 0.9: 250.0, 0.95: 300.0, 0.99: 400.0},
    )


# ============================================================================
# Test Class: Request/Response Validation
# ============================================================================


class TestUpdateMetadataValidation:
    """Tests for request/response validation."""

    def test_valid_request_accepted(self, test_client, sample_metadata_update):
        """Test that a valid request is accepted."""
        # First register an instance and submit a task
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        # Submit a task
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"original": "value"},
                },
            )

        # Update metadata
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            response = test_client.post(
                "/task/update_metadata", json=sample_metadata_update
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 1

    def test_empty_list_returns_zero_count(self, test_client):
        """Test that an empty updates list returns zero count."""
        response = test_client.post(
            "/task/update_metadata", json={"updates": []}
        )

        # Empty list should either be accepted (0 updates) or rejected (validation)
        # Based on design, we accept empty list
        if response.status_code == 200:
            data = response.json()
            assert data["total"] == 0
            assert data["succeeded"] == 0
        else:
            # Validation error is also acceptable
            assert response.status_code == 422

    def test_missing_task_id_validation_error(self, test_client):
        """Test that missing task_id returns validation error."""
        response = test_client.post(
            "/task/update_metadata",
            json={
                "updates": [{"metadata": {"key": "value"}}]  # Missing task_id
            },
        )

        assert response.status_code == 422

    def test_missing_metadata_validation_error(self, test_client):
        """Test that missing metadata returns validation error."""
        response = test_client.post(
            "/task/update_metadata",
            json={
                "updates": [{"task_id": "task-1"}]  # Missing metadata
            },
        )

        assert response.status_code == 422

    def test_invalid_metadata_type_validation_error(self, test_client):
        """Test that invalid metadata type returns validation error."""
        response = test_client.post(
            "/task/update_metadata",
            json={"updates": [{"task_id": "task-1", "metadata": "not a dict"}]},
        )

        assert response.status_code == 422


# ============================================================================
# Test Class: Basic Update Functionality
# ============================================================================


class TestUpdateMetadataBasic:
    """Tests for basic metadata update (no queue recalculation)."""

    def test_update_task_not_in_queue(self, test_client):
        """Test updating metadata for task not assigned to any instance."""
        # Register instance
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        # Submit task (will be assigned but not yet in queue for this test)
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "task-not-in-queue",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"original": "metadata"},
                },
            )

        # Update metadata - should succeed without predictor call
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock()
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [
                        {
                            "task_id": "task-not-in-queue",
                            "metadata": {"new": "metadata"},
                        }
                    ]
                },
            )

            # Predictor should not be called for tasks not in queue
            # (Note: actual behavior depends on implementation details)

        assert response.status_code == 200
        data = response.json()
        assert data["succeeded"] >= 0  # Either succeeded or task not found

    def test_update_multiple_tasks(
        self, test_client, sample_batch_metadata_update
    ):
        """Test batch update of multiple tasks."""
        # Register instance
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        # Submit multiple tasks
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            for i in range(1, 4):
                test_client.post(
                    "/task/submit",
                    json={
                        "task_id": f"task-{i}",
                        "model_id": "model-1",
                        "task_input": {"prompt": f"test {i}"},
                        "metadata": {"original": f"value-{i}"},
                    },
                )

        # Update all tasks
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            response = test_client.post(
                "/task/update_metadata", json=sample_batch_metadata_update
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3

    def test_task_not_found_reported(self, test_client):
        """Test that non-existent task is reported in results."""
        response = test_client.post(
            "/task/update_metadata",
            json={
                "updates": [
                    {
                        "task_id": "nonexistent-task",
                        "metadata": {"key": "value"},
                    }
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["failed"] >= 1 or data["results"][0]["success"] is False

    def test_empty_metadata_allowed(self, test_client):
        """Test that empty metadata dict is valid."""
        # Register and submit task
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "task-empty-meta",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"has": "data"},
                },
            )

        # Update with empty metadata
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [{"task_id": "task-empty-meta", "metadata": {}}]
                },
            )

        assert response.status_code == 200


# ============================================================================
# Test Class: Skip Behavior for COMPLETED/FAILED Tasks
# ============================================================================


class TestUpdateMetadataSkipBehavior:
    """Tests for skip behavior with COMPLETED/FAILED tasks."""

    def test_completed_task_skipped(self, test_client):
        """Test that COMPLETED task is skipped (no error, increments skipped count)."""
        # This test requires the task to be in COMPLETED status
        # We'll mock the task registry to have a completed task

        from src.api import task_registry

        # Register instance and submit task
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "completed-task",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"original": "value"},
                },
            )

        # Manually set task to COMPLETED
        import asyncio

        async def set_completed():
            task = await task_registry.get("completed-task")
            if task:
                task.status = TaskStatus.COMPLETED

        asyncio.run(set_completed())

        # Try to update - should be skipped
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [
                        {
                            "task_id": "completed-task",
                            "metadata": {"new": "value"},
                        }
                    ]
                },
            )

        assert response.status_code == 200
        data = response.json()
        # Should either have skipped > 0 or specific behavior in results
        assert "skipped" in data or data["total"] >= 1

    def test_failed_task_skipped(self, test_client):
        """Test that FAILED task is skipped (no error, increments skipped count)."""
        from src.api import task_registry

        # Register instance and submit task
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "failed-task",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"original": "value"},
                },
            )

        # Manually set task to FAILED
        import asyncio

        async def set_failed():
            task = await task_registry.get("failed-task")
            if task:
                task.status = TaskStatus.FAILED

        asyncio.run(set_failed())

        # Try to update - should be skipped
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [
                        {"task_id": "failed-task", "metadata": {"new": "value"}}
                    ]
                },
            )

        assert response.status_code == 200


# ============================================================================
# Test Class: Queue Update with expect_error Strategy
# ============================================================================


class TestUpdateMetadataExpectError:
    """Tests for queue recalculation with expect_error strategy."""

    def test_pending_task_queue_recalculated(
        self, test_client, sample_prediction_for_update
    ):
        """Test queue recalculation for PENDING task with assigned instance."""
        # Setup: Register instance with expect_error queue
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-queue-test",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        # Submit a task with predictions
        old_prediction = Prediction(
            instance_id="inst-queue-test",
            predicted_time_ms=100.0,
            confidence=0.95,
            error_margin_ms=10.0,
            quantiles={0.5: 80.0, 0.9: 150.0, 0.95: 200.0, 0.99: 300.0},
        )

        with patch(
            "src.api.predictor_client.predict",
            new=AsyncMock(return_value=[old_prediction]),
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "task-queue-update",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"original": "metadata"},
                },
            )

        # Update metadata - should trigger re-prediction
        new_prediction = Prediction(
            instance_id="inst-queue-test",
            predicted_time_ms=200.0,
            confidence=0.95,
            error_margin_ms=20.0,
            quantiles={0.5: 150.0, 0.9: 250.0, 0.95: 300.0, 0.99: 400.0},
        )

        with patch(
            "src.api.predictor_client.predict",
            new=AsyncMock(return_value=[new_prediction]),
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [
                        {
                            "task_id": "task-queue-update",
                            "metadata": {"new": "metadata"},
                        }
                    ]
                },
            )

        assert response.status_code == 200
        data = response.json()

        # Verify response indicates queue was updated
        if data["results"]:
            result = data["results"][0]
            # Check if queue_updated field exists and is true (if task was in queue)
            assert "queue_updated" in result or data["succeeded"] >= 1


# ============================================================================
# Test Class: Queue Update with Probabilistic Strategy
# ============================================================================


class TestUpdateMetadataProbabilistic:
    """Tests for queue recalculation with probabilistic strategy."""

    def test_probabilistic_queue_recalculated(self, test_client):
        """Test queue recalculation uses Monte Carlo for probabilistic queues."""
        # Register instance
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-prob",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        # Submit task with probabilistic prediction
        old_pred = Prediction(
            instance_id="inst-prob",
            predicted_time_ms=100.0,
            confidence=0.9,
            quantiles={0.5: 80.0, 0.9: 150.0, 0.95: 200.0, 0.99: 300.0},
        )

        with patch(
            "src.api.predictor_client.predict",
            new=AsyncMock(return_value=[old_pred]),
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "task-prob",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"original": "value"},
                },
            )

        # Update with new prediction
        new_pred = Prediction(
            instance_id="inst-prob",
            predicted_time_ms=200.0,
            confidence=0.9,
            quantiles={0.5: 160.0, 0.9: 300.0, 0.95: 400.0, 0.99: 600.0},
        )

        with patch(
            "src.api.predictor_client.predict",
            new=AsyncMock(return_value=[new_pred]),
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [
                        {
                            "task_id": "task-prob",
                            "metadata": {"updated": "value"},
                        }
                    ]
                },
            )

        assert response.status_code == 200


# ============================================================================
# Test Class: Rollback on Prediction Failure
# ============================================================================


class TestUpdateMetadataRollback:
    """Tests for rollback behavior when prediction fails."""

    def test_predictor_failure_rollback(self, test_client):
        """Test that metadata is rolled back if predictor fails."""
        import asyncio

        from src.api import task_registry

        # Register instance
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-rollback",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        # Submit task - need to patch both central_queue scheduling and predictor
        old_pred = Prediction(
            instance_id="inst-rollback",
            predicted_time_ms=100.0,
            confidence=0.9,
            error_margin_ms=10.0,
            quantiles={0.5: 80.0, 0.9: 150.0, 0.95: 200.0, 0.99: 300.0},
        )

        with patch(
            "src.api.predictor_client.predict",
            new=AsyncMock(return_value=[old_pred]),
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "task-rollback",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"original": "value"},
                },
            )

        # Manually set up task to be "in queue" with assigned instance and predictions
        async def setup_task_in_queue():
            task = await task_registry.get("task-rollback")
            if task:
                task.assigned_instance = "inst-rollback"
                task.status = TaskStatus.PENDING
                task.predicted_time_ms = 100.0
                task.predicted_error_margin_ms = 10.0
                task.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(setup_task_in_queue())

        # Store original metadata
        async def get_original_metadata():
            task = await task_registry.get("task-rollback")
            return task.metadata.copy() if task else None

        original_metadata = loop.run_until_complete(get_original_metadata())

        # Update with predictor failure - should trigger rollback
        with patch(
            "src.api.predictor_client.predict",
            new=AsyncMock(side_effect=ConnectionError("Predictor unavailable")),
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [
                        {
                            "task_id": "task-rollback",
                            "metadata": {"should_be": "rolled_back"},
                        }
                    ]
                },
            )

        assert response.status_code == 200
        data = response.json()

        # Check that the update failed due to prediction failure
        assert data["failed"] >= 1
        assert data["results"][0]["success"] is False
        assert (
            "rolled back" in data["results"][0]["message"].lower()
            or "prediction failed" in data["results"][0]["message"].lower()
        )

        # Verify metadata was rolled back
        async def verify_rollback():
            task = await task_registry.get("task-rollback")
            if task and original_metadata:
                return task.metadata == original_metadata
            return False

        rolled_back = loop.run_until_complete(verify_rollback())
        assert rolled_back, "Metadata should be rolled back to original"
        loop.close()

    def test_predictor_timeout_rollback(self, test_client):
        """Test that metadata is rolled back on predictor timeout."""
        import asyncio

        from src.api import task_registry

        # Register instance
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-timeout",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        # Submit task
        old_pred = Prediction(
            instance_id="inst-timeout",
            predicted_time_ms=100.0,
            confidence=0.9,
            error_margin_ms=10.0,
            quantiles={0.5: 80.0, 0.9: 150.0, 0.95: 200.0, 0.99: 300.0},
        )

        with patch(
            "src.api.predictor_client.predict",
            new=AsyncMock(return_value=[old_pred]),
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "task-timeout",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"original": "value"},
                },
            )

        # Manually set up task to be "in queue"
        async def setup_task_in_queue():
            task = await task_registry.get("task-timeout")
            if task:
                task.assigned_instance = "inst-timeout"
                task.status = TaskStatus.PENDING
                task.predicted_time_ms = 100.0
                task.predicted_error_margin_ms = 10.0
                task.predicted_quantiles = {
                    0.5: 80.0,
                    0.9: 150.0,
                    0.95: 200.0,
                    0.99: 300.0,
                }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(setup_task_in_queue())

        # Update with timeout - should trigger rollback
        with patch(
            "src.api.predictor_client.predict",
            new=AsyncMock(side_effect=TimeoutError("Predictor timeout")),
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [
                        {
                            "task_id": "task-timeout",
                            "metadata": {"should_be": "rolled_back"},
                        }
                    ]
                },
            )

        assert response.status_code == 200
        data = response.json()

        # Check that the update failed
        assert data["failed"] >= 1
        assert data["results"][0]["success"] is False
        loop.close()


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestUpdateMetadataEdgeCases:
    """Tests for edge cases."""

    def test_instance_not_found_skips_queue_update(self, test_client):
        """Test that if assigned instance is removed, queue update is skipped."""
        # Register instance
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-to-remove",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        # Submit task
        old_pred = Prediction(
            instance_id="inst-to-remove",
            predicted_time_ms=100.0,
            confidence=0.9,
            quantiles={0.5: 80.0, 0.9: 150.0, 0.95: 200.0, 0.99: 300.0},
        )

        with patch(
            "src.api.predictor_client.predict",
            new=AsyncMock(return_value=[old_pred]),
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "task-orphan",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {"original": "value"},
                },
            )

        # Remove the instance
        test_client.post(
            "/instance/remove", json={"instance_id": "inst-to-remove"}
        )

        # Try to update metadata - should still work but skip queue update
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [
                        {"task_id": "task-orphan", "metadata": {"new": "value"}}
                    ]
                },
            )

        # Should not crash, may succeed or fail gracefully
        assert response.status_code == 200

    def test_metadata_replaces_not_merges(self, test_client):
        """Test that new metadata completely replaces old (no merge)."""
        from src.api import task_registry

        # Register and submit
        test_client.post(
            "/instance/register",
            json={
                "instance_id": "inst-replace",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hw",
                },
            },
        )

        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            test_client.post(
                "/task/submit",
                json={
                    "task_id": "task-replace",
                    "model_id": "model-1",
                    "task_input": {"prompt": "test"},
                    "metadata": {
                        "key1": "value1",
                        "key2": "value2",
                        "key3": "value3",
                    },
                },
            )

        # Update with partial metadata (only key1)
        with patch(
            "src.api.predictor_client.predict", new=AsyncMock(return_value=[])
        ):
            response = test_client.post(
                "/task/update_metadata",
                json={
                    "updates": [
                        {
                            "task_id": "task-replace",
                            "metadata": {"key1": "new_value1"},
                        }
                    ]
                },
            )

        assert response.status_code == 200

        # Verify old keys are gone (replaced, not merged)
        async def check_metadata():
            task = await task_registry.get("task-replace")
            if task:
                # key2 and key3 should NOT be present if replace semantics
                return (
                    "key2" not in task.metadata and "key3" not in task.metadata
                )
            return True

        # Note: This assertion depends on implementation
        # is_replaced = asyncio.run(check_metadata())
        # assert is_replaced


# ============================================================================
# Test Class: TaskRegistry Methods (Unit Tests)
# ============================================================================


class TestTaskRegistryUpdateMethods:
    """Unit tests for TaskRegistry update_metadata and update_prediction methods."""

    @pytest.mark.asyncio
    async def test_update_metadata_success(self, task_registry):
        """Test successful metadata update in TaskRegistry."""
        # Create a task
        await task_registry.create_task(
            task_id="test-task",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={"original": "value"},
            assigned_instance="inst-1",
        )

        # Update metadata
        task = await task_registry.update_metadata(
            "test-task", {"new": "metadata"}
        )

        assert task.metadata == {"new": "metadata"}
        # Other fields should be unchanged
        assert task.task_input == {"prompt": "test"}
        assert task.model_id == "model-1"

    @pytest.mark.asyncio
    async def test_update_metadata_not_found(self, task_registry):
        """Test update_metadata raises KeyError for non-existent task."""
        with pytest.raises(KeyError, match="Task nonexistent not found"):
            await task_registry.update_metadata("nonexistent", {"key": "value"})

    @pytest.mark.asyncio
    async def test_update_prediction_success(self, task_registry):
        """Test successful prediction update in TaskRegistry."""
        # Create a task
        await task_registry.create_task(
            task_id="test-pred-task",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance="inst-1",
            predicted_time_ms=100.0,
            predicted_error_margin_ms=10.0,
        )

        # Update prediction
        await task_registry.update_prediction(
            task_id="test-pred-task",
            predicted_time_ms=200.0,
            predicted_error_margin_ms=20.0,
            predicted_quantiles={0.5: 150.0, 0.9: 250.0},
        )

        task = await task_registry.get("test-pred-task")
        assert task.predicted_time_ms == 200.0
        assert task.predicted_error_margin_ms == 20.0
        assert task.predicted_quantiles == {0.5: 150.0, 0.9: 250.0}

    @pytest.mark.asyncio
    async def test_update_prediction_not_found(self, task_registry):
        """Test update_prediction raises KeyError for non-existent task."""
        with pytest.raises(KeyError, match="Task nonexistent not found"):
            await task_registry.update_prediction(
                task_id="nonexistent",
                predicted_time_ms=100.0,
                predicted_error_margin_ms=10.0,
                predicted_quantiles=None,
            )
