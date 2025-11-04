"""
Tests for queue type mismatch issue when switching strategies.

This test reproduces the bug where switching from probabilistic to min_time
strategy causes queue type mismatch errors during update_queue operations.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.scheduler import (
    MinimumExpectedTimeStrategy,
    ProbabilisticSchedulingStrategy,
    get_strategy,
)
from src.predictor_client import Prediction, PredictorClient
from src.instance_registry import InstanceRegistry
from src.model import (
    Instance,
    InstanceStatus,
    InstanceQueueProbabilistic,
    InstanceQueueExpectError,
)


class TestQueueTypeMismatch:
    """Test queue type mismatch when switching strategies."""

    @pytest.mark.asyncio
    async def test_min_time_strategy_with_wrong_queue_type(self):
        """
        Test that MinimumExpectedTimeStrategy handles queue type mismatch gracefully.

        This reproduces the bug where the queue is initialized as InstanceQueueProbabilistic
        but min_time strategy expects InstanceQueueExpectError. The fix should auto-correct it.
        """
        # Setup: Create predictor client and registry
        predictor_client = AsyncMock(spec=PredictorClient)
        instance_registry = InstanceRegistry(queue_info_type="probabilistic")  # Wrong type!

        # Create an instance
        instance = Instance(
            instance_id="test-instance-1",
            model_id="test-model",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "python",
                "software_version": "3.9",
                "hardware_name": "cpu",
            },
            status=InstanceStatus.ACTIVE,
        )

        # Register instance (will create InstanceQueueProbabilistic)
        await instance_registry.register(instance)

        # Create min_time strategy
        strategy = MinimumExpectedTimeStrategy(predictor_client, instance_registry)

        # Create a prediction
        prediction = Prediction(
            instance_id="test-instance-1",
            predicted_time_ms=100.0,
            error_margin_ms=10.0,
        )

        # Try to update queue - this should trigger the mismatch warning and skip update
        with patch("src.scheduler.logger") as mock_logger:
            await strategy.update_queue("test-instance-1", prediction)

            # Verify that warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Queue info type mismatch" in warning_msg
            assert "expected InstanceQueueExpectError" in warning_msg
            assert "got InstanceQueueProbabilistic" in warning_msg
            assert "Skipping update" in warning_msg

        # After the update attempt, the queue should remain unchanged (still wrong type)
        updated_queue = await instance_registry.get_queue_info("test-instance-1")
        assert isinstance(updated_queue, InstanceQueueProbabilistic)
        # Values should be unchanged from initial state
        assert updated_queue.quantiles == [0.5, 0.9, 0.95, 0.99]
        assert updated_queue.values == [0.0, 0.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_probabilistic_strategy_with_wrong_queue_type(self):
        """
        Test that ProbabilisticSchedulingStrategy handles queue type mismatch gracefully.

        This tests the opposite case: queue is InstanceQueueExpectError but
        probabilistic strategy expects InstanceQueueProbabilistic. The fix should auto-correct it.
        """
        # Setup: Create predictor client and registry
        predictor_client = AsyncMock(spec=PredictorClient)
        instance_registry = InstanceRegistry(queue_info_type="expect_error")  # Wrong type!

        # Create an instance
        instance = Instance(
            instance_id="test-instance-2",
            model_id="test-model",
            endpoint="http://localhost:8002",
            platform_info={
                "software_name": "python",
                "software_version": "3.9",
                "hardware_name": "cpu",
            },
            status=InstanceStatus.ACTIVE,
        )

        # Register instance (will create InstanceQueueExpectError)
        await instance_registry.register(instance)

        # Create probabilistic strategy
        strategy = ProbabilisticSchedulingStrategy(predictor_client, instance_registry)

        # Create a prediction with quantiles
        prediction = Prediction(
            instance_id="test-instance-2",
            predicted_time_ms=100.0,
            quantiles={0.5: 100.0, 0.9: 150.0, 0.95: 180.0, 0.99: 200.0},
        )

        # Try to update queue - this should trigger the mismatch warning and skip update
        with patch("src.scheduler.logger") as mock_logger:
            await strategy.update_queue("test-instance-2", prediction)

            # Verify that warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Queue info type mismatch" in warning_msg
            assert "expected InstanceQueueProbabilistic" in warning_msg
            assert "got InstanceQueueExpectError" in warning_msg
            assert "Skipping update" in warning_msg

        # After the update attempt, the queue should remain unchanged (still wrong type)
        updated_queue = await instance_registry.get_queue_info("test-instance-2")
        assert isinstance(updated_queue, InstanceQueueExpectError)
        # Values should be unchanged from initial state
        assert updated_queue.expected_time_ms == 0.0
        assert updated_queue.error_margin_ms == 0.0

    @pytest.mark.asyncio
    async def test_strategy_switch_updates_queue_types_correctly(self):
        """
        Test that switching strategies properly updates queue types for all instances.

        This tests the fix: ensure queue types are consistent after strategy switch.
        """
        # Setup: Start with probabilistic strategy
        predictor_client = AsyncMock(spec=PredictorClient)
        instance_registry = InstanceRegistry(queue_info_type="probabilistic")

        # Create and register multiple instances
        instances = []
        for i in range(3):
            instance = Instance(
                instance_id=f"instance-{i}",
                model_id="test-model",
                endpoint=f"http://localhost:800{i}",
                platform_info={
                    "software_name": "python",
                    "software_version": "3.9",
                    "hardware_name": "cpu",
                },
                status=InstanceStatus.ACTIVE,
            )
            instances.append(instance)
            await instance_registry.register(instance)

        # Verify initial queue types are InstanceQueueProbabilistic
        for instance in instances:
            queue_info = await instance_registry.get_queue_info(instance.instance_id)
            assert isinstance(queue_info, InstanceQueueProbabilistic)

        # Switch to min_time strategy (simulate reinitialize_instance_queues)
        # First update the registry's queue type
        instance_registry._queue_info_type = "expect_error"

        # Then update all existing queue infos
        for instance in instances:
            new_queue = InstanceQueueExpectError(
                instance_id=instance.instance_id,
                expected_time_ms=0.0,
                error_margin_ms=0.0,
            )
            await instance_registry.update_queue_info(instance.instance_id, new_queue)

        # Verify queue types are now InstanceQueueExpectError
        for instance in instances:
            queue_info = await instance_registry.get_queue_info(instance.instance_id)
            assert isinstance(queue_info, InstanceQueueExpectError)

        # Create min_time strategy
        strategy = MinimumExpectedTimeStrategy(predictor_client, instance_registry)

        # Create predictions
        predictions = [
            Prediction(
                instance_id=f"instance-{i}",
                predicted_time_ms=100.0 * (i + 1),
                error_margin_ms=10.0 * (i + 1),
            )
            for i in range(3)
        ]

        # Update queues - should work without warnings
        with patch("src.scheduler.logger") as mock_logger:
            for prediction in predictions:
                await strategy.update_queue(prediction.instance_id, prediction)

            # Verify no warnings were logged
            mock_logger.warning.assert_not_called()

        # Verify queues were updated correctly
        for i, instance in enumerate(instances):
            queue_info = await instance_registry.get_queue_info(instance.instance_id)
            assert isinstance(queue_info, InstanceQueueExpectError)
            assert queue_info.expected_time_ms == 100.0 * (i + 1)
            # Error margin should be updated based on the error accumulation formula
            import math
            expected_error = math.sqrt(0.0**2 + (10.0 * (i + 1))**2)
            assert queue_info.error_margin_ms == pytest.approx(expected_error)

    @pytest.mark.asyncio
    async def test_empty_queue_initialization(self):
        """
        Test that when no queue exists, update_queue initializes with correct type.
        """
        # Setup: Create predictor client and registry
        predictor_client = AsyncMock(spec=PredictorClient)
        instance_registry = InstanceRegistry(queue_info_type="expect_error")

        # Create an instance but don't set any queue info
        instance = Instance(
            instance_id="test-instance-no-queue",
            model_id="test-model",
            endpoint="http://localhost:8003",
            platform_info={
                "software_name": "python",
                "software_version": "3.9",
                "hardware_name": "cpu",
            },
            status=InstanceStatus.ACTIVE,
        )
        await instance_registry.register(instance)

        # Clear the queue info to simulate no queue
        instance_registry._queue_info.pop(instance.instance_id, None)

        # Create min_time strategy
        strategy = MinimumExpectedTimeStrategy(predictor_client, instance_registry)

        # Create a prediction
        prediction = Prediction(
            instance_id="test-instance-no-queue",
            predicted_time_ms=150.0,
            error_margin_ms=15.0,
        )

        # Update queue - should initialize with correct type
        await strategy.update_queue("test-instance-no-queue", prediction)

        # Verify queue was created and updated
        updated_queue = await instance_registry.get_queue_info("test-instance-no-queue")
        assert isinstance(updated_queue, InstanceQueueExpectError)
        assert updated_queue.expected_time_ms == 150.0
        assert updated_queue.error_margin_ms == 15.0

    @pytest.mark.asyncio
    async def test_new_instance_after_strategy_switch_has_correct_type(self):
        """
        Test that new instances registered after strategy switch have correct queue type.

        This tests that the fix properly handles new instances after switching strategies.
        """
        # Setup: Start with probabilistic, switch to min_time
        predictor_client = AsyncMock(spec=PredictorClient)
        instance_registry = InstanceRegistry(queue_info_type="probabilistic")

        # Register initial instance
        instance1 = Instance(
            instance_id="instance-before-switch",
            model_id="test-model",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "python",
                "software_version": "3.9",
                "hardware_name": "cpu",
            },
            status=InstanceStatus.ACTIVE,
        )
        await instance_registry.register(instance1)

        # Verify initial type
        queue1 = await instance_registry.get_queue_info(instance1.instance_id)
        assert isinstance(queue1, InstanceQueueProbabilistic)

        # Simulate strategy switch: Update registry type FIRST
        instance_registry._queue_info_type = "expect_error"

        # Update existing instance queue
        new_queue1 = InstanceQueueExpectError(
            instance_id=instance1.instance_id,
            expected_time_ms=0.0,
            error_margin_ms=0.0,
        )
        await instance_registry.update_queue_info(instance1.instance_id, new_queue1)

        # Register NEW instance after switch
        instance2 = Instance(
            instance_id="instance-after-switch",
            model_id="test-model",
            endpoint="http://localhost:8002",
            platform_info={
                "software_name": "python",
                "software_version": "3.9",
                "hardware_name": "cpu",
            },
            status=InstanceStatus.ACTIVE,
        )
        await instance_registry.register(instance2)

        # Verify BOTH instances have correct queue type
        queue1_after = await instance_registry.get_queue_info(instance1.instance_id)
        assert isinstance(queue1_after, InstanceQueueExpectError)

        queue2 = await instance_registry.get_queue_info(instance2.instance_id)
        assert isinstance(queue2, InstanceQueueExpectError)

        # Test that min_time strategy works with both instances
        strategy = MinimumExpectedTimeStrategy(predictor_client, instance_registry)

        predictions = [
            Prediction(instance_id="instance-before-switch", predicted_time_ms=100.0, error_margin_ms=10.0),
            Prediction(instance_id="instance-after-switch", predicted_time_ms=200.0, error_margin_ms=20.0),
        ]

        # Update queues - should work without warnings
        with patch("src.scheduler.logger") as mock_logger:
            for pred in predictions:
                await strategy.update_queue(pred.instance_id, pred)

            # Verify no warnings
            mock_logger.warning.assert_not_called()