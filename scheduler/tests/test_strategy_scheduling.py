"""Unit tests for scheduling strategies with custom quantiles support.

This module tests:
1. Task scheduling for round-robin, min_time, and probabilistic strategies
2. Queue information updates
3. Predictor service requests
4. Custom quantiles handling for probabilistic strategy
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.algorithms import (
    MinimumExpectedTimeStrategy,
    ProbabilisticSchedulingStrategy,
    RoundRobinStrategy,
    get_strategy,
)
from src.clients.models import Prediction
from src.clients.predictor_library_client import PredictorClient
from src.model import (
    Instance,
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
    InstanceStatus,
)
from src.registry.instance_registry import InstanceRegistry


class TestSchedulingStrategies:
    """Test all scheduling strategies for task scheduling and queue updates."""

    @pytest.fixture
    def mock_predictor_client(self):
        """Create a mock predictor client."""
        client = AsyncMock(spec=PredictorClient)
        return client

    @pytest.fixture
    def instance_registry_probabilistic(self):
        """Create an instance registry configured for probabilistic strategy."""
        return InstanceRegistry(queue_info_type="probabilistic")

    @pytest.fixture
    def instance_registry_expect_error(self):
        """Create an instance registry configured for min_time strategy."""
        return InstanceRegistry(queue_info_type="expect_error")

    @pytest.fixture
    async def test_instances(self):
        """Create test instances."""
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
        return instances

    @pytest.mark.asyncio
    async def test_round_robin_strategy_scheduling(
        self,
        mock_predictor_client,
        instance_registry_probabilistic,
        test_instances,
    ):
        """Test round-robin strategy schedules tasks in order and makes predictor requests."""
        # Register instances
        for instance in test_instances:
            await instance_registry_probabilistic.register(instance)

        # Create round-robin strategy
        strategy = RoundRobinStrategy(
            mock_predictor_client, instance_registry_probabilistic
        )

        # Mock predictor responses
        predictions = [
            Prediction(
                instance_id=f"instance-{i}",
                predicted_time_ms=100.0 * (i + 1),
            )
            for i in range(3)
        ]
        mock_predictor_client.predict.return_value = predictions

        # Test scheduling multiple tasks
        metadata = {"task_type": "test", "complexity": 1}

        # Schedule first task - should select instance-0
        result1 = await strategy.schedule_task("test-model", metadata, test_instances)
        assert result1.selected_instance_id == "instance-0"

        # Schedule second task - should select instance-1
        result2 = await strategy.schedule_task("test-model", metadata, test_instances)
        assert result2.selected_instance_id == "instance-1"

        # Schedule third task - should select instance-2
        result3 = await strategy.schedule_task("test-model", metadata, test_instances)
        assert result3.selected_instance_id == "instance-2"

        # Schedule fourth task - should wrap around to instance-0
        result4 = await strategy.schedule_task("test-model", metadata, test_instances)
        assert result4.selected_instance_id == "instance-0"

        # Verify predictor was called with correct parameters
        assert mock_predictor_client.predict.call_count == 4
        for call_args in mock_predictor_client.predict.call_args_list:
            kwargs = call_args.kwargs
            assert kwargs["model_id"] == "test-model"
            assert kwargs["metadata"] == metadata
            assert kwargs["prediction_type"] == "expect_error"  # Default type

        # Verify queue updates (round-robin doesn't update queues)
        for instance in test_instances:
            queue = await instance_registry_probabilistic.get_queue_info(
                instance.instance_id
            )
            assert isinstance(queue, InstanceQueueProbabilistic)
            # Queue values should remain at initial state (not updated by round-robin)
            assert queue.values == [0.0, 0.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_min_time_strategy_scheduling(
        self,
        mock_predictor_client,
        instance_registry_expect_error,
        test_instances,
    ):
        """Test min_time strategy selects instance with minimum expected time and updates queues."""
        # Register instances
        for instance in test_instances:
            await instance_registry_expect_error.register(instance)

        # Create min_time strategy
        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry_expect_error
        )

        # Set up initial queue states with different expected times
        for i, instance in enumerate(test_instances):
            initial_queue = InstanceQueueExpectError(
                instance_id=instance.instance_id,
                expected_time_ms=50.0 * i,  # 0, 50, 100
                error_margin_ms=5.0 * i,  # 0, 5, 10
            )
            await instance_registry_expect_error.update_queue_info(
                instance.instance_id, initial_queue
            )

        # Mock predictor responses - instance-1 has lowest new task time
        predictions = [
            Prediction(
                instance_id="instance-0",
                predicted_time_ms=200.0,
                error_margin_ms=20.0,
            ),
            Prediction(
                instance_id="instance-1",
                predicted_time_ms=50.0,  # Lowest task time
                error_margin_ms=5.0,
            ),
            Prediction(
                instance_id="instance-2",
                predicted_time_ms=150.0,
                error_margin_ms=15.0,
            ),
        ]
        mock_predictor_client.predict.return_value = predictions

        # Schedule task
        metadata = {"task_type": "compute", "size": "medium"}
        result = await strategy.schedule_task("test-model", metadata, test_instances)

        # Should select instance-1 (50 + 5 + 50 = 105, lowest total)
        assert result.selected_instance_id == "instance-1"
        assert result.selected_prediction.predicted_time_ms == 50.0

        # Verify predictor was called correctly
        mock_predictor_client.predict.assert_called_once_with(
            model_id="test-model",
            metadata=metadata,
            instances=test_instances,
            prediction_type="expect_error",
        )

        # Verify queue was updated for selected instance
        import math

        updated_queue = await instance_registry_expect_error.get_queue_info(
            "instance-1"
        )
        assert isinstance(updated_queue, InstanceQueueExpectError)
        assert updated_queue.expected_time_ms == 100.0  # 50 + 50
        expected_error = math.sqrt(5.0**2 + 5.0**2)
        assert updated_queue.error_margin_ms == pytest.approx(expected_error)

        # Other queues should remain unchanged
        queue0 = await instance_registry_expect_error.get_queue_info("instance-0")
        assert queue0.expected_time_ms == 0.0
        queue2 = await instance_registry_expect_error.get_queue_info("instance-2")
        assert queue2.expected_time_ms == 100.0

    @pytest.mark.asyncio
    async def test_probabilistic_strategy_with_default_quantiles(
        self,
        mock_predictor_client,
        instance_registry_probabilistic,
        test_instances,
    ):
        """Test probabilistic strategy with default quantiles."""
        # Register instances
        for instance in test_instances:
            await instance_registry_probabilistic.register(instance)

        # Create probabilistic strategy
        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry_probabilistic
        )

        # Mock predictor responses with quantile predictions
        predictions = [
            Prediction(
                instance_id=f"instance-{i}",
                predicted_time_ms=100.0 * (i + 1),
                quantiles={
                    0.5: 80.0 * (i + 1),
                    0.9: 120.0 * (i + 1),
                    0.95: 140.0 * (i + 1),
                    0.99: 160.0 * (i + 1),
                },
            )
            for i in range(3)
        ]
        mock_predictor_client.predict.return_value = predictions

        # Schedule task
        metadata = {"task_type": "ml_inference", "model_size": "large"}

        # Mock random sampling to make test deterministic
        with patch("numpy.random.random") as mock_random:
            mock_random.return_value = 0.5  # Sample at median
            result = await strategy.schedule_task(
                "test-model", metadata, test_instances
            )

        # Verify predictor was called with quantile prediction type and default quantiles
        mock_predictor_client.predict.assert_called_once_with(
            model_id="test-model",
            metadata=metadata,
            instances=test_instances,
            prediction_type="quantile",
            quantiles=[0.5, 0.9, 0.95, 0.99],
        )

        # Verify selected instance (depends on sampling)
        assert result.selected_instance_id in [
            "instance-0",
            "instance-1",
            "instance-2",
        ]

        # Verify queue was updated using Monte Carlo sampling
        selected_queue = await instance_registry_probabilistic.get_queue_info(
            result.selected_instance_id
        )
        assert isinstance(selected_queue, InstanceQueueProbabilistic)
        assert selected_queue.quantiles == [0.5, 0.9, 0.95, 0.99]
        # Values should be updated (non-zero after Monte Carlo sampling)
        assert any(v > 0 for v in selected_queue.values)

    @pytest.mark.asyncio
    async def test_probabilistic_strategy_with_custom_quantiles(
        self, mock_predictor_client
    ):
        """Test probabilistic strategy handles custom quantiles correctly."""
        # Create instance registry with custom quantiles
        instance_registry = InstanceRegistry(queue_info_type="probabilistic")
        custom_quantiles = [0.25, 0.5, 0.75, 0.9, 0.95]
        instance_registry._quantiles = custom_quantiles

        # Create test instances
        instances = []
        for i in range(2):
            instance = Instance(
                instance_id=f"instance-{i}",
                model_id="test-model",
                endpoint=f"http://localhost:900{i}",
                platform_info={
                    "software_name": "python",
                    "software_version": "3.9",
                    "hardware_name": "gpu",
                },
                status=InstanceStatus.ACTIVE,
            )
            instances.append(instance)
            await instance_registry.register(instance)

        # Verify instances were registered with custom quantiles
        for instance in instances:
            queue = await instance_registry.get_queue_info(instance.instance_id)
            assert isinstance(queue, InstanceQueueProbabilistic)
            assert queue.quantiles == custom_quantiles
            assert len(queue.values) == len(custom_quantiles)

        # Create strategy
        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        # Mock predictor responses with matching quantiles
        predictions = [
            Prediction(
                instance_id=f"instance-{i}",
                predicted_time_ms=100.0 * (i + 1),
                quantiles={
                    0.25: 50.0 * (i + 1),
                    0.5: 100.0 * (i + 1),
                    0.75: 150.0 * (i + 1),
                    0.9: 200.0 * (i + 1),
                    0.95: 250.0 * (i + 1),
                },
            )
            for i in range(2)
        ]
        mock_predictor_client.predict.return_value = predictions

        # Schedule task
        metadata = {"task_type": "custom", "priority": "high"}
        result = await strategy.schedule_task("test-model", metadata, instances)

        # Verify task was scheduled
        assert result.selected_instance_id in ["instance-0", "instance-1"]

        # Verify queue was updated with custom quantiles preserved
        updated_queue = await instance_registry.get_queue_info(
            result.selected_instance_id
        )
        assert isinstance(updated_queue, InstanceQueueProbabilistic)
        assert updated_queue.quantiles == custom_quantiles

    @pytest.mark.asyncio
    async def test_strategy_handles_no_predictions(
        self,
        mock_predictor_client,
        instance_registry_probabilistic,
        test_instances,
    ):
        """Test strategies handle case when predictor returns no predictions."""
        # Register instances
        for instance in test_instances:
            await instance_registry_probabilistic.register(instance)

        # Mock predictor to return empty list
        mock_predictor_client.predict.return_value = []

        # Test each strategy
        strategies = [
            RoundRobinStrategy(mock_predictor_client, instance_registry_probabilistic),
            MinimumExpectedTimeStrategy(
                mock_predictor_client, instance_registry_probabilistic
            ),
            ProbabilisticSchedulingStrategy(
                mock_predictor_client, instance_registry_probabilistic
            ),
        ]

        metadata = {"task_type": "test"}

        for strategy in strategies:
            result = await strategy.schedule_task(
                "test-model", metadata, test_instances
            )
            # Should fallback to first instance when no predictions
            assert result.selected_instance_id == "instance-0"
            assert result.selected_prediction is None

    @pytest.mark.asyncio
    async def test_strategy_handles_predictor_errors(
        self,
        mock_predictor_client,
        instance_registry_probabilistic,
        test_instances,
    ):
        """Test strategies handle predictor service errors gracefully.

        The library predictor client raises standard Python exceptions
        which propagate directly through get_predictions().
        """
        # Register instances
        for instance in test_instances:
            await instance_registry_probabilistic.register(instance)

        # Test different error scenarios (library client exceptions)
        error_scenarios = [
            (
                ValueError("No trained model available for this platform"),
                ValueError,
                "No trained model available",
            ),
            (
                TimeoutError("Predictor service timeout"),
                TimeoutError,
                "Predictor service timeout",
            ),
            (
                ConnectionError("Predictor service unavailable"),
                ConnectionError,
                "Predictor service unavailable",
            ),
        ]

        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry_probabilistic
        )
        metadata = {"task_type": "test"}

        for mock_error, expected_exception, expected_message in error_scenarios:
            mock_predictor_client.predict.side_effect = mock_error

            with pytest.raises(expected_exception) as exc_info:
                await strategy.schedule_task("test-model", metadata, test_instances)

            assert expected_message in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_min_time_strategy_queue_accumulation(
        self, mock_predictor_client, instance_registry_expect_error
    ):
        """Test min_time strategy correctly accumulates queue times and errors."""
        # Create and register single instance
        instance = Instance(
            instance_id="instance-0",
            model_id="test-model",
            endpoint="http://localhost:8000",
            platform_info={
                "software_name": "python",
                "software_version": "3.9",
                "hardware_name": "cpu",
            },
            status=InstanceStatus.ACTIVE,
        )
        await instance_registry_expect_error.register(instance)

        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry_expect_error
        )

        # Schedule multiple tasks to test accumulation
        import math

        accumulated_time = 0.0
        accumulated_error_sq = 0.0

        for i in range(3):
            # Mock predictor response
            prediction = Prediction(
                instance_id="instance-0",
                predicted_time_ms=100.0,
                error_margin_ms=10.0,
            )
            mock_predictor_client.predict.return_value = [prediction]

            # Schedule task
            metadata = {"task_id": f"task-{i}"}
            _result = await strategy.schedule_task("test-model", metadata, [instance])

            # Update expected values
            accumulated_time += 100.0
            accumulated_error_sq += 10.0**2
            expected_error = math.sqrt(accumulated_error_sq)

            # Verify queue accumulation
            queue = await instance_registry_expect_error.get_queue_info("instance-0")
            assert isinstance(queue, InstanceQueueExpectError)
            assert queue.expected_time_ms == pytest.approx(accumulated_time)
            assert queue.error_margin_ms == pytest.approx(expected_error)

    @pytest.mark.asyncio
    async def test_probabilistic_strategy_monte_carlo_update(
        self, mock_predictor_client, instance_registry_probabilistic
    ):
        """Test probabilistic strategy uses Monte Carlo sampling for queue updates."""
        # Create and register instance
        instance = Instance(
            instance_id="instance-0",
            model_id="test-model",
            endpoint="http://localhost:8000",
            platform_info={
                "software_name": "python",
                "software_version": "3.9",
                "hardware_name": "gpu",
            },
            status=InstanceStatus.ACTIVE,
        )
        await instance_registry_probabilistic.register(instance)

        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry_probabilistic
        )

        # Set initial queue state
        initial_queue = InstanceQueueProbabilistic(
            instance_id="instance-0",
            quantiles=[0.5, 0.9, 0.95, 0.99],
            values=[100.0, 150.0, 180.0, 200.0],
        )
        await instance_registry_probabilistic.update_queue_info(
            "instance-0", initial_queue
        )

        # Mock predictor response with quantiles
        prediction = Prediction(
            instance_id="instance-0",
            predicted_time_ms=50.0,
            quantiles={
                0.5: 40.0,
                0.9: 60.0,
                0.95: 70.0,
                0.99: 80.0,
            },
        )
        mock_predictor_client.predict.return_value = [prediction]

        # Schedule task
        metadata = {"task_type": "inference"}

        # Use a fixed seed for reproducible Monte Carlo sampling
        import numpy as np

        np.random.seed(42)

        _result = await strategy.schedule_task("test-model", metadata, [instance])

        # Verify queue was updated
        updated_queue = await instance_registry_probabilistic.get_queue_info(
            "instance-0"
        )
        assert isinstance(updated_queue, InstanceQueueProbabilistic)
        assert updated_queue.quantiles == [0.5, 0.9, 0.95, 0.99]

        # Values should be updated and generally larger than initial values
        # (since we're adding task time to queue time)
        assert all(v > 0 for v in updated_queue.values)
        # Approximate check - values should increase from initial
        assert (
            updated_queue.values[0] > initial_queue.values[0] * 0.8
        )  # Some tolerance for Monte Carlo


class TestStrategyFactory:
    """Test the get_strategy factory function."""

    @pytest.fixture
    def mock_predictor_client(self):
        """Create a mock predictor client."""
        return AsyncMock(spec=PredictorClient)

    @pytest.fixture
    def instance_registry(self):
        """Create an instance registry."""
        return InstanceRegistry()

    def test_get_strategy_min_time(self, mock_predictor_client, instance_registry):
        """Test getting min_time strategy."""
        strategy = get_strategy("min_time", mock_predictor_client, instance_registry)
        assert isinstance(strategy, MinimumExpectedTimeStrategy)

    def test_get_strategy_probabilistic(self, mock_predictor_client, instance_registry):
        """Test getting probabilistic strategy."""
        strategy = get_strategy(
            "probabilistic", mock_predictor_client, instance_registry
        )
        assert isinstance(strategy, ProbabilisticSchedulingStrategy)

    def test_get_strategy_round_robin(self, mock_predictor_client, instance_registry):
        """Test getting round-robin strategy."""
        strategy = get_strategy("round_robin", mock_predictor_client, instance_registry)
        assert isinstance(strategy, RoundRobinStrategy)

    def test_get_strategy_unknown_defaults_to_probabilistic(
        self, mock_predictor_client, instance_registry
    ):
        """Test unknown strategy name defaults to probabilistic."""
        strategy = get_strategy(
            "unknown_strategy", mock_predictor_client, instance_registry
        )
        assert isinstance(strategy, ProbabilisticSchedulingStrategy)
