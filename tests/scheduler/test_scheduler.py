"""Unit tests for scheduling strategies.

Tests all scheduling strategies and the factory function.
"""

import pytest

from swarmpilot.scheduler.algorithms import (
    AdaptiveBootstrapStrategy,
    MinimumExpectedTimeStrategy,
    ProbabilisticSchedulingStrategy,
    RoundRobinStrategy,
    get_strategy,
)
from swarmpilot.scheduler.clients.models import Prediction
from swarmpilot.scheduler.models import (
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)

# ============================================================================
# MinimumExpectedTimeStrategy Tests
# ============================================================================


class TestMinimumExpectedTimeStrategy:
    """Tests for MinimumExpectedTimeStrategy."""

    async def test_select_instance_with_minimum_time(
        self, mock_predictor_client, instance_registry
    ):
        """Test selecting instance with minimum predicted time (no queue)."""
        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=200.0,
                error_margin_ms=10.0,
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=100.0,
                error_margin_ms=5.0,
            ),
            Prediction(
                instance_id="inst-3",
                predicted_time_ms=150.0,
                error_margin_ms=8.0,
            ),
        ]

        queue_info = {
            "inst-1": InstanceQueueExpectError(
                instance_id="inst-1", expected_time_ms=0.0, error_margin_ms=0.0
            ),
            "inst-2": InstanceQueueExpectError(
                instance_id="inst-2", expected_time_ms=0.0, error_margin_ms=0.0
            ),
            "inst-3": InstanceQueueExpectError(
                instance_id="inst-3", expected_time_ms=0.0, error_margin_ms=0.0
            ),
        }

        selected = strategy.select_instance(predictions, queue_info)
        assert selected == "inst-2"

    async def test_select_instance_with_queue_info(
        self, mock_predictor_client, instance_registry
    ):
        """Test selecting instance considering queue state."""
        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                error_margin_ms=5.0,
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=100.0,
                error_margin_ms=5.0,
            ),
            Prediction(
                instance_id="inst-3",
                predicted_time_ms=100.0,
                error_margin_ms=5.0,
            ),
        ]

        # inst-2 has a longer queue
        queue_info = {
            "inst-1": InstanceQueueExpectError(
                instance_id="inst-1",
                expected_time_ms=50.0,
                error_margin_ms=10.0,
            ),
            "inst-2": InstanceQueueExpectError(
                instance_id="inst-2",
                expected_time_ms=200.0,
                error_margin_ms=20.0,
            ),
            "inst-3": InstanceQueueExpectError(
                instance_id="inst-3", expected_time_ms=30.0, error_margin_ms=5.0
            ),
        }

        selected = strategy.select_instance(predictions, queue_info)
        # inst-3 has total time 30 + 5 + 100 = 135
        # inst-1 has total time 50 + 10 + 100 = 160
        # inst-2 has total time 200 + 20 + 100 = 320
        assert selected == "inst-3"

    async def test_select_instance_empty_list(
        self, mock_predictor_client, instance_registry
    ):
        """Test selection with empty predictions list."""
        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry
        )

        selected = strategy.select_instance([], {})
        assert selected is None

    async def test_select_instance_single_prediction(
        self, mock_predictor_client, instance_registry
    ):
        """Test selection with single prediction."""
        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                error_margin_ms=5.0,
            ),
        ]

        queue_info = {
            "inst-1": InstanceQueueExpectError(
                instance_id="inst-1", expected_time_ms=0.0, error_margin_ms=0.0
            ),
        }

        selected = strategy.select_instance(predictions, queue_info)
        assert selected == "inst-1"


# ============================================================================
# ProbabilisticSchedulingStrategy Tests
# ============================================================================


class TestProbabilisticSchedulingStrategy:
    """Tests for ProbabilisticSchedulingStrategy."""

    async def test_default_target_quantile(
        self, mock_predictor_client, instance_registry
    ):
        """Test that default target quantile is 0.9."""
        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )
        assert strategy.target_quantile == 0.9

    async def test_select_based_on_quantile(
        self, mock_predictor_client, instance_registry
    ):
        """Test selection based on sampling from quantile distribution."""
        import numpy as np

        # Set random seed for reproducibility
        np.random.seed(42)

        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                quantiles={0.5: 80.0, 0.9: 150.0},
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=120.0,
                quantiles={0.5: 100.0, 0.9: 130.0},
            ),
            Prediction(
                instance_id="inst-3",
                predicted_time_ms=90.0,
                quantiles={0.5: 70.0, 0.9: 200.0},
            ),
        ]

        # Probabilistic selection based on sampling - should return valid instance
        selected = strategy.select_instance(predictions, {})
        assert selected in ["inst-1", "inst-2", "inst-3"]

    async def test_select_empty_list(
        self, mock_predictor_client, instance_registry
    ):
        """Test selection with empty predictions list."""
        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        selected = strategy.select_instance([], {})
        assert selected is None

    async def test_fallback_to_min_time(
        self, mock_predictor_client, instance_registry
    ):
        """Test fallback to minimum expected time when quantile not available."""
        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=200.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=150.0),
        ]

        # No quantiles available, should fallback to min time
        selected = strategy.select_instance(predictions, {})
        assert selected == "inst-2"

    async def test_partial_quantile_availability(
        self, mock_predictor_client, instance_registry
    ):
        """Test when only some predictions have quantiles - uses fallback to predicted_time_ms."""
        import numpy as np

        # Set seed for reproducibility
        np.random.seed(42)

        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=200.0,
                quantiles={0.5: 150.0, 0.9: 250.0},
            ),
            Prediction(
                instance_id="inst-3",
                predicted_time_ms=150.0,
                quantiles={0.5: 100.0, 0.9: 180.0},
            ),
        ]

        # Probabilistic selection - should return valid instance
        selected = strategy.select_instance(predictions, {})
        assert selected in ["inst-1", "inst-2", "inst-3"]

    async def test_quantile_not_in_dict(
        self, mock_predictor_client, instance_registry
    ):
        """Test when quantiles exist but target quantile not in them."""
        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                quantiles={0.5: 80.0, 0.9: 150.0},  # No 0.95
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=150.0,
                quantiles={0.5: 100.0, 0.9: 200.0},  # No 0.95
            ),
        ]

        # Should fallback to min time
        selected = strategy.select_instance(predictions, {})
        assert selected == "inst-1"

    async def test_select_with_equal_quantile_values(
        self, mock_predictor_client, instance_registry
    ):
        """Test selection when multiple instances have equal quantile values."""
        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                quantiles={0.9: 150.0},
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=120.0,
                quantiles={0.9: 150.0},
            ),
        ]

        # Should select first one encountered
        selected = strategy.select_instance(predictions, {})
        assert selected == "inst-1"

    async def test_select_with_non_probabilistic_queue(
        self, mock_predictor_client, instance_registry
    ):
        """Test selection when queue is not InstanceQueueProbabilistic (line 476)."""
        import numpy as np

        from swarmpilot.scheduler.models import (
            Instance,
            InstanceQueueExpectError,
        )

        # Set seed for reproducibility
        np.random.seed(42)

        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        # Register instances
        for i in [1, 2]:
            instance = Instance(
                instance_id=f"inst-{i}",
                model_id="model-1",
                endpoint=f"http://localhost:800{i}",
                platform_info={
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware",
                },
            )
            await instance_registry.register(instance)

        # Set queue info to ExpectError (not Probabilistic)
        queue_info = {
            "inst-1": InstanceQueueExpectError(
                instance_id="inst-1",
                expected_time_ms=50.0,
                error_margin_ms=10.0,
            ),
            "inst-2": InstanceQueueExpectError(
                instance_id="inst-2", expected_time_ms=30.0, error_margin_ms=5.0
            ),
        }

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                quantiles={0.5: 80.0, 0.9: 150.0},
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=120.0,
                quantiles={0.5: 100.0, 0.9: 130.0},
            ),
        ]

        # Should handle non-probabilistic queue by using zeros (line 476)
        selected = strategy.select_instance(predictions, queue_info)
        assert selected in ["inst-1", "inst-2"]


# ============================================================================
# RoundRobinStrategy Tests
# ============================================================================


class TestRoundRobinStrategy:
    """Tests for RoundRobinStrategy."""

    async def test_round_robin_cycling(
        self, mock_predictor_client, instance_registry
    ):
        """Test that strategy cycles through instances."""
        strategy = RoundRobinStrategy(mock_predictor_client, instance_registry)

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=100.0),
        ]

        # First call
        selected1 = strategy.select_instance(predictions, {})
        assert selected1 == "inst-1"

        # Second call
        selected2 = strategy.select_instance(predictions, {})
        assert selected2 == "inst-2"

        # Third call
        selected3 = strategy.select_instance(predictions, {})
        assert selected3 == "inst-3"

        # Fourth call - should wrap around
        selected4 = strategy.select_instance(predictions, {})
        assert selected4 == "inst-1"

    async def test_round_robin_empty_list(
        self, mock_predictor_client, instance_registry
    ):
        """Test selection with empty predictions list."""
        strategy = RoundRobinStrategy(mock_predictor_client, instance_registry)

        selected = strategy.select_instance([], {})
        assert selected is None

    async def test_round_robin_single_instance(
        self, mock_predictor_client, instance_registry
    ):
        """Test cycling with single instance."""
        strategy = RoundRobinStrategy(mock_predictor_client, instance_registry)

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
        ]

        # Should always return the same instance
        assert strategy.select_instance(predictions, {}) == "inst-1"
        assert strategy.select_instance(predictions, {}) == "inst-1"
        assert strategy.select_instance(predictions, {}) == "inst-1"

    async def test_round_robin_counter_persistence(
        self, mock_predictor_client, instance_registry
    ):
        """Test that counter persists across calls."""
        strategy = RoundRobinStrategy(mock_predictor_client, instance_registry)

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
        ]

        # Make several calls
        for _ in range(10):
            strategy.select_instance(predictions, {})

        # Counter should be at 10, next selection should be inst-1 (10 % 2 = 0)
        selected = strategy.select_instance(predictions, {})
        assert selected == "inst-1"

    async def test_round_robin_different_list_sizes(
        self, mock_predictor_client, instance_registry
    ):
        """Test round robin with changing prediction list size."""
        strategy = RoundRobinStrategy(mock_predictor_client, instance_registry)

        # Start with 3 instances
        predictions_3 = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=100.0),
        ]

        strategy.select_instance(
            predictions_3, {}
        )  # counter = 0, select inst-1
        strategy.select_instance(
            predictions_3, {}
        )  # counter = 1, select inst-2

        # Now use 2 instances
        predictions_2 = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
        ]

        # counter = 2, 2 % 2 = 0, should select inst-1
        selected = strategy.select_instance(predictions_2, {})
        assert selected == "inst-1"


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestGetStrategy:
    """Tests for get_strategy factory function."""

    async def test_get_min_time_strategy(
        self, mock_predictor_client, instance_registry
    ):
        """Test getting MinimumExpectedTimeStrategy."""
        strategy = get_strategy(
            "min_time", mock_predictor_client, instance_registry
        )
        assert isinstance(strategy, MinimumExpectedTimeStrategy)

    async def test_get_probabilistic_strategy(
        self, mock_predictor_client, instance_registry
    ):
        """Test getting ProbabilisticSchedulingStrategy."""
        strategy = get_strategy(
            "probabilistic", mock_predictor_client, instance_registry
        )
        assert isinstance(strategy, ProbabilisticSchedulingStrategy)

    async def test_get_round_robin_strategy(
        self, mock_predictor_client, instance_registry
    ):
        """Test getting RoundRobinStrategy."""
        strategy = get_strategy(
            "round_robin", mock_predictor_client, instance_registry
        )
        assert isinstance(strategy, RoundRobinStrategy)

    async def test_unknown_strategy_defaults_to_adaptive_bootstrap(
        self, mock_predictor_client, instance_registry
    ):
        """Test that unknown strategy name defaults to adaptive bootstrap."""
        strategy = get_strategy(
            "unknown_strategy", mock_predictor_client, instance_registry
        )
        assert isinstance(strategy, AdaptiveBootstrapStrategy)

    async def test_case_sensitivity(
        self, mock_predictor_client, instance_registry
    ):
        """Test that strategy names are case-sensitive."""
        # "MIN_TIME" should not match "min_time"
        strategy = get_strategy(
            "MIN_TIME", mock_predictor_client, instance_registry
        )
        # Should default to adaptive bootstrap
        assert isinstance(strategy, AdaptiveBootstrapStrategy)


# ============================================================================
# Additional Tests for Coverage
# ============================================================================


class TestMinimumExpectedTimeStrategyUpdate:
    """Additional tests for MinimumExpectedTimeStrategy update_queue."""

    async def test_update_queue_success(
        self, mock_predictor_client, instance_registry
    ):
        """Test successful queue update with error accumulation."""
        from swarmpilot.scheduler.models import (
            Instance,
            InstanceQueueExpectError,
        )

        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry
        )

        # Register an instance with initial queue state
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test-hardware",
            },
        )
        await instance_registry.register(instance)

        # Set initial queue info
        initial_queue = InstanceQueueExpectError(
            instance_id="inst-1", expected_time_ms=100.0, error_margin_ms=10.0
        )
        await instance_registry.update_queue_info("inst-1", initial_queue)

        # Create a prediction
        prediction = Prediction(
            instance_id="inst-1", predicted_time_ms=50.0, error_margin_ms=5.0
        )

        # Update queue
        await strategy.update_queue("inst-1", prediction)

        # Verify queue was updated correctly
        updated_queue = await instance_registry.get_queue_info("inst-1")
        assert isinstance(updated_queue, InstanceQueueExpectError)
        assert updated_queue.expected_time_ms == 150.0  # 100 + 50
        import math

        expected_error = math.sqrt(10.0**2 + 5.0**2)
        assert abs(updated_queue.error_margin_ms - expected_error) < 0.01

    async def test_update_queue_wrong_type(
        self, mock_predictor_client, instance_registry
    ):
        """Test update_queue with wrong queue info type."""
        from swarmpilot.scheduler.models import (
            Instance,
        )

        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry
        )

        # Register instance with probabilistic queue (wrong type)
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test-hardware",
            },
        )
        await instance_registry.register(instance)

        prob_queue = InstanceQueueProbabilistic(
            instance_id="inst-1", quantiles=[0.5, 0.9], values=[100.0, 200.0]
        )
        await instance_registry.update_queue_info("inst-1", prob_queue)

        # Try to update - should log warning but not crash
        prediction = Prediction(
            instance_id="inst-1", predicted_time_ms=50.0, error_margin_ms=5.0
        )

        await strategy.update_queue("inst-1", prediction)
        # Queue should remain unchanged due to type mismatch
        queue = await instance_registry.get_queue_info("inst-1")
        assert isinstance(queue, InstanceQueueProbabilistic)
        # Values should be unchanged
        assert queue.quantiles == [0.5, 0.9]
        assert queue.values == [100.0, 200.0]


class TestProbabilisticStrategyUpdate:
    """Additional tests for ProbabilisticSchedulingStrategy update_queue."""

    async def test_update_queue_with_quantiles(
        self, mock_predictor_client, instance_registry
    ):
        """Test update_queue with full quantile information."""
        import numpy as np

        from swarmpilot.scheduler.models import Instance

        # Set seed for reproducibility
        np.random.seed(42)

        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        # Register instance
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test-hardware",
            },
        )
        await instance_registry.register(instance)

        # Set initial queue
        initial_queue = InstanceQueueProbabilistic(
            instance_id="inst-1",
            quantiles=[0.5, 0.9, 0.95, 0.99],
            values=[100.0, 200.0, 300.0, 500.0],
        )
        await instance_registry.update_queue_info("inst-1", initial_queue)

        # Create prediction with quantiles
        prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=50.0,
            quantiles={0.5: 40.0, 0.9: 60.0, 0.95: 70.0, 0.99: 100.0},
        )

        # Update queue
        await strategy.update_queue("inst-1", prediction)

        # Verify queue was updated (values should be higher)
        updated_queue = await instance_registry.get_queue_info("inst-1")
        assert isinstance(updated_queue, InstanceQueueProbabilistic)
        # Values should be sum of queue + task samples
        assert all(
            v > initial_queue.values[i]
            for i, v in enumerate(updated_queue.values)
        )

    async def test_update_queue_without_quantiles(
        self, mock_predictor_client, instance_registry
    ):
        """Test update_queue fallback when prediction has no quantiles."""
        from swarmpilot.scheduler.models import Instance

        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        # Register instance
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test-hardware",
            },
        )
        await instance_registry.register(instance)

        # Set initial queue
        initial_queue = InstanceQueueProbabilistic(
            instance_id="inst-1",
            quantiles=[0.5, 0.9, 0.95, 0.99],
            values=[100.0, 200.0, 300.0, 500.0],
        )
        await instance_registry.update_queue_info("inst-1", initial_queue)

        # Create prediction without quantiles
        prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=50.0,
            quantiles=None,  # No quantiles
        )

        # Update queue (should use fallback)
        await strategy.update_queue("inst-1", prediction)

        # Verify queue was updated using predicted_time_ms
        updated_queue = await instance_registry.get_queue_info("inst-1")
        assert isinstance(updated_queue, InstanceQueueProbabilistic)
        assert updated_queue.values[0] == 150.0  # 100 + 50

    async def test_update_queue_no_existing_queue(
        self, mock_predictor_client, instance_registry
    ):
        """Test update_queue when no queue exists (line 517)."""
        from unittest.mock import AsyncMock

        from swarmpilot.scheduler.models import Instance

        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        # Register instance without setting queue info
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test-hardware",
            },
        )
        await instance_registry.register(instance)

        # Mock get_queue_info to return None (simulating missing queue)
        original_get_queue = instance_registry.get_queue_info
        instance_registry.get_queue_info = AsyncMock(return_value=None)

        # Create prediction
        prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=50.0,
            quantiles={0.5: 40.0, 0.9: 60.0, 0.95: 70.0, 0.99: 100.0},
        )

        # Update queue - should initialize new queue (line 517)
        await strategy.update_queue("inst-1", prediction)

        # Restore original method
        instance_registry.get_queue_info = original_get_queue

        # Verify queue was created and updated
        updated_queue = await instance_registry.get_queue_info("inst-1")
        assert isinstance(updated_queue, InstanceQueueProbabilistic)
        assert updated_queue.quantiles == [0.5, 0.9, 0.95, 0.99]
        # Values should be approximately the prediction quantiles
        assert all(v > 0 for v in updated_queue.values)

    async def test_update_queue_wrong_type(
        self, mock_predictor_client, instance_registry
    ):
        """Test update_queue with wrong queue info type."""
        from swarmpilot.scheduler.models import (
            Instance,
            InstanceQueueExpectError,
        )

        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        # Register instance with expect_error queue (wrong type)
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test-hardware",
            },
        )
        await instance_registry.register(instance)

        exp_queue = InstanceQueueExpectError(
            instance_id="inst-1", expected_time_ms=100.0, error_margin_ms=10.0
        )
        await instance_registry.update_queue_info("inst-1", exp_queue)

        # Try to update - should log warning but not crash
        prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=50.0,
            quantiles={0.5: 40.0, 0.9: 60.0},
        )

        await strategy.update_queue("inst-1", prediction)
        # Queue should remain unchanged due to type mismatch
        queue = await instance_registry.get_queue_info("inst-1")
        assert isinstance(queue, InstanceQueueExpectError)
        # Values should be unchanged
        assert queue.expected_time_ms == 100.0
        assert queue.error_margin_ms == 10.0

    async def test_select_with_queue_info(
        self, mock_predictor_client, instance_registry
    ):
        """Test selection considering queue information."""
        import numpy as np

        # Set seed for reproducibility
        np.random.seed(42)

        strategy = ProbabilisticSchedulingStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                quantiles={0.5: 80.0, 0.9: 120.0},
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=100.0,
                quantiles={0.5: 80.0, 0.9: 120.0},
            ),
        ]

        # Add queue info with different states
        queue_info = {
            "inst-1": InstanceQueueProbabilistic(
                instance_id="inst-1",
                quantiles=[0.5, 0.9],
                values=[50.0, 100.0],  # Shorter queue
            ),
            "inst-2": InstanceQueueProbabilistic(
                instance_id="inst-2",
                quantiles=[0.5, 0.9],
                values=[200.0, 400.0],  # Longer queue
            ),
        }

        # Selection should consider queue state
        selected = strategy.select_instance(predictions, queue_info)
        assert selected in ["inst-1", "inst-2"]


class TestSchedulingStrategyErrors:
    """Tests for error handling in scheduling strategies.

    The library predictor client raises standard Python exceptions
    (ValueError, ConnectionError, TimeoutError) directly. These
    propagate through get_predictions() without conversion.
    """

    async def test_get_predictions_model_not_found(
        self, mock_predictor_client, instance_registry
    ):
        """Test get_predictions when model is not found."""
        from swarmpilot.scheduler.models import Instance

        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry
        )

        # Library client raises ValueError for model not found
        mock_predictor_client.predict.side_effect = ValueError(
            "No trained model for model-1"
        )

        instances = [
            Instance(
                instance_id="inst-1",
                model_id="model-1",
                endpoint="http://localhost:8001",
                platform_info={
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test",
                },
            )
        ]

        with pytest.raises(ValueError, match="No trained model"):
            await strategy.get_predictions("model-1", {}, instances)

    async def test_get_predictions_invalid_metadata(
        self, mock_predictor_client, instance_registry
    ):
        """Test get_predictions with invalid metadata."""
        from swarmpilot.scheduler.models import Instance

        strategy = MinimumExpectedTimeStrategy(
            mock_predictor_client, instance_registry
        )

        # Library client raises ValueError for invalid features
        mock_predictor_client.predict.side_effect = ValueError(
            "Invalid task metadata: missing required features"
        )

        instances = [
            Instance(
                instance_id="inst-1",
                model_id="model-1",
                endpoint="http://localhost:8001",
                platform_info={
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test",
                },
            )
        ]

        with pytest.raises(ValueError, match="Invalid task metadata"):
            await strategy.get_predictions("model-1", {}, instances)


class TestRoundRobinStrategyUpdate:
    """Additional tests for RoundRobinStrategy."""

    async def test_update_queue_noop(
        self, mock_predictor_client, instance_registry
    ):
        """Test that RoundRobinStrategy update_queue is a no-op."""
        from swarmpilot.scheduler.models import Instance

        strategy = RoundRobinStrategy(mock_predictor_client, instance_registry)

        # Register instance
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test-hardware",
            },
        )
        await instance_registry.register(instance)

        # Get initial queue state
        initial_queue = await instance_registry.get_queue_info("inst-1")

        # Create prediction
        prediction = Prediction(instance_id="inst-1", predicted_time_ms=100.0)

        # Update queue (should be no-op)
        await strategy.update_queue("inst-1", prediction)

        # Verify queue unchanged
        final_queue = await instance_registry.get_queue_info("inst-1")
        assert final_queue == initial_queue


# ============================================================================
# RoundRobinStrategy.select_instance Tests
# ============================================================================


class TestRoundRobinStrategySelect:
    """Tests for RoundRobinStrategy select_instance method."""

    async def test_select_instance_empty(
        self, mock_predictor_client, instance_registry
    ):
        """Test select_instance with empty predictions."""
        strategy = RoundRobinStrategy(mock_predictor_client, instance_registry)

        selected = strategy.select_instance([], {})
        assert selected is None

    async def test_select_instance_single(
        self, mock_predictor_client, instance_registry
    ):
        """Test select_instance with single prediction."""
        strategy = RoundRobinStrategy(mock_predictor_client, instance_registry)

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0)
        ]

        selected = strategy.select_instance(predictions, {})
        assert selected == "inst-1"

    async def test_select_instance_random_choice(
        self, mock_predictor_client, instance_registry
    ):
        """Test select_instance makes a random choice."""
        strategy = RoundRobinStrategy(mock_predictor_client, instance_registry)

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=200.0),
            Prediction(instance_id="inst-3", predicted_time_ms=150.0),
        ]

        # Run multiple times to verify randomness
        selected_set = set()
        for _ in range(50):
            selected = strategy.select_instance(predictions, {})
            selected_set.add(selected)

        # Should have selected at least 2 different instances
        assert len(selected_set) >= 2


# ============================================================================
# PowerOfTwoStrategy Tests
# ============================================================================


class TestPowerOfTwoStrategy:
    """Tests for PowerOfTwoStrategy."""

    async def test_select_instance_empty(
        self, mock_predictor_client, instance_registry
    ):
        """Test select_instance with empty predictions."""
        from swarmpilot.scheduler.algorithms import PowerOfTwoStrategy

        strategy = PowerOfTwoStrategy(mock_predictor_client, instance_registry)

        selected = strategy.select_instance([], {})
        assert selected is None

    async def test_select_instance_with_queue_info(
        self, mock_predictor_client, instance_registry
    ):
        """Test select_instance with queue info."""
        from swarmpilot.scheduler.algorithms import PowerOfTwoStrategy

        strategy = PowerOfTwoStrategy(mock_predictor_client, instance_registry)

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=100.0),
        ]

        # Different queue expected times
        queue_info = {
            "inst-1": InstanceQueueExpectError(
                instance_id="inst-1",
                expected_time_ms=500.0,
                error_margin_ms=10.0,
            ),
            "inst-2": InstanceQueueExpectError(
                instance_id="inst-2",
                expected_time_ms=100.0,
                error_margin_ms=10.0,
            ),
            "inst-3": InstanceQueueExpectError(
                instance_id="inst-3",
                expected_time_ms=200.0,
                error_margin_ms=10.0,
            ),
        }

        selected = strategy.select_instance(predictions, queue_info)
        assert selected in ["inst-1", "inst-2", "inst-3"]

    async def test_select_instance_without_queue_info(
        self, mock_predictor_client, instance_registry
    ):
        """Test select_instance without queue info (fallback)."""
        from swarmpilot.scheduler.algorithms import PowerOfTwoStrategy

        strategy = PowerOfTwoStrategy(mock_predictor_client, instance_registry)

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=200.0),
        ]

        selected = strategy.select_instance(predictions, {})
        assert selected in ["inst-1", "inst-2"]

    async def test_update_queue(self, mock_predictor_client, instance_registry):
        """Test update_queue updates queue with error accumulation."""
        import math

        from swarmpilot.scheduler.algorithms import PowerOfTwoStrategy
        from swarmpilot.scheduler.models import Instance

        strategy = PowerOfTwoStrategy(mock_predictor_client, instance_registry)

        # Register instance
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test",
            },
        )
        await instance_registry.register(instance)

        # Initialize queue
        initial_queue = InstanceQueueExpectError(
            instance_id="inst-1",
            expected_time_ms=100.0,
            error_margin_ms=10.0,
        )
        await instance_registry.update_queue_info("inst-1", initial_queue)

        prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=50.0,
            error_margin_ms=5.0,
        )

        await strategy.update_queue("inst-1", prediction)

        updated_queue = await instance_registry.get_queue_info("inst-1")
        assert updated_queue.expected_time_ms == 150.0  # 100 + 50
        expected_error = math.sqrt(10.0**2 + 5.0**2)
        assert abs(updated_queue.error_margin_ms - expected_error) < 0.01


# ============================================================================
# MinimumExpectedTimeServerlessStrategy Tests
# ============================================================================


class TestMinimumExpectedTimeServerlessStrategy:
    """Tests for MinimumExpectedTimeServerlessStrategy."""

    async def test_select_instance_empty(
        self, mock_predictor_client, instance_registry
    ):
        """Test select_instance with empty predictions."""
        from swarmpilot.scheduler.algorithms import (
            MinimumExpectedTimeServerlessStrategy,
        )

        strategy = MinimumExpectedTimeServerlessStrategy(
            mock_predictor_client, instance_registry
        )

        selected = strategy.select_instance([], {})
        assert selected is None

    async def test_select_instance_with_queue_info(
        self, mock_predictor_client, instance_registry
    ):
        """Test select_instance with queue info."""
        from swarmpilot.scheduler.algorithms import (
            MinimumExpectedTimeServerlessStrategy,
        )

        strategy = MinimumExpectedTimeServerlessStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=100.0),
        ]

        # inst-2 has lowest total (expected + error + prediction)
        queue_info = {
            "inst-1": InstanceQueueExpectError(
                instance_id="inst-1",
                expected_time_ms=200.0,
                error_margin_ms=50.0,
            ),
            "inst-2": InstanceQueueExpectError(
                instance_id="inst-2",
                expected_time_ms=50.0,
                error_margin_ms=10.0,
            ),
            "inst-3": InstanceQueueExpectError(
                instance_id="inst-3",
                expected_time_ms=100.0,
                error_margin_ms=20.0,
            ),
        }

        selected = strategy.select_instance(predictions, queue_info)
        # inst-1: 200 + 50 + 100 = 350
        # inst-2: 50 + 10 + 100 = 160
        # inst-3: 100 + 20 + 100 = 220
        assert selected == "inst-2"

    async def test_select_instance_without_queue_info(
        self, mock_predictor_client, instance_registry
    ):
        """Test select_instance without queue info (fallback)."""
        from swarmpilot.scheduler.algorithms import (
            MinimumExpectedTimeServerlessStrategy,
        )

        strategy = MinimumExpectedTimeServerlessStrategy(
            mock_predictor_client, instance_registry
        )

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=200.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=150.0),
        ]

        selected = strategy.select_instance(predictions, {})
        # Fallback to min predicted time
        assert selected == "inst-2"

    async def test_update_queue(self, mock_predictor_client, instance_registry):
        """Test update_queue updates queue with error accumulation."""
        import math

        from swarmpilot.scheduler.algorithms import (
            MinimumExpectedTimeServerlessStrategy,
        )
        from swarmpilot.scheduler.models import Instance

        strategy = MinimumExpectedTimeServerlessStrategy(
            mock_predictor_client, instance_registry
        )

        # Register instance
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test",
            },
        )
        await instance_registry.register(instance)

        # Initialize queue
        initial_queue = InstanceQueueExpectError(
            instance_id="inst-1",
            expected_time_ms=100.0,
            error_margin_ms=10.0,
        )
        await instance_registry.update_queue_info("inst-1", initial_queue)

        prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=50.0,
            error_margin_ms=5.0,
        )

        await strategy.update_queue("inst-1", prediction)

        updated_queue = await instance_registry.get_queue_info("inst-1")
        assert updated_queue.expected_time_ms == 150.0  # 100 + 50
        expected_error = math.sqrt(10.0**2 + 5.0**2)
        assert abs(updated_queue.error_margin_ms - expected_error) < 0.01

    async def test_update_queue_type_mismatch_skips(
        self, mock_predictor_client, instance_registry
    ):
        """Test update_queue skips when queue type is wrong (line 929-936)."""
        from swarmpilot.scheduler.algorithms import (
            MinimumExpectedTimeServerlessStrategy,
        )
        from swarmpilot.scheduler.models import Instance

        strategy = MinimumExpectedTimeServerlessStrategy(
            mock_predictor_client, instance_registry
        )

        # Register instance - this creates a default InstanceQueueProbabilistic
        instance = Instance(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test",
            },
        )
        await instance_registry.register(instance)

        # The default queue is InstanceQueueProbabilistic, not InstanceQueueExpectError
        # So update_queue should log a warning and skip the update
        prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=50.0,
            error_margin_ms=5.0,
        )

        # This should not raise and should skip the update due to type mismatch
        await strategy.update_queue("inst-1", prediction)

        # Queue should remain as InstanceQueueProbabilistic (not updated)

        updated_queue = await instance_registry.get_queue_info("inst-1")
        assert isinstance(updated_queue, InstanceQueueProbabilistic)


# ============================================================================
# get_strategy Factory Tests (additional cases)
# ============================================================================


class TestGetStrategyAdditional:
    """Additional tests for get_strategy factory function."""

    async def test_get_random_strategy(
        self, mock_predictor_client, instance_registry
    ):
        """Test get_strategy returns RandomStrategy."""
        from swarmpilot.scheduler.algorithms import RandomStrategy

        strategy = get_strategy(
            "random", mock_predictor_client, instance_registry
        )

        assert isinstance(strategy, RandomStrategy)

    async def test_get_po2_strategy(
        self, mock_predictor_client, instance_registry
    ):
        """Test get_strategy returns PowerOfTwoStrategy."""
        from swarmpilot.scheduler.algorithms import PowerOfTwoStrategy

        strategy = get_strategy("po2", mock_predictor_client, instance_registry)

        assert isinstance(strategy, PowerOfTwoStrategy)

    async def test_get_serverless_strategy(
        self, mock_predictor_client, instance_registry
    ):
        """Test get_strategy returns MinimumExpectedTimeServerlessStrategy."""
        from swarmpilot.scheduler.algorithms import (
            MinimumExpectedTimeServerlessStrategy,
        )

        strategy = get_strategy(
            "serverless", mock_predictor_client, instance_registry
        )

        assert isinstance(strategy, MinimumExpectedTimeServerlessStrategy)
