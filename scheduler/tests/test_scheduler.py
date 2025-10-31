"""
Unit tests for scheduling strategies.

Tests all scheduling strategies and the factory function.
"""

import pytest

from src.scheduler import (
    SchedulingStrategy,
    MinimumExpectedTimeStrategy,
    ProbabilisticSchedulingStrategy,
    RoundRobinStrategy,
    get_strategy,
)
from src.predictor_client import Prediction


# ============================================================================
# MinimumExpectedTimeStrategy Tests
# ============================================================================

class TestMinimumExpectedTimeStrategy:
    """Tests for MinimumExpectedTimeStrategy."""

    def test_select_instance_with_minimum_time(self):
        """Test selecting instance with minimum predicted time."""
        strategy = MinimumExpectedTimeStrategy()

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=200.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=150.0),
        ]

        selected = strategy.select_instance(predictions)
        assert selected == "inst-2"

    def test_select_instance_empty_list(self):
        """Test selection with empty predictions list."""
        strategy = MinimumExpectedTimeStrategy()

        selected = strategy.select_instance([])
        assert selected is None

    def test_select_instance_single_prediction(self):
        """Test selection with single prediction."""
        strategy = MinimumExpectedTimeStrategy()

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
        ]

        selected = strategy.select_instance(predictions)
        assert selected == "inst-1"

    def test_select_instance_equal_times(self):
        """Test selection when multiple instances have equal times."""
        strategy = MinimumExpectedTimeStrategy()

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=100.0),
        ]

        # Should select the first one encountered
        selected = strategy.select_instance(predictions)
        assert selected == "inst-1"

    def test_select_with_extreme_values(self):
        """Test selection with extreme time values."""
        strategy = MinimumExpectedTimeStrategy()

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=10000.0),
            Prediction(instance_id="inst-2", predicted_time_ms=1.0),
            Prediction(instance_id="inst-3", predicted_time_ms=5000.0),
        ]

        selected = strategy.select_instance(predictions)
        assert selected == "inst-2"


# ============================================================================
# ProbabilisticSchedulingStrategy Tests
# ============================================================================

class TestProbabilisticSchedulingStrategy:
    """Tests for ProbabilisticSchedulingStrategy."""

    def test_default_target_quantile(self):
        """Test that default target quantile is 0.9."""
        strategy = ProbabilisticSchedulingStrategy()
        assert strategy.target_quantile == 0.9

    def test_custom_target_quantile(self):
        """Test setting custom target quantile."""
        strategy = ProbabilisticSchedulingStrategy(target_quantile=0.95)
        assert strategy.target_quantile == 0.95

    def test_select_based_on_quantile(self):
        """Test selection based on target quantile value."""
        strategy = ProbabilisticSchedulingStrategy(target_quantile=0.9)

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                quantiles={0.5: 80.0, 0.9: 150.0}
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=120.0,
                quantiles={0.5: 100.0, 0.9: 130.0}
            ),
            Prediction(
                instance_id="inst-3",
                predicted_time_ms=90.0,
                quantiles={0.5: 70.0, 0.9: 200.0}
            ),
        ]

        # Should select inst-2 with quantile 0.9 = 130.0 (lowest)
        selected = strategy.select_instance(predictions)
        assert selected == "inst-2"

    def test_select_empty_list(self):
        """Test selection with empty predictions list."""
        strategy = ProbabilisticSchedulingStrategy()

        selected = strategy.select_instance([])
        assert selected is None

    def test_fallback_to_min_time(self):
        """Test fallback to minimum expected time when quantile not available."""
        strategy = ProbabilisticSchedulingStrategy(target_quantile=0.9)

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=200.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=150.0),
        ]

        # No quantiles available, should fallback to min time
        selected = strategy.select_instance(predictions)
        assert selected == "inst-2"

    def test_partial_quantile_availability(self):
        """Test when only some predictions have the target quantile."""
        strategy = ProbabilisticSchedulingStrategy(target_quantile=0.9)

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=200.0,
                quantiles={0.5: 150.0, 0.9: 250.0}
            ),
            Prediction(
                instance_id="inst-3",
                predicted_time_ms=150.0,
                quantiles={0.5: 100.0, 0.9: 180.0}
            ),
        ]

        # Should select from instances with quantiles (inst-2 or inst-3)
        selected = strategy.select_instance(predictions)
        assert selected == "inst-3"  # Lower 0.9 quantile value

    def test_quantile_not_in_dict(self):
        """Test when quantiles exist but target quantile not in them."""
        strategy = ProbabilisticSchedulingStrategy(target_quantile=0.95)

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                quantiles={0.5: 80.0, 0.9: 150.0}  # No 0.95
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=150.0,
                quantiles={0.5: 100.0, 0.9: 200.0}  # No 0.95
            ),
        ]

        # Should fallback to min time
        selected = strategy.select_instance(predictions)
        assert selected == "inst-1"

    def test_select_with_equal_quantile_values(self):
        """Test selection when multiple instances have equal quantile values."""
        strategy = ProbabilisticSchedulingStrategy(target_quantile=0.9)

        predictions = [
            Prediction(
                instance_id="inst-1",
                predicted_time_ms=100.0,
                quantiles={0.9: 150.0}
            ),
            Prediction(
                instance_id="inst-2",
                predicted_time_ms=120.0,
                quantiles={0.9: 150.0}
            ),
        ]

        # Should select first one encountered
        selected = strategy.select_instance(predictions)
        assert selected == "inst-1"


# ============================================================================
# RoundRobinStrategy Tests
# ============================================================================

class TestRoundRobinStrategy:
    """Tests for RoundRobinStrategy."""

    def test_round_robin_cycling(self):
        """Test that strategy cycles through instances."""
        strategy = RoundRobinStrategy()

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=100.0),
        ]

        # First call
        selected1 = strategy.select_instance(predictions)
        assert selected1 == "inst-1"

        # Second call
        selected2 = strategy.select_instance(predictions)
        assert selected2 == "inst-2"

        # Third call
        selected3 = strategy.select_instance(predictions)
        assert selected3 == "inst-3"

        # Fourth call - should wrap around
        selected4 = strategy.select_instance(predictions)
        assert selected4 == "inst-1"

    def test_round_robin_empty_list(self):
        """Test selection with empty predictions list."""
        strategy = RoundRobinStrategy()

        selected = strategy.select_instance([])
        assert selected is None

    def test_round_robin_single_instance(self):
        """Test cycling with single instance."""
        strategy = RoundRobinStrategy()

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
        ]

        # Should always return the same instance
        assert strategy.select_instance(predictions) == "inst-1"
        assert strategy.select_instance(predictions) == "inst-1"
        assert strategy.select_instance(predictions) == "inst-1"

    def test_round_robin_counter_persistence(self):
        """Test that counter persists across calls."""
        strategy = RoundRobinStrategy()

        predictions = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
        ]

        # Make several calls
        for _ in range(10):
            strategy.select_instance(predictions)

        # Counter should be at 10, next selection should be inst-1 (10 % 2 = 0)
        selected = strategy.select_instance(predictions)
        assert selected == "inst-1"

    def test_round_robin_different_list_sizes(self):
        """Test round robin with changing prediction list size."""
        strategy = RoundRobinStrategy()

        # Start with 3 instances
        predictions_3 = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
            Prediction(instance_id="inst-3", predicted_time_ms=100.0),
        ]

        strategy.select_instance(predictions_3)  # counter = 0, select inst-1
        strategy.select_instance(predictions_3)  # counter = 1, select inst-2

        # Now use 2 instances
        predictions_2 = [
            Prediction(instance_id="inst-1", predicted_time_ms=100.0),
            Prediction(instance_id="inst-2", predicted_time_ms=100.0),
        ]

        # counter = 2, 2 % 2 = 0, should select inst-1
        selected = strategy.select_instance(predictions_2)
        assert selected == "inst-1"


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestGetStrategy:
    """Tests for get_strategy factory function."""

    def test_get_min_time_strategy(self):
        """Test getting MinimumExpectedTimeStrategy."""
        strategy = get_strategy("min_time")
        assert isinstance(strategy, MinimumExpectedTimeStrategy)

    def test_get_probabilistic_strategy(self):
        """Test getting ProbabilisticSchedulingStrategy."""
        strategy = get_strategy("probabilistic")
        assert isinstance(strategy, ProbabilisticSchedulingStrategy)

    def test_get_round_robin_strategy(self):
        """Test getting RoundRobinStrategy."""
        strategy = get_strategy("round_robin")
        assert isinstance(strategy, RoundRobinStrategy)

    def test_unknown_strategy_defaults_to_probabilistic(self):
        """Test that unknown strategy name defaults to probabilistic."""
        strategy = get_strategy("unknown_strategy")
        assert isinstance(strategy, ProbabilisticSchedulingStrategy)

    def test_default_strategy(self):
        """Test default strategy when no name provided."""
        strategy = get_strategy()
        assert isinstance(strategy, ProbabilisticSchedulingStrategy)

    def test_case_sensitivity(self):
        """Test that strategy names are case-sensitive."""
        # "MIN_TIME" should not match "min_time"
        strategy = get_strategy("MIN_TIME")
        # Should default to probabilistic
        assert isinstance(strategy, ProbabilisticSchedulingStrategy)
