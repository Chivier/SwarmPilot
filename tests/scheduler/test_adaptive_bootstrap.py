"""Unit tests for AdaptiveBootstrapStrategy.

Tests cold-start round-robin, warm probabilistic delegation,
cold-to-warm transition, thread safety, and factory integration.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from swarmpilot.scheduler.algorithms.adaptive_bootstrap import (
    AdaptiveBootstrapStrategy,
)
from swarmpilot.scheduler.algorithms.base import ScheduleResult
from swarmpilot.scheduler.algorithms.factory import get_strategy
from swarmpilot.scheduler.clients.models import Prediction

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_storage():
    """Create a mock ModelStorage with configurable model_exists."""
    storage = MagicMock()
    storage.generate_model_key.return_value = "test_key"
    storage.model_exists.return_value = False
    return storage


@pytest.fixture
def mock_predictor_client(mock_storage):
    """Create a mock PredictorClient with PredictorLowLevel storage."""
    client = MagicMock()
    client._low_level = MagicMock()
    client._low_level._storage = mock_storage
    client.predict = AsyncMock()
    return client


@pytest.fixture
def mock_instance_registry():
    """Create a mock InstanceRegistry."""
    registry = MagicMock()
    registry._quantiles = [0.5, 0.9, 0.95, 0.99]
    registry.get_all_queue_info = AsyncMock(return_value={})
    registry.get_queue_info = AsyncMock(return_value=None)
    registry.update_queue_info = AsyncMock()
    return registry


@pytest.fixture
def make_instance():
    """Factory for creating mock Instance objects."""

    def _make(instance_id: str, platform_info: dict | None = None):
        inst = MagicMock()
        inst.instance_id = instance_id
        inst.platform_info = platform_info or {
            "software_name": "docker",
            "software_version": "20.10",
            "hardware_name": "test-gpu",
        }
        return inst

    return _make


@pytest.fixture
def instances(make_instance):
    """Create a list of 3 test instances."""
    return [
        make_instance("inst-1"),
        make_instance("inst-2"),
        make_instance("inst-3"),
    ]


@pytest.fixture
def strategy(mock_predictor_client, mock_instance_registry):
    """Create an AdaptiveBootstrapStrategy instance."""
    return AdaptiveBootstrapStrategy(
        predictor_client=mock_predictor_client,
        instance_registry=mock_instance_registry,
        target_quantile=0.9,
    )


# ============================================================================
# Cold Model Tests (Round-Robin)
# ============================================================================


class TestColdModelRoundRobin:
    """Tests for cold-start models using round-robin."""

    @pytest.mark.asyncio
    async def test_cold_model_uses_round_robin(
        self, strategy, instances, mock_predictor_client
    ):
        """Cold models should use round-robin, no predict() call."""
        # model_exists returns False (cold)
        result = await strategy.schedule_task("model-1", {}, instances)

        assert result.selected_instance_id == "inst-1"
        assert result.selected_prediction is None
        mock_predictor_client.predict.assert_not_called()

    @pytest.mark.asyncio
    async def test_cold_round_robin_rotates(
        self, strategy, instances, mock_predictor_client
    ):
        """Successive cold calls should rotate through instances."""
        r1 = await strategy.schedule_task("model-1", {}, instances)
        r2 = await strategy.schedule_task("model-1", {}, instances)
        r3 = await strategy.schedule_task("model-1", {}, instances)
        r4 = await strategy.schedule_task("model-1", {}, instances)

        assert r1.selected_instance_id == "inst-1"
        assert r2.selected_instance_id == "inst-2"
        assert r3.selected_instance_id == "inst-3"
        assert r4.selected_instance_id == "inst-1"  # wraps around

    @pytest.mark.asyncio
    async def test_cold_returns_none_prediction(self, strategy, instances):
        """Cold-start should return ScheduleResult with None prediction."""
        result = await strategy.schedule_task("model-1", {}, instances)
        assert result.selected_prediction is None


# ============================================================================
# Warm Model Tests (Probabilistic Delegation)
# ============================================================================


class TestWarmModelProbabilistic:
    """Tests for warm models delegating to probabilistic."""

    @pytest.mark.asyncio
    async def test_warm_model_uses_probabilistic(
        self, strategy, instances, mock_storage
    ):
        """Warm models should delegate to ProbabilisticSchedulingStrategy."""
        mock_storage.model_exists.return_value = True

        # Mock the probabilistic delegate's schedule_task
        expected = ScheduleResult(
            selected_instance_id="inst-2",
            selected_prediction=Prediction(
                instance_id="inst-2",
                predicted_time_ms=100.0,
                confidence=None,
                quantiles={0.5: 80.0, 0.9: 120.0},
            ),
        )
        strategy._probabilistic.schedule_task = AsyncMock(return_value=expected)

        result = await strategy.schedule_task("model-1", {}, instances)

        assert result.selected_instance_id == "inst-2"
        assert result.selected_prediction is not None
        strategy._probabilistic.schedule_task.assert_awaited_once_with(
            "model-1", {}, instances
        )

    @pytest.mark.asyncio
    async def test_warm_checks_all_platforms(
        self, strategy, mock_storage, make_instance
    ):
        """All platforms must be warm to delegate to probabilistic."""
        # Two different platforms
        inst_a = make_instance(
            "inst-a",
            {
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "gpu-A",
            },
        )
        inst_b = make_instance(
            "inst-b",
            {
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "gpu-B",
            },
        )

        # First call: both cold → round-robin
        mock_storage.model_exists.return_value = False
        result = await strategy.schedule_task("model-1", {}, [inst_a, inst_b])
        assert result.selected_prediction is None

        # Second call: only one warm → still round-robin
        mock_storage.model_exists.side_effect = [True, False]
        result = await strategy.schedule_task("model-1", {}, [inst_a, inst_b])
        assert result.selected_prediction is None


# ============================================================================
# Transition Tests
# ============================================================================


class TestColdToWarmTransition:
    """Tests for transitioning from cold to warm."""

    @pytest.mark.asyncio
    async def test_transition_cold_to_warm(self, strategy, instances, mock_storage):
        """Strategy switches to probabilistic when model becomes available."""
        # Start cold
        mock_storage.model_exists.return_value = False
        r1 = await strategy.schedule_task("model-1", {}, instances)
        assert r1.selected_prediction is None

        # Model gets trained → warm
        mock_storage.model_exists.return_value = True
        expected = ScheduleResult(
            selected_instance_id="inst-1",
            selected_prediction=Prediction(
                instance_id="inst-1",
                predicted_time_ms=50.0,
                confidence=None,
                quantiles={0.5: 50.0, 0.9: 80.0},
            ),
        )
        strategy._probabilistic.schedule_task = AsyncMock(return_value=expected)

        r2 = await strategy.schedule_task("model-1", {}, instances)
        assert r2.selected_prediction is not None
        strategy._probabilistic.schedule_task.assert_awaited_once()


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestRoundRobinThreadSafety:
    """Tests for thread-safe round-robin counter."""

    @pytest.mark.asyncio
    async def test_round_robin_thread_safety(self, strategy, instances, mock_storage):
        """Concurrent calls should produce distinct selections."""
        mock_storage.model_exists.return_value = False

        results = []
        num_calls = 30

        async def schedule_one():
            r = await strategy.schedule_task("model-1", {}, instances)
            results.append(r.selected_instance_id)

        tasks = [schedule_one() for _ in range(num_calls)]
        await asyncio.gather(*tasks)

        # All 30 calls should have produced a result
        assert len(results) == num_calls

        # Each instance should be selected exactly 10 times (30 / 3)
        for inst_id in ["inst-1", "inst-2", "inst-3"]:
            assert results.count(inst_id) == 10

    def test_counter_increments_atomically(self, strategy):
        """Counter should not skip or duplicate under contention."""
        results = []
        num_threads = 100

        def increment():
            with strategy._rr_lock:
                val = strategy._rr_counter
                strategy._rr_counter += 1
            results.append(val)

        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All values should be unique (no skips or duplicates)
        assert len(set(results)) == num_threads


# ============================================================================
# Factory Integration Tests
# ============================================================================


class TestFactoryIntegration:
    """Tests for factory creating AdaptiveBootstrapStrategy."""

    def test_factory_creates_adaptive(
        self, mock_predictor_client, mock_instance_registry
    ):
        """get_strategy('adaptive_bootstrap') returns correct type."""
        s = get_strategy(
            "adaptive_bootstrap",
            mock_predictor_client,
            mock_instance_registry,
        )
        assert isinstance(s, AdaptiveBootstrapStrategy)

    def test_factory_default_is_adaptive(
        self, mock_predictor_client, mock_instance_registry
    ):
        """Unknown strategy name should default to AdaptiveBootstrapStrategy."""
        s = get_strategy(
            "unknown_name",
            mock_predictor_client,
            mock_instance_registry,
        )
        assert isinstance(s, AdaptiveBootstrapStrategy)

    def test_factory_preserves_target_quantile(
        self, mock_predictor_client, mock_instance_registry
    ):
        """Factory should pass target_quantile through."""
        s = get_strategy(
            "adaptive_bootstrap",
            mock_predictor_client,
            mock_instance_registry,
            target_quantile=0.75,
        )
        assert isinstance(s, AdaptiveBootstrapStrategy)
        assert s.target_quantile == 0.75


# ============================================================================
# Worker Queue Manager Wiring
# ============================================================================


class TestWorkerQueueManagerWiring:
    """Tests for set_worker_queue_manager propagation."""

    def test_set_worker_queue_manager_propagates(self, strategy):
        """Manager should be set on both self and probabilistic delegate."""
        mock_manager = MagicMock()
        strategy.set_worker_queue_manager(mock_manager)

        assert strategy._worker_queue_manager is mock_manager
        assert strategy._probabilistic._worker_queue_manager is mock_manager

    def test_get_prediction_type(self, strategy):
        """Prediction type should be 'quantile'."""
        assert strategy.get_prediction_type() == "quantile"
