"""Unit tests for PYLET-018: Queue-Aware Scheduling Integration.

Tests the queue state adapter that converts scheduler-side queue state
(from WorkerQueueManager) to the format existing algorithms expect
(InstanceQueueExpectError).
"""

from unittest.mock import MagicMock

from src.algorithms.queue_state_adapter import (
    get_all_queue_info_from_manager,
    get_queue_info_from_manager,
)
from src.models.queue import InstanceQueueExpectError
from src.services.worker_queue_manager import WorkerQueueManager
from src.services.worker_queue_thread import WorkerQueueThread


class TestGetQueueInfoFromManager:
    """Tests for get_queue_info_from_manager adapter function."""

    def test_returns_queue_info_for_registered_worker(self):
        """Test adapter returns correct queue info for a registered worker."""
        # Arrange
        mock_manager = MagicMock(spec=WorkerQueueManager)
        mock_thread = MagicMock(spec=WorkerQueueThread)
        mock_thread.get_estimated_wait_time.return_value = 500.0
        mock_manager.get_worker.return_value = mock_thread

        # Act
        result = get_queue_info_from_manager(
            worker_queue_manager=mock_manager,
            instance_id="worker-1",
            avg_exec_time_ms=100.0,
        )

        # Assert
        assert isinstance(result, InstanceQueueExpectError)
        assert result.instance_id == "worker-1"
        assert result.expected_time_ms == 500.0
        assert result.error_margin_ms == 20.0  # 20% of avg_exec_time_ms
        mock_thread.get_estimated_wait_time.assert_called_once_with(100.0)

    def test_returns_zero_values_for_unknown_worker(self):
        """Test adapter returns zero values for an unregistered worker."""
        # Arrange
        mock_manager = MagicMock(spec=WorkerQueueManager)
        mock_manager.get_worker.return_value = None

        # Act
        result = get_queue_info_from_manager(
            worker_queue_manager=mock_manager,
            instance_id="unknown-worker",
            avg_exec_time_ms=100.0,
        )

        # Assert
        assert isinstance(result, InstanceQueueExpectError)
        assert result.instance_id == "unknown-worker"
        assert result.expected_time_ms == 0.0
        assert result.error_margin_ms == 0.0

    def test_handles_zero_avg_exec_time(self):
        """Test adapter handles zero average execution time correctly."""
        # Arrange
        mock_manager = MagicMock(spec=WorkerQueueManager)
        mock_thread = MagicMock(spec=WorkerQueueThread)
        mock_thread.get_estimated_wait_time.return_value = 0.0
        mock_manager.get_worker.return_value = mock_thread

        # Act
        result = get_queue_info_from_manager(
            worker_queue_manager=mock_manager,
            instance_id="worker-1",
            avg_exec_time_ms=0.0,
        )

        # Assert
        assert result.expected_time_ms == 0.0
        assert result.error_margin_ms == 0.0

    def test_calculates_error_margin_as_20_percent(self):
        """Test error margin is 20% of average execution time."""
        # Arrange
        mock_manager = MagicMock(spec=WorkerQueueManager)
        mock_thread = MagicMock(spec=WorkerQueueThread)
        mock_thread.get_estimated_wait_time.return_value = 1000.0
        mock_manager.get_worker.return_value = mock_thread

        # Act
        result = get_queue_info_from_manager(
            worker_queue_manager=mock_manager,
            instance_id="worker-1",
            avg_exec_time_ms=250.0,
        )

        # Assert
        assert result.error_margin_ms == 50.0  # 20% of 250.0


class TestGetAllQueueInfoFromManager:
    """Tests for get_all_queue_info_from_manager function."""

    def test_returns_info_for_all_instances(self):
        """Test batch function returns queue info for all specified instances."""
        # Arrange
        mock_manager = MagicMock(spec=WorkerQueueManager)

        mock_thread_1 = MagicMock(spec=WorkerQueueThread)
        mock_thread_1.get_estimated_wait_time.return_value = 100.0

        mock_thread_2 = MagicMock(spec=WorkerQueueThread)
        mock_thread_2.get_estimated_wait_time.return_value = 200.0

        mock_manager.get_worker.side_effect = lambda id: {
            "worker-1": mock_thread_1,
            "worker-2": mock_thread_2,
        }.get(id)

        # Act
        result = get_all_queue_info_from_manager(
            worker_queue_manager=mock_manager,
            instance_ids=["worker-1", "worker-2"],
            avg_exec_time_ms=50.0,
        )

        # Assert
        assert len(result) == 2
        assert "worker-1" in result
        assert "worker-2" in result
        assert result["worker-1"].expected_time_ms == 100.0
        assert result["worker-2"].expected_time_ms == 200.0

    def test_returns_empty_dict_for_empty_list(self):
        """Test batch function returns empty dict for empty instance list."""
        # Arrange
        mock_manager = MagicMock(spec=WorkerQueueManager)

        # Act
        result = get_all_queue_info_from_manager(
            worker_queue_manager=mock_manager,
            instance_ids=[],
            avg_exec_time_ms=100.0,
        )

        # Assert
        assert result == {}

    def test_handles_mixed_registered_unregistered_workers(self):
        """Test batch function handles mix of registered and unregistered workers."""
        # Arrange
        mock_manager = MagicMock(spec=WorkerQueueManager)

        mock_thread = MagicMock(spec=WorkerQueueThread)
        mock_thread.get_estimated_wait_time.return_value = 500.0

        mock_manager.get_worker.side_effect = lambda id: (
            mock_thread if id == "worker-1" else None
        )

        # Act
        result = get_all_queue_info_from_manager(
            worker_queue_manager=mock_manager,
            instance_ids=["worker-1", "worker-unknown"],
            avg_exec_time_ms=100.0,
        )

        # Assert
        assert len(result) == 2
        assert result["worker-1"].expected_time_ms == 500.0
        assert result["worker-unknown"].expected_time_ms == 0.0


class TestSchedulingStrategyIntegration:
    """Tests for SchedulingStrategy base class queue manager integration."""

    def test_set_worker_queue_manager(self):
        """Test setting worker queue manager on strategy."""
        from src.algorithms.base import SchedulingStrategy

        # Create a concrete implementation for testing
        class TestStrategy(SchedulingStrategy):
            def select_instance(self, predictions, queue_info):
                return None

            def update_queue(self, instance_id, prediction):
                pass

        # Arrange
        mock_predictor = MagicMock()
        mock_registry = MagicMock()
        mock_manager = MagicMock(spec=WorkerQueueManager)

        strategy = TestStrategy(mock_predictor, mock_registry)

        # Act
        strategy.set_worker_queue_manager(mock_manager)

        # Assert
        assert strategy._worker_queue_manager is mock_manager

    def test_get_scheduler_queue_depth_with_manager(self):
        """Test getting queue depth when manager is set."""
        from src.algorithms.base import SchedulingStrategy

        class TestStrategy(SchedulingStrategy):
            def select_instance(self, predictions, queue_info):
                return None

            def update_queue(self, instance_id, prediction):
                pass

        # Arrange
        mock_predictor = MagicMock()
        mock_registry = MagicMock()
        mock_manager = MagicMock(spec=WorkerQueueManager)
        mock_manager.get_queue_depth.return_value = 5

        strategy = TestStrategy(mock_predictor, mock_registry)
        strategy.set_worker_queue_manager(mock_manager)

        # Act
        depth = strategy.get_scheduler_queue_depth("worker-1")

        # Assert
        assert depth == 5
        mock_manager.get_queue_depth.assert_called_once_with("worker-1")

    def test_get_scheduler_queue_depth_without_manager(self):
        """Test getting queue depth returns 0 when no manager set."""
        from src.algorithms.base import SchedulingStrategy

        class TestStrategy(SchedulingStrategy):
            def select_instance(self, predictions, queue_info):
                return None

            def update_queue(self, instance_id, prediction):
                pass

        # Arrange
        mock_predictor = MagicMock()
        mock_registry = MagicMock()

        strategy = TestStrategy(mock_predictor, mock_registry)

        # Act
        depth = strategy.get_scheduler_queue_depth("worker-1")

        # Assert
        assert depth == 0

    def test_get_scheduler_queue_depth_for_unknown_worker(self):
        """Test getting queue depth for unknown worker returns 0."""
        from src.algorithms.base import SchedulingStrategy

        class TestStrategy(SchedulingStrategy):
            def select_instance(self, predictions, queue_info):
                return None

            def update_queue(self, instance_id, prediction):
                pass

        # Arrange
        mock_predictor = MagicMock()
        mock_registry = MagicMock()
        mock_manager = MagicMock(spec=WorkerQueueManager)
        mock_manager.get_queue_depth.return_value = 0

        strategy = TestStrategy(mock_predictor, mock_registry)
        strategy.set_worker_queue_manager(mock_manager)

        # Act
        depth = strategy.get_scheduler_queue_depth("unknown-worker")

        # Assert
        assert depth == 0


class TestGracefulDegradation:
    """Tests for graceful degradation when queue manager not available."""

    def test_adapter_works_with_none_manager(self):
        """Test adapter handles None manager gracefully."""
        # Act & Assert - should not raise
        result = get_queue_info_from_manager(
            worker_queue_manager=None,
            instance_id="worker-1",
            avg_exec_time_ms=100.0,
        )

        assert isinstance(result, InstanceQueueExpectError)
        assert result.expected_time_ms == 0.0
        assert result.error_margin_ms == 0.0

    def test_batch_adapter_works_with_none_manager(self):
        """Test batch adapter handles None manager gracefully."""
        # Act & Assert - should not raise
        result = get_all_queue_info_from_manager(
            worker_queue_manager=None,
            instance_ids=["worker-1", "worker-2"],
            avg_exec_time_ms=100.0,
        )

        assert len(result) == 2
        assert all(v.expected_time_ms == 0.0 for v in result.values())
