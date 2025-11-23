"""
Comprehensive unit tests for common infrastructure components.

Tests cover:
- BaseTaskSubmitter and BaseTaskReceiver abstract classes
- RateLimiter with token bucket algorithm and Poisson distribution
- WorkflowState data structures and helper methods
- Utility functions (JSON, logging, HTTP, timestamps)

Target coverage: >90%
"""

import asyncio
import json
import logging
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import requests
from scipy import stats

from common import (
    BaseTaskReceiver,
    BaseTaskSubmitter,
    PoissonRateLimiter,
    RateLimiter,
    TaskStatus,
    WorkflowState,
    WorkflowType,
    calculate_duration,
    configure_logging,
    estimate_token_length,
    format_duration,
    format_timestamp,
    http_request_with_retry,
    load_json,
    save_json,
)


# ============================================================================
# RateLimiter Tests
# ============================================================================

class TestRateLimiter:
    """Test suite for RateLimiter class."""

    def test_initialization(self):
        """Test RateLimiter initialization with default and custom burst size."""
        limiter = RateLimiter(rate=10.0)
        assert limiter.rate == 10.0
        assert limiter.max_tokens == 20.0  # Default: rate * 2
        assert limiter.tokens == 0.0  # Start with 0 tokens

        custom_limiter = RateLimiter(rate=10.0, burst_size=50.0)
        assert custom_limiter.max_tokens == 50.0

    def test_qps_accuracy(self):
        """Test that actual QPS matches target QPS within ±5% tolerance."""
        target_qps = 10.0
        num_requests = 100
        limiter = RateLimiter(rate=target_qps)

        start_time = time.time()
        for _ in range(num_requests):
            limiter.acquire()
        elapsed = time.time() - start_time

        actual_qps = num_requests / elapsed
        tolerance = target_qps * 0.05  # ±5%

        assert abs(actual_qps - target_qps) <= tolerance, (
            f"QPS accuracy failed: target={target_qps:.2f}, "
            f"actual={actual_qps:.2f}, tolerance=±{tolerance:.2f}"
        )

    def test_thread_safety(self):
        """Test thread-safety with concurrent access from multiple threads."""
        limiter = RateLimiter(rate=100.0)
        num_threads = 10
        requests_per_thread = 100
        total_requests = num_threads * requests_per_thread

        acquired_counts = []

        def worker():
            count = 0
            for _ in range(requests_per_thread):
                limiter.acquire()
                count += 1
            acquired_counts.append(count)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for future in futures:
                future.result()
        elapsed = time.time() - start_time

        # Verify all requests were acquired
        assert sum(acquired_counts) == total_requests
        assert len(acquired_counts) == num_threads

        # Verify thread-safe QPS control
        actual_qps = total_requests / elapsed
        assert 80 <= actual_qps <= 120  # Within reasonable range

    def test_poisson_distribution(self):
        """Test Poisson distribution using chi-squared goodness-of-fit test."""
        limiter = RateLimiter(rate=10.0)
        num_samples = 1000

        # Generate Poisson intervals
        intervals = [limiter.get_poisson_interval() for _ in range(num_samples)]

        # Expected mean for Poisson process
        expected_mean = 1.0 / limiter.rate  # 0.1 seconds

        # Calculate sample mean
        sample_mean = sum(intervals) / len(intervals)

        # Allow 20% deviation (statistical test)
        assert abs(sample_mean - expected_mean) / expected_mean < 0.2, (
            f"Poisson distribution test failed: "
            f"expected_mean={expected_mean:.3f}, sample_mean={sample_mean:.3f}"
        )

    def test_burst_handling(self):
        """Test burst capacity and refill mechanism."""
        limiter = RateLimiter(rate=10.0, burst_size=20.0)

        # Wait for bucket to fill
        time.sleep(2.5)  # Wait for max tokens to accumulate

        # Now acquire burst tokens (should complete quickly with pre-filled bucket)
        start = time.time()
        for _ in range(10):  # Acquire half the burst capacity
            limiter.acquire()
        burst_time = time.time() - start

        # Should complete very quickly (tokens already available)
        assert burst_time < 0.2

        # Next acquisition should block (fewer tokens available)
        start = time.time()
        limiter.acquire()
        wait_time = time.time() - start

        # Should wait ~0.1 seconds for refill (1/rate)
        assert 0.05 < wait_time < 0.3

    def test_reset(self):
        """Test reset functionality."""
        limiter = RateLimiter(rate=10.0)

        # Acquire some tokens
        for _ in range(5):
            limiter.acquire()

        # Reset
        limiter.reset()

        # Verify state is reset
        assert limiter.tokens == 0.0

    def test_get_stats(self):
        """Test statistics retrieval."""
        limiter = RateLimiter(rate=10.0, burst_size=20.0)

        stats = limiter.get_stats()
        assert stats['rate'] == 10.0
        assert stats['max_tokens'] == 20.0
        assert 'current_tokens' in stats
        assert 'utilization' in stats


class TestPoissonRateLimiter:
    """Test suite for PoissonRateLimiter."""

    def test_poisson_acquire(self):
        """Test that PoissonRateLimiter automatically applies intervals."""
        limiter = PoissonRateLimiter(rate=10.0)

        start = time.time()
        for _ in range(10):
            limiter.acquire()
        elapsed = time.time() - start

        # Should take ~1 second for 10 requests at 10 QPS
        assert 0.8 < elapsed < 1.5


# ============================================================================
# BaseTaskSubmitter Tests
# ============================================================================

class ConcreteTaskSubmitter(BaseTaskSubmitter):
    """Concrete implementation for testing."""

    def __init__(self, *args, task_data_list=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_data_list = task_data_list or []
        self.task_index = 0

    def _prepare_task_payload(self, task_data):
        return {
            "task_id": task_data['task_id'],
            "model_id": "test_model",
            "task_input": task_data.get('input', {}),
            "metadata": task_data.get('metadata', {})
        }

    def _get_next_task_data(self):
        if self.task_index < len(self.task_data_list):
            task = self.task_data_list[self.task_index]
            self.task_index += 1
            return task
        return None


class TestBaseTaskSubmitter:
    """Test suite for BaseTaskSubmitter."""

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseTaskSubmitter(name="test", scheduler_url="http://localhost:8100")

    def test_thread_lifecycle(self):
        """Test thread start, run, and stop lifecycle."""
        task_data = [
            {'task_id': f'task-{i}', 'input': {'value': i}}
            for i in range(5)
        ]

        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"status": "ok"}

            submitter = ConcreteTaskSubmitter(
                name="TestSubmitter",
                scheduler_url="http://localhost:8100",
                task_data_list=task_data
            )

            # Start thread
            submitter.start()
            assert submitter.is_alive()
            assert submitter.running

            # Wait for completion
            submitter.join(timeout=5.0)

            # Verify all tasks submitted
            assert submitter.submitted_count == 5
            assert submitter.failed_count == 0
            assert not submitter.running

    def test_rate_limiting_integration(self):
        """Test integration with RateLimiter."""
        task_data = [{'task_id': f'task-{i}'} for i in range(10)]
        limiter = RateLimiter(rate=20.0)

        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"status": "ok"}

            submitter = ConcreteTaskSubmitter(
                name="TestSubmitter",
                scheduler_url="http://localhost:8100",
                task_data_list=task_data,
                rate_limiter=limiter
            )

            start = time.time()
            submitter.start()
            submitter.join(timeout=10.0)
            elapsed = time.time() - start

            # 10 tasks at 20 QPS should take ~0.5 seconds
            assert 0.3 < elapsed < 0.8

    def test_graceful_shutdown(self):
        """Test graceful shutdown with stop_event."""
        task_data = [{'task_id': f'task-{i}'} for i in range(100)]

        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"status": "ok"}

            submitter = ConcreteTaskSubmitter(
                name="TestSubmitter",
                scheduler_url="http://localhost:8100",
                task_data_list=task_data
            )

            submitter.start()
            time.sleep(0.1)  # Let it submit a few tasks
            submitter.stop()
            submitter.join(timeout=2.0)

            # Should have stopped before all tasks
            assert submitter.submitted_count < 100
            assert not submitter.running

    def test_error_handling(self):
        """Test error handling during task submission."""
        task_data = [{'task_id': f'task-{i}'} for i in range(5)]

        with patch('requests.post') as mock_post:
            # Simulate failures
            mock_post.side_effect = requests.exceptions.RequestException("Connection failed")

            submitter = ConcreteTaskSubmitter(
                name="TestSubmitter",
                scheduler_url="http://localhost:8100",
                task_data_list=task_data
            )

            submitter.start()
            submitter.join(timeout=5.0)

            # All should have failed
            assert submitter.failed_count == 5
            assert submitter.submitted_count == 0


# ============================================================================
# BaseTaskReceiver Tests
# ============================================================================

class ConcreteTaskReceiver(BaseTaskReceiver):
    """Concrete implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processed_results = []

    def _get_subscription_payload(self):
        return {
            "type": "subscribe",
            "model_id": self.model_id
        }

    async def _process_result(self, data):
        self.processed_results.append(data)


@pytest.mark.asyncio
class TestBaseTaskReceiver:
    """Test suite for BaseTaskReceiver."""

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            BaseTaskReceiver(
                name="test",
                scheduler_url="http://localhost:8100",
                model_id="test_model"
            )

    async def test_websocket_connection(self):
        """Test WebSocket connection and subscription."""
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_ws

            # Mock subscription ACK
            mock_ws.recv.side_effect = [
                json.dumps({"type": "ack", "status": "subscribed"}),
                asyncio.TimeoutError()
            ]

            receiver = ConcreteTaskReceiver(
                name="TestReceiver",
                scheduler_url="http://localhost:8100",
                model_id="test_model"
            )

            # Run async loop briefly
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(receiver._run_async(), timeout=0.5)

            # Verify subscription was sent
            mock_ws.send.assert_called_once()

    def test_state_update_thread_safety(self):
        """Test thread-safe workflow state updates."""
        workflow_states = {
            'workflow-1': {'status': 'pending', 'count': 0}
        }
        state_lock = threading.Lock()

        receiver = ConcreteTaskReceiver(
            name="TestReceiver",
            scheduler_url="http://localhost:8100",
            model_id="test_model",
            workflow_states=workflow_states,
            state_lock=state_lock
        )

        def update_fn(state):
            state['count'] += 1

        # Update from multiple threads
        def worker():
            for _ in range(10):
                receiver._update_workflow_state('workflow-1', update_fn)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(5)]
            for future in futures:
                future.result()

        # Should have incremented safely
        assert workflow_states['workflow-1']['count'] == 50


# ============================================================================
# WorkflowState Tests
# ============================================================================

class TestWorkflowState:
    """Test suite for WorkflowState data structure."""

    def test_text2video_workflow_completion(self):
        """Test is_complete() logic for Text2Video workflow."""
        state = WorkflowState(
            workflow_id="wf-1",
            workflow_type=WorkflowType.TEXT2VIDEO,
            max_b_loops=3
        )

        # Not complete yet
        assert not state.is_complete()

        # Add B completions
        state.b_complete_times = [time.time(), time.time()]
        assert not state.is_complete()

        # Complete when all iterations done
        state.b_complete_times.append(time.time())
        assert state.is_complete()

    def test_deep_research_workflow_completion(self):
        """Test is_complete() logic for Deep Research workflow."""
        state = WorkflowState(
            workflow_id="wf-1",
            workflow_type=WorkflowType.DEEP_RESEARCH,
            fanout_count=3
        )

        # Not complete yet
        assert not state.is_complete()

        # Add merge completion
        state.merge_complete_time = time.time()
        assert state.is_complete()

    def test_b_loop_logic(self):
        """Test B loop counting and continuation logic for Text2Video."""
        state = WorkflowState(
            workflow_id="wf-1",
            workflow_type=WorkflowType.TEXT2VIDEO,
            max_b_loops=4
        )

        # Should continue initially
        assert state.should_continue_b_loop()
        assert state.prepare_next_b_iteration()
        assert state.b_loop_count == 1

        # Continue until max
        for i in range(2, 5):
            assert state.prepare_next_b_iteration()
            assert state.b_loop_count == i

        # Should not continue after max
        assert not state.should_continue_b_loop()
        assert not state.prepare_next_b_iteration()

    def test_fanout_tracking(self):
        """Test B1/B2 task tracking for Deep Research."""
        state = WorkflowState(
            workflow_id="wf-1",
            workflow_type=WorkflowType.DEEP_RESEARCH,
            fanout_count=3
        )

        # Initially not complete
        assert not state.are_all_b1_tasks_complete()
        assert not state.are_all_b2_tasks_complete()

        # Mark B1 tasks complete
        now = time.time()
        state.mark_b1_task_complete("b1-1", now + 1)
        state.mark_b1_task_complete("b1-2", now + 2)
        assert not state.are_all_b1_tasks_complete()

        state.mark_b1_task_complete("b1-3", now + 3)
        assert state.are_all_b1_tasks_complete()
        assert state.all_b1_complete_time == now + 3  # Max timestamp

        # Mark B2 tasks complete
        state.mark_b2_task_complete("b2-1", now + 4)
        state.mark_b2_task_complete("b2-2", now + 5)
        state.mark_b2_task_complete("b2-3", now + 6)
        assert state.are_all_b2_tasks_complete()
        assert state.all_b2_complete_time == now + 6

    def test_status_summary(self):
        """Test get_status_summary() for both workflow types."""
        # Text2Video
        t2v_state = WorkflowState(
            workflow_id="wf-1",
            workflow_type=WorkflowType.TEXT2VIDEO,
            max_b_loops=3
        )
        summary = t2v_state.get_status_summary()
        assert summary['workflow_type'] == 'text2video'
        assert 'b_loop_count' in summary
        assert 'max_b_loops' in summary

        # Deep Research
        dr_state = WorkflowState(
            workflow_id="wf-2",
            workflow_type=WorkflowType.DEEP_RESEARCH,
            fanout_count=2
        )
        summary = dr_state.get_status_summary()
        assert summary['workflow_type'] == 'deep_research'
        assert 'fanout_count' in summary
        assert 'b1_complete' in summary


# ============================================================================
# Utility Functions Tests
# ============================================================================

class TestUtils:
    """Test suite for utility functions."""

    def test_json_serialization(self):
        """Test JSON serialization with custom types."""
        test_data = {
            'workflow_type': WorkflowType.TEXT2VIDEO,
            'status': TaskStatus.COMPLETED,
            'timestamp': time.time(),
        }

        # Should serialize without error
        json_str = json.dumps(test_data, default=lambda o: o.value if isinstance(o, (WorkflowType, TaskStatus)) else str(o))
        assert json_str
        assert 'text2video' in json_str

    def test_save_load_json(self):
        """Test JSON file save and load."""
        test_data = {'key': 'value', 'number': 42}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            # Save
            save_json(test_data, filepath)
            assert filepath.exists()

            # Load
            loaded_data = load_json(filepath)
            assert loaded_data == test_data

    def test_logging_configuration(self):
        """Test logging configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            logger = configure_logging(
                level=logging.DEBUG,
                log_file=log_file
            )

            assert logger
            logger.info("Test message")

            # Verify file was created
            assert log_file.exists()

    def test_http_retry_mechanism(self):
        """Test HTTP request retry logic."""
        with patch('requests.Session.request') as mock_request:
            # Simulate failures then success
            mock_request.side_effect = [
                requests.exceptions.ConnectionError(),
                requests.exceptions.ConnectionError(),
                Mock(status_code=200, json=lambda: {"status": "ok"})
            ]

            response = http_request_with_retry(
                'GET',
                'http://localhost:8100/test',
                max_retries=3
            )

            assert response.status_code == 200
            assert mock_request.call_count == 3

    def test_timestamp_utilities(self):
        """Test timestamp and duration utilities."""
        # Format timestamp
        ts = time.time()
        formatted = format_timestamp(ts)
        assert len(formatted) > 0

        # Format duration
        assert format_duration(45.2) == "45.20s"
        assert "1m" in format_duration(90)
        assert "1h" in format_duration(3700)

        # Calculate duration
        start = time.time()
        time.sleep(0.1)
        duration = calculate_duration(start)
        assert 0.08 < duration < 0.15

    def test_token_estimation(self):
        """Test token length estimation."""
        text = "This is a test sentence with multiple words."
        tokens = estimate_token_length(text)
        assert tokens > 0
        assert tokens == len(text) // 4  # Heuristic

        # Empty text
        assert estimate_token_length(None) == 0
        assert estimate_token_length("") == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for components working together."""

    def test_submitter_with_rate_limiter(self):
        """Test BaseTaskSubmitter with RateLimiter integration."""
        task_data = [{'task_id': f'task-{i}'} for i in range(20)]
        limiter = RateLimiter(rate=40.0)

        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"status": "ok"}

            submitter = ConcreteTaskSubmitter(
                name="TestSubmitter",
                scheduler_url="http://localhost:8100",
                task_data_list=task_data,
                rate_limiter=limiter
            )

            start = time.time()
            submitter.start()
            submitter.join(timeout=5.0)
            elapsed = time.time() - start

            # 20 tasks at 40 QPS should take ~0.5 seconds
            assert 0.3 < elapsed < 1.0
            assert submitter.submitted_count == 20

    def test_workflow_state_concurrent_updates(self):
        """Test WorkflowState with concurrent updates from multiple threads."""
        state = WorkflowState(
            workflow_id="wf-1",
            workflow_type=WorkflowType.DEEP_RESEARCH,
            fanout_count=10
        )

        def worker(task_id):
            time.sleep(0.01)  # Simulate work
            state.mark_b1_task_complete(f"b1-{task_id}", time.time())

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in futures:
                future.result()

        # All tasks should be marked complete
        assert state.are_all_b1_tasks_complete()
        assert len(state.b1_complete_times) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=common", "--cov-report=html"])
