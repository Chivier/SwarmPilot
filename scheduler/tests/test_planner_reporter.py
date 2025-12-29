"""Comprehensive unit tests for PlannerReporter.

Tests the background task that reports uncompleted task counts to planner.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.model import TaskStatus
from src.planner_reporter import PlannerReporter


class TestPlannerReporterInit:
    """Tests for PlannerReporter initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        mock_registry = MagicMock()
        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=10.0,
            timeout=5.0,
        )

        assert reporter._planner_url == "http://planner:8000"
        assert reporter._interval == 10.0
        assert reporter._timeout == 5.0
        assert reporter._model_id is None
        assert reporter._reporter_task is None
        assert reporter._shutdown is False
        assert reporter._throughput_tracker is None

    def test_initialization_with_throughput_tracker(self):
        """Test initialization with throughput tracker."""
        mock_registry = MagicMock()
        mock_tracker = MagicMock()

        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=10.0,
            throughput_tracker=mock_tracker,
        )

        assert reporter._throughput_tracker is mock_tracker


class TestPlannerReporterSetModelId:
    """Tests for set_model_id method."""

    def test_set_model_id_first_time(self):
        """Test setting model ID for the first time."""
        reporter = PlannerReporter(
            task_registry=MagicMock(),
            planner_url="http://planner:8000",
            interval=10.0,
        )

        reporter.set_model_id("test_model")

        assert reporter._model_id == "test_model"

    def test_set_model_id_already_set_ignores(self):
        """Test that setting model ID when already set is ignored."""
        reporter = PlannerReporter(
            task_registry=MagicMock(),
            planner_url="http://planner:8000",
            interval=10.0,
        )

        reporter.set_model_id("first_model")
        reporter.set_model_id("second_model")

        # Should still be first model
        assert reporter._model_id == "first_model"


class TestPlannerReporterStart:
    """Tests for start method."""

    @pytest.fixture
    def reporter(self):
        """Create a PlannerReporter instance."""
        return PlannerReporter(
            task_registry=MagicMock(),
            planner_url="http://planner:8000",
            interval=0.1,  # Short interval for testing
        )

    @pytest.mark.asyncio
    async def test_start_creates_task(self, reporter):
        """Test start creates reporter task."""
        await reporter.start()

        assert reporter._reporter_task is not None
        assert reporter._shutdown is False

        # Cleanup
        await reporter.shutdown()

    @pytest.mark.asyncio
    async def test_start_already_running_warns(self, reporter):
        """Test start when already running logs warning."""
        await reporter.start()

        with patch("src.planner_reporter.logger") as mock_logger:
            await reporter.start()
            mock_logger.warning.assert_called_with(
                "Planner reporter already running"
            )

        # Cleanup
        await reporter.shutdown()


class TestPlannerReporterShutdown:
    """Tests for shutdown method."""

    @pytest.fixture
    def reporter(self):
        return PlannerReporter(
            task_registry=MagicMock(),
            planner_url="http://planner:8000",
            interval=0.1,
        )

    @pytest.mark.asyncio
    async def test_shutdown_graceful(self, reporter):
        """Test graceful shutdown."""
        await reporter.start()
        await reporter.shutdown()

        assert reporter._shutdown is True
        assert reporter._reporter_task is None

    @pytest.mark.asyncio
    async def test_shutdown_without_start(self, reporter):
        """Test shutdown when never started."""
        await reporter.shutdown()

        assert reporter._shutdown is True
        assert reporter._reporter_task is None

    @pytest.mark.asyncio
    async def test_shutdown_cancels_task(self, reporter):
        """Test shutdown cancels running task."""
        await reporter.start()

        # Wait for task to be running
        await asyncio.sleep(0.05)

        await reporter.shutdown()

        assert reporter._reporter_task is None


class TestPlannerReporterReportLoop:
    """Tests for _report_loop method."""

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.get_count_by_status = AsyncMock(return_value=0)
        return registry

    @pytest.fixture
    def reporter(self, mock_registry):
        return PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=0.05,  # Very short interval
        )

    @pytest.mark.asyncio
    async def test_report_loop_skips_without_model_id(self, reporter):
        """Test report loop skips reporting when model_id not set."""
        with patch.object(reporter, "_report_to_planner") as mock_report:
            await reporter.start()
            await asyncio.sleep(0.15)  # Let it run a few cycles
            await reporter.shutdown()

            # Should not have called _report_to_planner
            mock_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_report_loop_reports_with_model_id(self, reporter):
        """Test report loop calls _report_to_planner when model_id is set."""
        reporter.set_model_id("test_model")

        with patch.object(
            reporter, "_report_to_planner", new_callable=AsyncMock
        ) as mock_report:
            await reporter.start()
            await asyncio.sleep(0.12)  # Let it run a cycle
            await reporter.shutdown()

            # Should have called _report_to_planner at least once
            assert mock_report.call_count >= 1

    @pytest.mark.asyncio
    async def test_report_loop_continues_on_exception(self, reporter):
        """Test report loop continues after exception."""
        reporter.set_model_id("test_model")

        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test error")
            # Second call succeeds silently

        with patch.object(
            reporter, "_report_to_planner", side_effect=side_effect
        ):
            await reporter.start()
            await asyncio.sleep(0.15)  # Let it run a couple cycles
            await reporter.shutdown()

        # Should have continued after error
        assert call_count >= 2


class TestPlannerReporterReportToPlanner:
    """Tests for _report_to_planner method."""

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.get_count_by_status = AsyncMock(return_value=5)
        return registry

    @pytest.fixture
    def reporter(self, mock_registry):
        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=10.0,
        )
        reporter.set_model_id("test_model")
        return reporter

    @pytest.mark.asyncio
    async def test_report_to_planner_success(self, reporter, mock_registry):
        """Test successful report to planner."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        reporter._http_client = MagicMock()
        reporter._http_client.post = AsyncMock(return_value=mock_response)

        await reporter._report_to_planner()

        # Verify HTTP call
        reporter._http_client.post.assert_called_once()
        call_args = reporter._http_client.post.call_args

        assert "/submit_target" in call_args[0][0]
        assert call_args[1]["json"]["model_id"] == "test_model"
        assert call_args[1]["json"]["value"] == 10.0  # 5 pending + 5 running

    @pytest.mark.asyncio
    async def test_report_to_planner_http_status_error(
        self, reporter, mock_registry
    ):
        """Test handling of HTTP status error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        error = httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=mock_response,
        )

        reporter._http_client = MagicMock()
        reporter._http_client.post = AsyncMock(side_effect=error)

        # Should not raise
        with patch("src.planner_reporter.log_http_error"):
            await reporter._report_to_planner()

    @pytest.mark.asyncio
    async def test_report_to_planner_connection_error(self, reporter):
        """Test handling of connection error."""
        reporter._http_client = MagicMock()
        reporter._http_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        # Should not raise
        with patch("src.planner_reporter.log_http_error"):
            await reporter._report_to_planner()

    @pytest.mark.asyncio
    async def test_report_to_planner_unexpected_error(self, reporter):
        """Test handling of unexpected error."""
        reporter._http_client = MagicMock()
        reporter._http_client.post = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        # Should not raise
        await reporter._report_to_planner()

    @pytest.mark.asyncio
    async def test_report_to_planner_calls_report_throughput(self, reporter):
        """Test that _report_to_planner calls _report_throughput."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        reporter._http_client = MagicMock()
        reporter._http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(
            reporter, "_report_throughput", new_callable=AsyncMock
        ) as mock_throughput:
            await reporter._report_to_planner()
            mock_throughput.assert_called_once()


class TestPlannerReporterReportThroughput:
    """Tests for _report_throughput method."""

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.get_count_by_status = AsyncMock(return_value=0)
        return registry

    @pytest.fixture
    def mock_tracker(self):
        tracker = MagicMock()
        tracker.get_averages_for_recent_instances_seconds = AsyncMock(
            return_value={
                "http://inst1:8080": 0.5,
                "http://inst2:8080": 1.0,
            }
        )
        return tracker

    @pytest.fixture
    def reporter(self, mock_registry, mock_tracker):
        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=10.0,
            throughput_tracker=mock_tracker,
        )
        reporter.set_model_id("test_model")
        return reporter

    @pytest.mark.asyncio
    async def test_report_throughput_no_tracker(self, mock_registry):
        """Test _report_throughput returns early when no tracker."""
        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=10.0,
            throughput_tracker=None,  # No tracker
        )

        # Should return without doing anything
        await reporter._report_throughput()

    @pytest.mark.asyncio
    async def test_report_throughput_empty_averages(self, mock_registry):
        """Test _report_throughput with empty averages."""
        mock_tracker = MagicMock()
        mock_tracker.get_averages_for_recent_instances_seconds = AsyncMock(
            return_value={}
        )

        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=10.0,
            throughput_tracker=mock_tracker,
        )

        reporter._http_client = MagicMock()
        reporter._http_client.post = AsyncMock()

        await reporter._report_throughput()

        # Should not have called post
        reporter._http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_report_throughput_success(self, reporter, mock_tracker):
        """Test successful throughput reporting."""
        mock_response = MagicMock()
        reporter._http_client = MagicMock()
        reporter._http_client.post = AsyncMock(return_value=mock_response)

        await reporter._report_throughput()

        # Should have called post twice (once for each instance)
        assert reporter._http_client.post.call_count == 2

        # Verify call args
        calls = reporter._http_client.post.call_args_list
        urls = [c[0][0] for c in calls]
        assert all("/submit_throughput" in url for url in urls)

    @pytest.mark.asyncio
    async def test_report_throughput_http_error_continues(
        self, reporter, mock_tracker
    ):
        """Test throughput reporting continues after HTTP error."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            return MagicMock()

        reporter._http_client = MagicMock()
        reporter._http_client.post = AsyncMock(side_effect=side_effect)

        await reporter._report_throughput()

        # Should have tried both instances
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_report_throughput_exception_handling(self, mock_registry):
        """Test throughput reporting handles exceptions."""
        mock_tracker = MagicMock()
        mock_tracker.get_averages_for_recent_instances_seconds = AsyncMock(
            side_effect=Exception("Tracker error")
        )

        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=10.0,
            throughput_tracker=mock_tracker,
        )

        # Should not raise
        await reporter._report_throughput()


class TestPlannerReporterIntegration:
    """Integration tests for PlannerReporter."""

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.get_count_by_status = AsyncMock(
            side_effect=lambda status: {
                TaskStatus.PENDING: 3,
                TaskStatus.RUNNING: 2,
            }.get(status, 0)
        )
        return registry

    @pytest.mark.asyncio
    async def test_full_reporting_cycle(self, mock_registry):
        """Test full reporting cycle from start to shutdown."""
        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=0.05,
        )
        reporter.set_model_id("test_model")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        reporter._http_client = MagicMock()
        reporter._http_client.post = AsyncMock(return_value=mock_response)
        reporter._http_client.aclose = AsyncMock()

        await reporter.start()
        await asyncio.sleep(0.12)  # Let it run a couple cycles
        await reporter.shutdown()

        # Should have made at least one report
        assert reporter._http_client.post.call_count >= 1

        # Verify the reported value
        call_args = reporter._http_client.post.call_args
        assert call_args[1]["json"]["value"] == 5.0  # 3 pending + 2 running

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_registry):
        """Test reporter handles concurrent start/shutdown."""
        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=0.05,
        )

        # Multiple starts
        await reporter.start()
        await reporter.start()  # Should warn

        await asyncio.sleep(0.05)

        # Multiple shutdowns
        await reporter.shutdown()
        await reporter.shutdown()  # Should be safe
