"""Unit tests for PlannerReporter throughput reporting.

Tests throughput data submission to planner's /submit_throughput endpoint.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from swarmpilot.scheduler.utils.planner_reporter import PlannerReporter


class TestPlannerReporterThroughputIntegration:
    """Tests for throughput reporting in PlannerReporter."""

    @pytest.fixture
    def mock_task_registry(self):
        """Create a mock task registry."""
        registry = MagicMock()
        registry.get_count_by_status = AsyncMock(return_value=0)
        return registry

    @pytest.fixture
    def mock_throughput_tracker(self):
        """Create a mock throughput tracker."""
        tracker = MagicMock()
        tracker.get_averages_for_recent_instances_seconds = AsyncMock(
            return_value={
                "http://inst1:8080": 0.5,
                "http://inst2:8080": 1.0,
            }
        )
        return tracker

    @pytest.fixture
    def reporter_with_tracker(self, mock_task_registry, mock_throughput_tracker):
        """Create PlannerReporter with throughput tracker."""
        reporter = PlannerReporter(
            task_registry=mock_task_registry,
            planner_url="http://planner:8000",
            interval=10.0,
            throughput_tracker=mock_throughput_tracker,
        )
        reporter.set_model_id("test_model")
        return reporter

    @pytest.fixture
    def reporter_without_tracker(self, mock_task_registry):
        """Create PlannerReporter without throughput tracker."""
        reporter = PlannerReporter(
            task_registry=mock_task_registry,
            planner_url="http://planner:8000",
            interval=10.0,
            throughput_tracker=None,
        )
        reporter.set_model_id("test_model")
        return reporter

    @pytest.mark.asyncio
    async def test_reports_throughput_when_tracker_present(
        self, reporter_with_tracker, mock_throughput_tracker
    ):
        """When tracker is present, throughput data is sent to /submit_throughput."""
        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        reporter_with_tracker._http_client = MagicMock()
        reporter_with_tracker._http_client.post = AsyncMock(return_value=mock_response)

        # Call _report_to_planner
        await reporter_with_tracker._report_to_planner()

        # Verify throughput tracker was queried
        mock_throughput_tracker.get_averages_for_recent_instances_seconds.assert_called_once()

        # Verify /submit_throughput calls were made for each instance
        calls = reporter_with_tracker._http_client.post.call_args_list

        # Should have 3 calls: 1 for submit_target + 2 for submit_throughput
        assert len(calls) == 3

        # Check submit_target call
        submit_target_call = calls[0]
        assert "/v1/submit_target" in str(submit_target_call)

        # Check submit_throughput calls
        throughput_calls = [c for c in calls if "/v1/submit_throughput" in str(c)]
        assert len(throughput_calls) == 2

    @pytest.mark.asyncio
    async def test_no_throughput_calls_when_tracker_none(
        self, reporter_without_tracker
    ):
        """When tracker is None, no /submit_throughput calls are made."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        reporter_without_tracker._http_client = MagicMock()
        reporter_without_tracker._http_client.post = AsyncMock(
            return_value=mock_response
        )

        await reporter_without_tracker._report_to_planner()

        # Should only have 1 call (submit_target), no submit_throughput
        calls = reporter_without_tracker._http_client.post.call_args_list
        assert len(calls) == 1
        assert "/v1/submit_target" in str(calls[0])

    @pytest.mark.asyncio
    async def test_throughput_call_includes_correct_payload(
        self, reporter_with_tracker, mock_throughput_tracker
    ):
        """Verify /submit_throughput payload has correct structure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        reporter_with_tracker._http_client = MagicMock()
        reporter_with_tracker._http_client.post = AsyncMock(return_value=mock_response)

        await reporter_with_tracker._report_to_planner()

        calls = reporter_with_tracker._http_client.post.call_args_list

        # Find submit_throughput calls
        for call in calls:
            url = call[0][0] if call[0] else call.kwargs.get("url", "")
            if "/v1/submit_throughput" in str(url):
                json_data = call.kwargs.get("json", call[1].get("json", {}))
                assert "instance_url" in json_data
                assert "avg_execution_time" in json_data
                assert json_data["avg_execution_time"] > 0

    @pytest.mark.asyncio
    async def test_continues_on_throughput_http_error(
        self, reporter_with_tracker, mock_throughput_tracker
    ):
        """HTTP errors on /submit_throughput don't crash the reporter."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "/v1/submit_throughput" in str(args) or "/v1/submit_throughput" in str(kwargs):
                raise httpx.HTTPError("Connection refused")
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        reporter_with_tracker._http_client = MagicMock()
        reporter_with_tracker._http_client.post = AsyncMock(side_effect=side_effect)

        # Should not raise even with HTTP errors
        await reporter_with_tracker._report_to_planner()

        # Verify attempts were made
        assert call_count > 0

    @pytest.mark.asyncio
    async def test_empty_throughput_data_no_calls(
        self, reporter_with_tracker, mock_throughput_tracker
    ):
        """When tracker returns empty data, no /submit_throughput calls are made."""
        mock_throughput_tracker.get_averages_for_recent_instances_seconds = AsyncMock(
            return_value={}
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        reporter_with_tracker._http_client = MagicMock()
        reporter_with_tracker._http_client.post = AsyncMock(return_value=mock_response)

        await reporter_with_tracker._report_to_planner()

        calls = reporter_with_tracker._http_client.post.call_args_list

        # Should only have submit_target call, no submit_throughput
        assert len(calls) == 1
        assert "/v1/submit_target" in str(calls[0])

    @pytest.mark.asyncio
    async def test_throughput_reported_for_each_instance(
        self, reporter_with_tracker, mock_throughput_tracker
    ):
        """Each instance's throughput is reported separately."""
        # Set up 3 instances
        mock_throughput_tracker.get_averages_for_recent_instances_seconds = AsyncMock(
            return_value={
                "http://inst1:8080": 0.5,
                "http://inst2:8080": 1.0,
                "http://inst3:8080": 2.0,
            }
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        reporter_with_tracker._http_client = MagicMock()
        reporter_with_tracker._http_client.post = AsyncMock(return_value=mock_response)

        await reporter_with_tracker._report_to_planner()

        calls = reporter_with_tracker._http_client.post.call_args_list

        # 1 submit_target + 3 submit_throughput = 4 total
        assert len(calls) == 4

        throughput_calls = [c for c in calls if "/v1/submit_throughput" in str(c)]
        assert len(throughput_calls) == 3


class TestPlannerReporterThroughputInit:
    """Tests for PlannerReporter initialization with throughput tracker."""

    def test_accepts_throughput_tracker_parameter(self):
        """PlannerReporter accepts throughput_tracker in constructor."""
        mock_registry = MagicMock()
        mock_tracker = MagicMock()

        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=10.0,
            throughput_tracker=mock_tracker,
        )

        assert reporter._throughput_tracker is mock_tracker

    def test_throughput_tracker_defaults_to_none(self):
        """PlannerReporter defaults throughput_tracker to None."""
        mock_registry = MagicMock()

        reporter = PlannerReporter(
            task_registry=mock_registry,
            planner_url="http://planner:8000",
            interval=10.0,
        )

        assert reporter._throughput_tracker is None
