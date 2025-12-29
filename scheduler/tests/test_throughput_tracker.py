"""Unit tests for ThroughputTracker.

Tests sliding window behavior, average calculations, and thread safety.
"""


import pytest

from src.throughput_tracker import InstanceThroughputData, ThroughputTracker


class TestInstanceThroughputData:
    """Tests for InstanceThroughputData dataclass."""

    def test_add_execution_time(self):
        """Test adding execution times to the window."""
        data = InstanceThroughputData(
            instance_endpoint="http://test:8001", window_size=3
        )

        data.add_execution_time(100.0)
        data.add_execution_time(200.0)

        assert data.get_sample_count() == 2
        assert data.get_average_ms() == 150.0

    def test_sliding_window_overflow(self):
        """Test that old values are dropped when window is full."""
        data = InstanceThroughputData(
            instance_endpoint="http://test:8001", window_size=3
        )

        data.add_execution_time(100.0)
        data.add_execution_time(200.0)
        data.add_execution_time(300.0)
        data.add_execution_time(400.0)  # Should drop 100.0

        assert data.get_sample_count() == 3
        assert data.get_average_ms() == 300.0  # (200 + 300 + 400) / 3

    def test_empty_window_returns_none(self):
        """Test that empty window returns None for average."""
        data = InstanceThroughputData(
            instance_endpoint="http://test:8001", window_size=3
        )

        assert data.get_average_ms() is None
        assert data.get_sample_count() == 0

    def test_single_value(self):
        """Test average with single value."""
        data = InstanceThroughputData(
            instance_endpoint="http://test:8001", window_size=3
        )

        data.add_execution_time(500.0)

        assert data.get_sample_count() == 1
        assert data.get_average_ms() == 500.0


class TestThroughputTracker:
    """Tests for ThroughputTracker class."""

    @pytest.mark.asyncio
    async def test_record_execution_time(self):
        """Test recording execution times."""
        tracker = ThroughputTracker(window_size=5)

        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.record_execution_time("http://test:8001", 200.0)

        avg = await tracker.get_average_execution_time_seconds(
            "http://test:8001"
        )
        assert avg == 0.15  # 150ms = 0.15s

    @pytest.mark.asyncio
    async def test_multiple_instances(self):
        """Test tracking multiple instances independently."""
        tracker = ThroughputTracker(window_size=5)

        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.record_execution_time("http://test:8002", 200.0)

        avg1 = await tracker.get_average_execution_time_seconds(
            "http://test:8001"
        )
        avg2 = await tracker.get_average_execution_time_seconds(
            "http://test:8002"
        )

        assert avg1 == 0.1  # 100ms
        assert avg2 == 0.2  # 200ms

    @pytest.mark.asyncio
    async def test_get_all_averages_seconds(self):
        """Test getting all averages at once."""
        tracker = ThroughputTracker(window_size=5)

        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.record_execution_time("http://test:8002", 200.0)

        averages = await tracker.get_all_averages_seconds()

        assert len(averages) == 2
        assert averages["http://test:8001"] == 0.1
        assert averages["http://test:8002"] == 0.2

    @pytest.mark.asyncio
    async def test_unknown_instance_returns_none(self):
        """Test that unknown instance returns None."""
        tracker = ThroughputTracker(window_size=5)

        avg = await tracker.get_average_execution_time_seconds(
            "http://unknown:8001"
        )
        assert avg is None

    @pytest.mark.asyncio
    async def test_remove_instance(self):
        """Test removing instance data."""
        tracker = ThroughputTracker(window_size=5)

        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.remove_instance("http://test:8001")

        avg = await tracker.get_average_execution_time_seconds(
            "http://test:8001"
        )
        assert avg is None

    @pytest.mark.asyncio
    async def test_clear_all(self):
        """Test clearing all data."""
        tracker = ThroughputTracker(window_size=5)

        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.record_execution_time("http://test:8002", 200.0)

        count = await tracker.clear_all()

        assert count == 2
        averages = await tracker.get_all_averages_seconds()
        assert len(averages) == 0

    @pytest.mark.asyncio
    async def test_window_size_configuration(self):
        """Test custom window size."""
        tracker = ThroughputTracker(window_size=2)

        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.record_execution_time("http://test:8001", 200.0)
        await tracker.record_execution_time(
            "http://test:8001", 300.0
        )  # Drops 100

        avg = await tracker.get_average_execution_time_seconds(
            "http://test:8001"
        )
        assert avg == 0.25  # (200 + 300) / 2 = 250ms = 0.25s

    @pytest.mark.asyncio
    async def test_empty_tracker_returns_empty_dict(self):
        """Test that empty tracker returns empty dict for get_all_averages."""
        tracker = ThroughputTracker(window_size=5)

        averages = await tracker.get_all_averages_seconds()
        assert averages == {}

    @pytest.mark.asyncio
    async def test_multiple_records_same_instance(self):
        """Test recording multiple times for the same instance."""
        tracker = ThroughputTracker(window_size=10)

        for i in range(5):
            await tracker.record_execution_time(
                "http://test:8001", 100.0 * (i + 1)
            )

        # Average of 100, 200, 300, 400, 500 = 300ms
        avg = await tracker.get_average_execution_time_seconds(
            "http://test:8001"
        )
        assert avg == 0.3  # 300ms = 0.3s

    @pytest.mark.asyncio
    async def test_conversion_to_seconds(self):
        """Test that milliseconds are correctly converted to seconds."""
        tracker = ThroughputTracker(window_size=5)

        await tracker.record_execution_time(
            "http://test:8001", 1000.0
        )  # 1000ms = 1s

        avg = await tracker.get_average_execution_time_seconds(
            "http://test:8001"
        )
        assert avg == 1.0

        await tracker.record_execution_time("http://test:8001", 500.0)  # 500ms
        # Average: (1000 + 500) / 2 = 750ms = 0.75s
        avg = await tracker.get_average_execution_time_seconds(
            "http://test:8001"
        )
        assert avg == 0.75

    @pytest.mark.asyncio
    async def test_get_averages_for_recent_instances_returns_only_new_data(
        self,
    ):
        """Test that get_averages_for_recent_instances_seconds returns only instances with new data."""
        tracker = ThroughputTracker(window_size=5)

        # Record data for two instances
        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.record_execution_time("http://test:8002", 200.0)

        # First call should return both
        averages = await tracker.get_averages_for_recent_instances_seconds()
        assert len(averages) == 2
        assert averages["http://test:8001"] == 0.1
        assert averages["http://test:8002"] == 0.2

        # Second call (no new data) should return empty
        averages = await tracker.get_averages_for_recent_instances_seconds()
        assert len(averages) == 0

    @pytest.mark.asyncio
    async def test_get_averages_for_recent_instances_clears_tracking(self):
        """Test that calling get_averages_for_recent_instances_seconds clears the tracking."""
        tracker = ThroughputTracker(window_size=5)

        await tracker.record_execution_time("http://test:8001", 100.0)

        # First call
        averages = await tracker.get_averages_for_recent_instances_seconds()
        assert len(averages) == 1

        # Record new data for a different instance
        await tracker.record_execution_time("http://test:8002", 200.0)

        # Second call should only have the new instance
        averages = await tracker.get_averages_for_recent_instances_seconds()
        assert len(averages) == 1
        assert "http://test:8002" in averages
        assert "http://test:8001" not in averages

    @pytest.mark.asyncio
    async def test_get_averages_for_recent_instances_uses_full_window(self):
        """Test that recent instances report uses the full sliding window average."""
        tracker = ThroughputTracker(window_size=5)

        # Record multiple data points for an instance
        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.record_execution_time("http://test:8001", 200.0)
        await tracker.record_execution_time("http://test:8001", 300.0)

        # Clear the tracking
        await tracker.get_averages_for_recent_instances_seconds()

        # Record one more data point
        await tracker.record_execution_time("http://test:8001", 400.0)

        # Should return the average of ALL data in the window, not just the new one
        averages = await tracker.get_averages_for_recent_instances_seconds()
        assert len(averages) == 1
        # Average: (100 + 200 + 300 + 400) / 4 = 250ms = 0.25s
        assert averages["http://test:8001"] == 0.25

    @pytest.mark.asyncio
    async def test_remove_instance_clears_new_data_tracking(self):
        """Test that removing an instance also clears it from new data tracking."""
        tracker = ThroughputTracker(window_size=5)

        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.remove_instance("http://test:8001")

        # Should return empty since instance was removed
        averages = await tracker.get_averages_for_recent_instances_seconds()
        assert len(averages) == 0

    @pytest.mark.asyncio
    async def test_clear_all_clears_new_data_tracking(self):
        """Test that clear_all also clears new data tracking."""
        tracker = ThroughputTracker(window_size=5)

        await tracker.record_execution_time("http://test:8001", 100.0)
        await tracker.record_execution_time("http://test:8002", 200.0)
        await tracker.clear_all()

        # Should return empty since all data was cleared
        averages = await tracker.get_averages_for_recent_instances_seconds()
        assert len(averages) == 0

    @pytest.mark.asyncio
    async def test_instance_with_empty_window_returns_none(self):
        """Test edge case where instance exists but has empty window (line 96)."""
        tracker = ThroughputTracker(window_size=5)

        # Create instance data directly with empty window (bypassing normal API)
        # This simulates a theoretical edge case for defensive code coverage
        async with tracker._lock:
            tracker._instances["http://test:8001"] = InstanceThroughputData(
                instance_endpoint="http://test:8001", window_size=5
            )
            # Don't add any execution times - window is empty

        # Should return None since window is empty
        avg = await tracker.get_average_execution_time_seconds(
            "http://test:8001"
        )
        assert avg is None
