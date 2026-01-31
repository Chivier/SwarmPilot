"""Tests for instance timeline tracker."""

import math
import tempfile
import threading
import os
import pytest
from pathlib import Path

from swarmpilot.planner.instance_timeline_tracker import (
    InstanceTimelineTracker,
    TimelineEntry,
    compute_instance_counts,
)


class TestInstanceTimelineTracker:
    """Tests for InstanceTimelineTracker class."""

    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary file path for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test_timeline.json")

    def test_record_migration_basic(self, temp_output_path):
        """Test basic migration recording."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        tracker.record_migration(
            event_type="deploy_migration",
            instance_counts={"model_a": 2, "model_b": 1},
            changes_count=3,
            success=True,
            target_distribution=[0.5, 0.5],
            score=0.95
        )

        entries = tracker.get_entries()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "deploy_migration"
        assert entries[0]["instance_counts"] == {"model_a": 2, "model_b": 1}
        assert entries[0]["changes_count"] == 3
        assert entries[0]["success"] is True
        assert entries[0]["total_instances"] == 3

    def test_record_migration_with_inf_score(self, temp_output_path):
        """Test that inf score is converted to None for JSON serialization."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        tracker.record_migration(
            event_type="auto_optimize",
            instance_counts={"model_a": 1},
            changes_count=1,
            success=True,
            score=float('inf')
        )

        entries = tracker.get_entries()
        assert len(entries) == 1
        assert entries[0]["score"] is None

    def test_record_migration_with_negative_inf_score(self, temp_output_path):
        """Test that -inf score is converted to None for JSON serialization."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        tracker.record_migration(
            event_type="auto_optimize",
            instance_counts={"model_a": 1},
            changes_count=1,
            success=True,
            score=float('-inf')
        )

        entries = tracker.get_entries()
        assert len(entries) == 1
        assert entries[0]["score"] is None

    def test_record_migration_with_nan_score(self, temp_output_path):
        """Test that nan score is converted to None for JSON serialization."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        tracker.record_migration(
            event_type="auto_optimize",
            instance_counts={"model_a": 1},
            changes_count=1,
            success=True,
            score=float('nan')
        )

        entries = tracker.get_entries()
        assert len(entries) == 1
        assert entries[0]["score"] is None

    def test_record_migration_with_none_score(self, temp_output_path):
        """Test that None score is preserved."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        tracker.record_migration(
            event_type="auto_optimize",
            instance_counts={"model_a": 1},
            changes_count=1,
            success=True,
            score=None
        )

        entries = tracker.get_entries()
        assert len(entries) == 1
        assert entries[0]["score"] is None

    def test_clear_entries(self, temp_output_path):
        """Test clearing timeline entries."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        # Add some entries
        tracker.record_migration(
            event_type="deploy_migration",
            instance_counts={"model_a": 2},
            changes_count=1,
            success=True
        )
        tracker.record_migration(
            event_type="auto_optimize",
            instance_counts={"model_b": 3},
            changes_count=2,
            success=True
        )

        assert tracker.get_entry_count() == 2

        # Clear entries
        tracker.clear()

        assert tracker.get_entry_count() == 0
        assert tracker.get_entries() == []

    def test_get_entry_count(self, temp_output_path):
        """Test getting entry count."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        assert tracker.get_entry_count() == 0

        tracker.record_migration(
            event_type="deploy_migration",
            instance_counts={"model_a": 1},
            changes_count=1,
            success=True
        )

        assert tracker.get_entry_count() == 1

        tracker.record_migration(
            event_type="auto_optimize",
            instance_counts={"model_b": 2},
            changes_count=1,
            success=True
        )

        assert tracker.get_entry_count() == 2

    def test_get_entries_returns_copies(self, temp_output_path):
        """Test that get_entries returns copies of entries."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        tracker.record_migration(
            event_type="deploy_migration",
            instance_counts={"model_a": 1},
            changes_count=1,
            success=True
        )

        entries1 = tracker.get_entries()
        entries2 = tracker.get_entries()

        # Modify first list
        entries1[0]["changes_count"] = 999

        # Second list should be unaffected
        assert entries2[0]["changes_count"] == 1

    def test_persistence_to_file(self, temp_output_path):
        """Test that entries are persisted to JSON file."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        tracker.record_migration(
            event_type="deploy_migration",
            instance_counts={"model_a": 2},
            changes_count=1,
            success=True
        )

        # Verify file exists and contains data
        assert Path(temp_output_path).exists()

        import json
        with open(temp_output_path) as f:
            data = json.load(f)

        assert "entries" in data
        assert len(data["entries"]) == 1
        assert data["entries"][0]["event_type"] == "deploy_migration"

    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "nested", "dir", "timeline.json")
            tracker = InstanceTimelineTracker(output_path=nested_path)

            # Verify directory was created
            assert Path(nested_path).parent.exists()

    def test_thread_safety_concurrent_writes(self, temp_output_path):
        """Test thread safety with concurrent writes."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)
        num_threads = 10
        writes_per_thread = 5

        def write_entries(thread_id):
            for i in range(writes_per_thread):
                tracker.record_migration(
                    event_type=f"thread_{thread_id}",
                    instance_counts={"model": thread_id},
                    changes_count=i,
                    success=True
                )

        threads = [threading.Thread(target=write_entries, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All entries should be recorded
        assert tracker.get_entry_count() == num_threads * writes_per_thread

    def test_thread_safety_concurrent_reads(self, temp_output_path):
        """Test thread safety with concurrent reads."""
        tracker = InstanceTimelineTracker(output_path=temp_output_path)

        # Add some entries
        for i in range(10):
            tracker.record_migration(
                event_type="test",
                instance_counts={"model": i},
                changes_count=i,
                success=True
            )

        results = []

        def read_entries():
            entries = tracker.get_entries()
            results.append(len(entries))

        threads = [threading.Thread(target=read_entries) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should return 10 entries
        assert all(r == 10 for r in results)


class TestComputeInstanceCounts:
    """Tests for compute_instance_counts function."""

    def test_compute_basic(self):
        """Test basic instance count computation."""
        class MockInstance:
            def __init__(self, model):
                self.current_model = model

        instances = [
            MockInstance("model_a"),
            MockInstance("model_a"),
            MockInstance("model_b"),
        ]

        counts = compute_instance_counts(instances)
        assert counts == {"model_a": 2, "model_b": 1}

    def test_compute_empty(self):
        """Test with empty instance list."""
        counts = compute_instance_counts([])
        assert counts == {}

    def test_compute_single_model(self):
        """Test with all instances running same model."""
        class MockInstance:
            def __init__(self, model):
                self.current_model = model

        instances = [MockInstance("model_a") for _ in range(5)]
        counts = compute_instance_counts(instances)
        assert counts == {"model_a": 5}


class TestTimelineEntry:
    """Tests for TimelineEntry dataclass."""

    def test_timeline_entry_creation(self):
        """Test creating a TimelineEntry."""
        entry = TimelineEntry(
            timestamp=1234567890.0,
            timestamp_iso="2009-02-13T23:31:30+00:00",
            event_type="deploy_migration",
            instance_counts={"model_a": 2},
            total_instances=2,
            changes_count=1,
            success=True,
            target_distribution=[0.5, 0.5],
            score=0.95
        )

        assert entry.timestamp == 1234567890.0
        assert entry.event_type == "deploy_migration"
        assert entry.instance_counts == {"model_a": 2}
        assert entry.total_instances == 2
        assert entry.changes_count == 1
        assert entry.success is True
        assert entry.target_distribution == [0.5, 0.5]
        assert entry.score == 0.95

    def test_timeline_entry_optional_fields(self):
        """Test TimelineEntry with optional fields as None."""
        entry = TimelineEntry(
            timestamp=1234567890.0,
            timestamp_iso="2009-02-13T23:31:30+00:00",
            event_type="auto_optimize",
            instance_counts={"model_a": 1},
            total_instances=1,
            changes_count=0,
            success=True,
            target_distribution=None,
            score=None
        )

        assert entry.target_distribution is None
        assert entry.score is None
