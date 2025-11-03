"""
Unit and integration tests for safe instance removal (drain/remove flow).

Tests the complete lifecycle:
1. Draining instances (stop accepting new tasks)
2. Waiting for pending tasks to complete
3. Safe removal after tasks complete
4. Task assignment exclusion for draining instances
"""

import pytest
import time
from datetime import datetime

from src.instance_registry import InstanceRegistry
from src.model import Instance, InstanceStatus


# ============================================================================
# Unit Tests for InstanceRegistry Drain Methods
# ============================================================================

class TestInstanceDraining:
    """Tests for instance draining functionality."""

    def test_start_draining_active_instance(self, instance_registry, sample_instance):
        """Test starting drain on an ACTIVE instance."""
        instance_registry.register(sample_instance)

        # Start draining
        drained = instance_registry.start_draining(sample_instance.instance_id)

        assert drained.status == InstanceStatus.DRAINING
        assert drained.drain_initiated_at is not None
        assert isinstance(drained.drain_initiated_at, str)

        # Verify timestamp format (ISO 8601 with Z)
        assert drained.drain_initiated_at.endswith('Z')

    def test_start_draining_nonexistent_instance(self, instance_registry):
        """Test draining non-existent instance raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            instance_registry.start_draining("nonexistent-id")

    def test_start_draining_already_draining(self, instance_registry, sample_instance):
        """Test draining already DRAINING instance raises ValueError."""
        instance_registry.register(sample_instance)
        instance_registry.start_draining(sample_instance.instance_id)

        with pytest.raises(ValueError, match="already in.*state"):
            instance_registry.start_draining(sample_instance.instance_id)

    def test_get_drain_status_active_instance(self, instance_registry, sample_instance):
        """Test getting drain status for ACTIVE instance."""
        instance_registry.register(sample_instance)

        status = instance_registry.get_drain_status(sample_instance.instance_id)

        assert status["instance_id"] == sample_instance.instance_id
        assert status["status"] == InstanceStatus.ACTIVE
        assert status["pending_tasks"] == 0
        assert status["running_tasks"] == 0
        assert status["can_remove"] is False  # ACTIVE instances cannot be removed
        assert status["drain_initiated_at"] is None

    def test_get_drain_status_draining_no_tasks(self, instance_registry, sample_instance):
        """Test drain status for DRAINING instance with no pending tasks."""
        instance_registry.register(sample_instance)
        instance_registry.start_draining(sample_instance.instance_id)

        status = instance_registry.get_drain_status(sample_instance.instance_id)

        assert status["status"] == InstanceStatus.DRAINING
        assert status["pending_tasks"] == 0
        assert status["can_remove"] is True  # Can remove when draining with no tasks
        assert status["drain_initiated_at"] is not None

    def test_get_drain_status_draining_with_tasks(self, instance_registry, sample_instance):
        """Test drain status for DRAINING instance with pending tasks."""
        instance_registry.register(sample_instance)

        # Add pending tasks
        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.increment_pending(sample_instance.instance_id)

        instance_registry.start_draining(sample_instance.instance_id)

        status = instance_registry.get_drain_status(sample_instance.instance_id)

        assert status["status"] == InstanceStatus.DRAINING
        assert status["pending_tasks"] == 2
        assert status["can_remove"] is False  # Cannot remove while tasks pending

    def test_get_drain_status_nonexistent_instance(self, instance_registry):
        """Test getting drain status for non-existent instance raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            instance_registry.get_drain_status("nonexistent-id")


class TestListActive:
    """Tests for list_active() method."""

    def test_list_active_excludes_draining(self, instance_registry, sample_instances):
        """Test that list_active() excludes DRAINING instances."""
        # Register 3 instances
        for instance in sample_instances:
            instance_registry.register(instance)

        # Drain one instance
        instance_registry.start_draining(sample_instances[0].instance_id)

        # List active instances
        active = instance_registry.list_active()

        assert len(active) == 2
        assert sample_instances[0].instance_id not in [i.instance_id for i in active]
        assert sample_instances[1].instance_id in [i.instance_id for i in active]
        assert sample_instances[2].instance_id in [i.instance_id for i in active]

    def test_list_active_with_model_filter(self, instance_registry):
        """Test list_active() with model_id filter."""
        # Create instances with different models
        instance1 = Instance(
            instance_id="inst-1",
            model_id="model-a",
            endpoint="http://localhost:8001",
            platform_info={"software_name": "test", "software_version": "1.0", "hardware_name": "hw1"}
        )
        instance2 = Instance(
            instance_id="inst-2",
            model_id="model-b",
            endpoint="http://localhost:8002",
            platform_info={"software_name": "test", "software_version": "1.0", "hardware_name": "hw2"}
        )
        instance3 = Instance(
            instance_id="inst-3",
            model_id="model-a",
            endpoint="http://localhost:8003",
            platform_info={"software_name": "test", "software_version": "1.0", "hardware_name": "hw3"}
        )

        instance_registry.register(instance1)
        instance_registry.register(instance2)
        instance_registry.register(instance3)

        # Drain one model-a instance
        instance_registry.start_draining("inst-1")

        # List active model-a instances
        active = instance_registry.list_active(model_id="model-a")

        assert len(active) == 1
        assert active[0].instance_id == "inst-3"

    def test_list_active_all_draining(self, instance_registry, sample_instances):
        """Test list_active() when all instances are draining."""
        for instance in sample_instances:
            instance_registry.register(instance)
            instance_registry.start_draining(instance.instance_id)

        active = instance_registry.list_active()
        assert len(active) == 0


class TestSafeRemove:
    """Tests for safe_remove() method."""

    def test_safe_remove_draining_no_tasks(self, instance_registry, sample_instance):
        """Test safe removal of DRAINING instance with no pending tasks."""
        instance_registry.register(sample_instance)
        instance_registry.start_draining(sample_instance.instance_id)

        # Should succeed
        removed = instance_registry.safe_remove(sample_instance.instance_id)

        assert removed.instance_id == sample_instance.instance_id
        assert instance_registry.get(sample_instance.instance_id) is None

    def test_safe_remove_active_instance_fails(self, instance_registry, sample_instance):
        """Test safe removal fails for ACTIVE instance."""
        instance_registry.register(sample_instance)

        with pytest.raises(ValueError, match="must be in DRAINING state"):
            instance_registry.safe_remove(sample_instance.instance_id)

    def test_safe_remove_with_pending_tasks_fails(self, instance_registry, sample_instance):
        """Test safe removal fails when instance has pending tasks."""
        instance_registry.register(sample_instance)
        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.start_draining(sample_instance.instance_id)

        with pytest.raises(ValueError, match="has 1 pending tasks"):
            instance_registry.safe_remove(sample_instance.instance_id)

    def test_safe_remove_nonexistent_instance(self, instance_registry):
        """Test safe removal of non-existent instance raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            instance_registry.safe_remove("nonexistent-id")

    def test_safe_remove_cleans_up_all_data(self, instance_registry, sample_instance):
        """Test safe removal removes instance, stats, and queue info."""
        instance_registry.register(sample_instance)
        instance_registry.start_draining(sample_instance.instance_id)

        instance_registry.safe_remove(sample_instance.instance_id)

        # Verify all data is cleaned up
        assert instance_registry.get(sample_instance.instance_id) is None
        assert instance_registry.get_stats(sample_instance.instance_id) is None
        assert instance_registry.get_queue_info(sample_instance.instance_id) is None


class TestDrainWorkflow:
    """Tests for complete drain → wait → remove workflow."""

    def test_complete_drain_workflow(self, instance_registry, sample_instance):
        """Test the complete workflow: drain → monitor → remove."""
        # 1. Register instance
        instance_registry.register(sample_instance)

        # 2. Add pending tasks
        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.increment_pending(sample_instance.instance_id)

        # 3. Start draining
        instance_registry.start_draining(sample_instance.instance_id)

        # 4. Check status - cannot remove yet
        status = instance_registry.get_drain_status(sample_instance.instance_id)
        assert status["can_remove"] is False
        assert status["pending_tasks"] == 2

        # 5. Simulate task completion
        instance_registry.decrement_pending(sample_instance.instance_id)
        instance_registry.increment_completed(sample_instance.instance_id)

        status = instance_registry.get_drain_status(sample_instance.instance_id)
        assert status["can_remove"] is False  # Still one task
        assert status["pending_tasks"] == 1

        # 6. Complete last task
        instance_registry.decrement_pending(sample_instance.instance_id)
        instance_registry.increment_completed(sample_instance.instance_id)

        status = instance_registry.get_drain_status(sample_instance.instance_id)
        assert status["can_remove"] is True  # Now safe to remove
        assert status["pending_tasks"] == 0

        # 7. Safe removal
        removed = instance_registry.safe_remove(sample_instance.instance_id)
        assert removed.instance_id == sample_instance.instance_id

    def test_drain_timestamp_persists(self, instance_registry, sample_instance):
        """Test that drain_initiated_at timestamp persists through status checks."""
        instance_registry.register(sample_instance)

        # Start draining
        instance_registry.start_draining(sample_instance.instance_id)
        drained = instance_registry.get(sample_instance.instance_id)
        original_timestamp = drained.drain_initiated_at

        # Check status multiple times
        for _ in range(3):
            status = instance_registry.get_drain_status(sample_instance.instance_id)
            assert status["drain_initiated_at"] == original_timestamp

    def test_list_all_vs_list_active_after_drain(self, instance_registry, sample_instances):
        """Test difference between list_all and list_active after draining."""
        for instance in sample_instances:
            instance_registry.register(instance)

        # Drain two instances
        instance_registry.start_draining(sample_instances[0].instance_id)
        instance_registry.start_draining(sample_instances[1].instance_id)

        all_instances = instance_registry.list_all()
        active_instances = instance_registry.list_active()

        assert len(all_instances) == 3
        assert len(active_instances) == 1
        assert active_instances[0].instance_id == sample_instances[2].instance_id


# ============================================================================
# Integration Test for Task Assignment Exclusion
# ============================================================================

class TestTaskAssignmentExclusion:
    """
    Integration tests verifying that DRAINING instances don't receive new tasks.

    Note: These tests verify the instance_registry.list_active() behavior.
    Full API-level testing is in test_api.py.
    """

    def test_draining_instances_excluded_from_active_list(self, instance_registry):
        """Test that draining instances are excluded from task assignment pool."""
        # Create instances
        instances = [
            Instance(
                instance_id=f"instance-{i}",
                model_id="test-model",
                endpoint=f"http://localhost:800{i}",
                platform_info={"software_name": "test", "software_version": "1.0", "hardware_name": f"hw{i}"}
            )
            for i in range(5)
        ]

        # Register all instances
        for instance in instances:
            instance_registry.register(instance)

        # Initially all should be available
        available = instance_registry.list_active(model_id="test-model")
        assert len(available) == 5

        # Drain 2 instances
        instance_registry.start_draining("instance-0")
        instance_registry.start_draining("instance-1")

        # Now only 3 should be available
        available = instance_registry.list_active(model_id="test-model")
        assert len(available) == 3

        # Verify the correct ones are available
        available_ids = {i.instance_id for i in available}
        assert "instance-0" not in available_ids
        assert "instance-1" not in available_ids
        assert "instance-2" in available_ids
        assert "instance-3" in available_ids
        assert "instance-4" in available_ids

    def test_multiple_drain_cycles(self, instance_registry):
        """Test multiple drain/remove cycles don't cause issues."""
        instance1 = Instance(
            instance_id="inst-1",
            model_id="model-a",
            endpoint="http://localhost:8001",
            platform_info={"software_name": "test", "software_version": "1.0", "hardware_name": "hw1"}
        )
        instance2 = Instance(
            instance_id="inst-2",
            model_id="model-a",
            endpoint="http://localhost:8002",
            platform_info={"software_name": "test", "software_version": "1.0", "hardware_name": "hw2"}
        )

        # Cycle 1: Register, drain, remove instance1
        instance_registry.register(instance1)
        instance_registry.start_draining("inst-1")
        instance_registry.safe_remove("inst-1")

        # Cycle 2: Register, drain, remove instance2
        instance_registry.register(instance2)
        instance_registry.start_draining("inst-2")
        instance_registry.safe_remove("inst-2")

        # Verify both are gone
        assert instance_registry.get("inst-1") is None
        assert instance_registry.get("inst-2") is None
        assert len(instance_registry.list_all()) == 0

    def test_concurrent_drain_multiple_instances(self, instance_registry, sample_instances):
        """Test draining multiple instances concurrently."""
        for instance in sample_instances:
            instance_registry.register(instance)

        # Drain all instances at once
        for instance in sample_instances:
            instance_registry.start_draining(instance.instance_id)

        # Verify all are draining
        for instance in sample_instances:
            status = instance_registry.get_drain_status(instance.instance_id)
            assert status["status"] == InstanceStatus.DRAINING

        # No instances should be available for new tasks
        active = instance_registry.list_active()
        assert len(active) == 0

        # But all should still exist in registry
        all_instances = instance_registry.list_all()
        assert len(all_instances) == 3
