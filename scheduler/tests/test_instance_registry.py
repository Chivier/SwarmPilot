"""
Unit tests for InstanceRegistry.

Tests instance management, stats tracking, queue info, and thread safety.
"""

import pytest
import threading
from time import sleep

from src.instance_registry import InstanceRegistry
from src.model import (
    Instance,
    InstanceQueueBase,
    InstanceQueueProbabilistic,
    InstanceStats,
)


# ============================================================================
# Basic Operations Tests
# ============================================================================

class TestBasicOperations:
    """Tests for basic registry operations."""

    def test_register_new_instance(self, instance_registry, sample_instance):
        """Test registering a new instance successfully."""
        instance_registry.register(sample_instance)

        retrieved = instance_registry.get(sample_instance.instance_id)
        assert retrieved is not None
        assert retrieved.instance_id == sample_instance.instance_id
        assert retrieved.model_id == sample_instance.model_id
        assert retrieved.endpoint == sample_instance.endpoint

    def test_register_duplicate_instance(self, instance_registry, sample_instance):
        """Test that registering duplicate instance raises ValueError."""
        instance_registry.register(sample_instance)

        with pytest.raises(ValueError, match="already exists"):
            instance_registry.register(sample_instance)

    def test_register_initializes_stats(self, instance_registry, sample_instance):
        """Test that registering instance initializes stats."""
        instance_registry.register(sample_instance)

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats is not None
        assert stats.pending_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 0

    def test_register_initializes_queue_info(self, instance_registry, sample_instance):
        """Test that registering instance initializes queue info."""
        instance_registry.register(sample_instance)

        queue_info = instance_registry.get_queue_info(sample_instance.instance_id)
        assert queue_info is not None
        assert isinstance(queue_info, InstanceQueueProbabilistic)
        assert queue_info.instance_id == sample_instance.instance_id
        assert len(queue_info.quantiles) == 4
        assert len(queue_info.values) == 4

    def test_remove_existing_instance(self, instance_registry, sample_instance):
        """Test removing an existing instance."""
        instance_registry.register(sample_instance)

        removed = instance_registry.remove(sample_instance.instance_id)
        assert removed.instance_id == sample_instance.instance_id

        # Verify instance is gone
        assert instance_registry.get(sample_instance.instance_id) is None
        assert instance_registry.get_stats(sample_instance.instance_id) is None
        assert instance_registry.get_queue_info(sample_instance.instance_id) is None

    def test_remove_nonexistent_instance(self, instance_registry):
        """Test that removing non-existent instance raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            instance_registry.remove("nonexistent-id")

    def test_get_existing_instance(self, instance_registry, sample_instance):
        """Test getting an existing instance."""
        instance_registry.register(sample_instance)

        retrieved = instance_registry.get(sample_instance.instance_id)
        assert retrieved is not None
        assert retrieved.instance_id == sample_instance.instance_id

    def test_get_nonexistent_instance(self, instance_registry):
        """Test getting a non-existent instance returns None."""
        result = instance_registry.get("nonexistent-id")
        assert result is None


# ============================================================================
# List Operations Tests
# ============================================================================

class TestListOperations:
    """Tests for listing instances."""

    def test_list_all_empty(self, instance_registry):
        """Test listing instances from empty registry."""
        instances = instance_registry.list_all()
        assert instances == []

    def test_list_all_with_instances(self, instance_registry, sample_instances):
        """Test listing all instances."""
        for instance in sample_instances:
            instance_registry.register(instance)

        all_instances = instance_registry.list_all()
        assert len(all_instances) == len(sample_instances)

        instance_ids = [i.instance_id for i in all_instances]
        for instance in sample_instances:
            assert instance.instance_id in instance_ids

    def test_list_filtered_by_model_id(self, instance_registry):
        """Test filtering instances by model_id."""
        # Register instances with different model_ids
        inst1 = Instance(instance_id="inst-1", model_id="model-a", endpoint="http://1")
        inst2 = Instance(instance_id="inst-2", model_id="model-b", endpoint="http://2")
        inst3 = Instance(instance_id="inst-3", model_id="model-a", endpoint="http://3")

        instance_registry.register(inst1)
        instance_registry.register(inst2)
        instance_registry.register(inst3)

        # Filter for model-a
        filtered = instance_registry.list_all(model_id="model-a")
        assert len(filtered) == 2
        assert all(i.model_id == "model-a" for i in filtered)

        # Filter for model-b
        filtered = instance_registry.list_all(model_id="model-b")
        assert len(filtered) == 1
        assert filtered[0].model_id == "model-b"

    def test_list_filtered_no_matches(self, instance_registry, sample_instances):
        """Test filtering with no matching instances."""
        for instance in sample_instances:
            instance_registry.register(instance)

        filtered = instance_registry.list_all(model_id="nonexistent-model")
        assert filtered == []


# ============================================================================
# Queue Info Management Tests
# ============================================================================

class TestQueueInfoManagement:
    """Tests for queue information management."""

    def test_get_queue_info_existing(self, instance_registry, sample_instance):
        """Test getting queue info for existing instance."""
        instance_registry.register(sample_instance)

        queue_info = instance_registry.get_queue_info(sample_instance.instance_id)
        assert queue_info is not None
        assert queue_info.instance_id == sample_instance.instance_id

    def test_get_queue_info_nonexistent(self, instance_registry):
        """Test getting queue info for non-existent instance returns None."""
        queue_info = instance_registry.get_queue_info("nonexistent-id")
        assert queue_info is None

    def test_update_queue_info_existing(self, instance_registry, sample_instance):
        """Test updating queue info for existing instance."""
        instance_registry.register(sample_instance)

        new_queue_info = InstanceQueueProbabilistic(
            instance_id=sample_instance.instance_id,
            quantiles=[0.5, 0.9],
            values=[50.0, 100.0]
        )

        instance_registry.update_queue_info(sample_instance.instance_id, new_queue_info)

        retrieved = instance_registry.get_queue_info(sample_instance.instance_id)
        assert retrieved is not None
        assert len(retrieved.quantiles) == 2
        assert retrieved.values == [50.0, 100.0]

    def test_update_queue_info_nonexistent(self, instance_registry):
        """Test updating queue info for non-existent instance does nothing."""
        # Should not raise error
        queue_info = InstanceQueueBase(instance_id="nonexistent")
        instance_registry.update_queue_info("nonexistent", queue_info)

        # Verify it wasn't added
        retrieved = instance_registry.get_queue_info("nonexistent")
        assert retrieved is None

    def test_update_queue_info_with_base_class(self, instance_registry, sample_instance):
        """Test updating with InstanceQueueBase instead of Probabilistic."""
        instance_registry.register(sample_instance)

        base_queue_info = InstanceQueueBase(instance_id=sample_instance.instance_id)
        instance_registry.update_queue_info(sample_instance.instance_id, base_queue_info)

        retrieved = instance_registry.get_queue_info(sample_instance.instance_id)
        assert retrieved is not None
        assert type(retrieved) == InstanceQueueBase


# ============================================================================
# Stats Management Tests
# ============================================================================

class TestStatsManagement:
    """Tests for instance statistics management."""

    def test_get_stats_existing(self, instance_registry, sample_instance):
        """Test getting stats for existing instance."""
        instance_registry.register(sample_instance)

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats is not None
        assert isinstance(stats, InstanceStats)

    def test_get_stats_nonexistent(self, instance_registry):
        """Test getting stats for non-existent instance returns None."""
        stats = instance_registry.get_stats("nonexistent-id")
        assert stats is None

    def test_increment_pending(self, instance_registry, sample_instance):
        """Test incrementing pending task count."""
        instance_registry.register(sample_instance)

        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.increment_pending(sample_instance.instance_id)

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 2

    def test_decrement_pending(self, instance_registry, sample_instance):
        """Test decrementing pending task count."""
        instance_registry.register(sample_instance)

        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.decrement_pending(sample_instance.instance_id)

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 2

    def test_decrement_pending_floor_at_zero(self, instance_registry, sample_instance):
        """Test that decrementing pending doesn't go below zero."""
        instance_registry.register(sample_instance)

        # Decrement when already at 0
        instance_registry.decrement_pending(sample_instance.instance_id)
        instance_registry.decrement_pending(sample_instance.instance_id)

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 0

    def test_increment_completed(self, instance_registry, sample_instance):
        """Test incrementing completed task count."""
        instance_registry.register(sample_instance)

        instance_registry.increment_completed(sample_instance.instance_id)
        instance_registry.increment_completed(sample_instance.instance_id)
        instance_registry.increment_completed(sample_instance.instance_id)

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats.completed_tasks == 3

    def test_increment_failed(self, instance_registry, sample_instance):
        """Test incrementing failed task count."""
        instance_registry.register(sample_instance)

        instance_registry.increment_failed(sample_instance.instance_id)

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats.failed_tasks == 1

    def test_stats_operations_on_nonexistent_instance(self, instance_registry):
        """Test that stats operations on non-existent instance don't raise errors."""
        # Should not raise errors
        instance_registry.increment_pending("nonexistent")
        instance_registry.decrement_pending("nonexistent")
        instance_registry.increment_completed("nonexistent")
        instance_registry.increment_failed("nonexistent")

    def test_combined_stats_operations(self, instance_registry, sample_instance):
        """Test multiple stats operations together."""
        instance_registry.register(sample_instance)

        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.decrement_pending(sample_instance.instance_id)
        instance_registry.increment_completed(sample_instance.instance_id)
        instance_registry.increment_failed(sample_instance.instance_id)

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 2
        assert stats.completed_tasks == 1
        assert stats.failed_tasks == 1


# ============================================================================
# Count Operations Tests
# ============================================================================

class TestCountOperations:
    """Tests for instance counting."""

    def test_get_total_count_empty(self, instance_registry):
        """Test total count on empty registry."""
        assert instance_registry.get_total_count() == 0

    def test_get_total_count_with_instances(self, instance_registry, sample_instances):
        """Test total count with instances."""
        for instance in sample_instances:
            instance_registry.register(instance)

        assert instance_registry.get_total_count() == len(sample_instances)

    def test_get_total_count_after_removal(self, instance_registry, sample_instances):
        """Test total count after removing instances."""
        for instance in sample_instances:
            instance_registry.register(instance)

        instance_registry.remove(sample_instances[0].instance_id)

        assert instance_registry.get_total_count() == len(sample_instances) - 1

    def test_get_active_count(self, instance_registry, sample_instances):
        """Test active count (currently same as total)."""
        for instance in sample_instances:
            instance_registry.register(instance)

        # Currently, active count = total count
        assert instance_registry.get_active_count() == instance_registry.get_total_count()


# ============================================================================
# Thread Safety Tests
# ============================================================================

@pytest.mark.slow
class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_registrations(self, instance_registry):
        """Test registering instances concurrently."""
        num_threads = 10
        instances_per_thread = 10

        def register_instances(thread_id):
            for i in range(instances_per_thread):
                instance = Instance(
                    instance_id=f"thread-{thread_id}-inst-{i}",
                    model_id="test-model",
                    endpoint=f"http://localhost:{8000 + thread_id}"
                )
                instance_registry.register(instance)

        threads = []
        for t in range(num_threads):
            thread = threading.Thread(target=register_instances, args=(t,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all instances were registered
        assert instance_registry.get_total_count() == num_threads * instances_per_thread

    def test_concurrent_stats_updates(self, instance_registry, sample_instance):
        """Test updating stats concurrently."""
        instance_registry.register(sample_instance)

        num_increments = 100

        def increment_stats():
            for _ in range(num_increments):
                instance_registry.increment_pending(sample_instance.instance_id)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=increment_stats)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == num_increments * 5

    def test_concurrent_list_and_modify(self, instance_registry, sample_instances):
        """Test listing while concurrently modifying."""
        # Register initial instances
        for instance in sample_instances[:2]:
            instance_registry.register(instance)

        results = []

        def list_instances():
            for _ in range(50):
                instances = instance_registry.list_all()
                results.append(len(instances))
                sleep(0.001)

        def add_remove_instances():
            instance_registry.register(sample_instances[2])
            sleep(0.01)
            instance_registry.remove(sample_instances[2].instance_id)

        list_thread = threading.Thread(target=list_instances)
        modify_thread = threading.Thread(target=add_remove_instances)

        list_thread.start()
        modify_thread.start()

        list_thread.join()
        modify_thread.join()

        # All results should be valid counts (2 or 3)
        assert all(count in [2, 3] for count in results)

    def test_concurrent_get_and_update_queue_info(self, instance_registry, sample_instance):
        """Test concurrent queue info operations."""
        instance_registry.register(sample_instance)

        def update_queue():
            for i in range(20):
                queue_info = InstanceQueueProbabilistic(
                    instance_id=sample_instance.instance_id,
                    quantiles=[0.5, 0.9],
                    values=[float(i), float(i * 2)]
                )
                instance_registry.update_queue_info(sample_instance.instance_id, queue_info)
                sleep(0.001)

        def read_queue():
            for _ in range(20):
                queue_info = instance_registry.get_queue_info(sample_instance.instance_id)
                assert queue_info is not None
                sleep(0.001)

        update_thread = threading.Thread(target=update_queue)
        read_thread = threading.Thread(target=read_queue)

        update_thread.start()
        read_thread.start()

        update_thread.join()
        read_thread.join()

        # Verify final state is consistent
        final_queue = instance_registry.get_queue_info(sample_instance.instance_id)
        assert final_queue is not None
