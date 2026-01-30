"""Unit tests for InstanceRegistry.

Tests instance management, stats tracking, queue info, and thread safety.
"""

import pytest

from src.models import (
    Instance,
    InstanceQueueBase,
    InstanceQueueProbabilistic,
    InstanceStats,
)

"""
# ============================================================================
# Basic Operations Tests
"""
# ============================================================================


class TestBasicOperations:
    """Tests for basic registry operations."""

    async def test_register_new_instance(self, instance_registry, sample_instance):
        """Test registering a new instance successfully."""
        await instance_registry.register(sample_instance)

        retrieved = await instance_registry.get(sample_instance.instance_id)
        assert retrieved is not None
        assert retrieved.instance_id == sample_instance.instance_id
        assert retrieved.model_id == sample_instance.model_id
        assert retrieved.endpoint == sample_instance.endpoint

    async def test_register_duplicate_instance(
        self, instance_registry, sample_instance
    ):
        """Test that registering duplicate instance raises ValueError."""
        await instance_registry.register(sample_instance)

        with pytest.raises(ValueError, match="already exists"):
            await instance_registry.register(sample_instance)

    async def test_register_initializes_stats(self, instance_registry, sample_instance):
        """Test that registering instance initializes stats."""
        await instance_registry.register(sample_instance)

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats is not None
        assert stats.pending_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 0

    async def test_register_initializes_queue_info(
        self, instance_registry, sample_instance
    ):
        """Test that registering instance initializes queue info."""
        await instance_registry.register(sample_instance)

        queue_info = await instance_registry.get_queue_info(sample_instance.instance_id)
        assert queue_info is not None
        assert isinstance(queue_info, InstanceQueueProbabilistic)
        assert queue_info.instance_id == sample_instance.instance_id
        assert len(queue_info.quantiles) == 4
        assert len(queue_info.values) == 4

    async def test_remove_existing_instance(self, instance_registry, sample_instance):
        """Test removing an existing instance."""
        await instance_registry.register(sample_instance)

        removed = await instance_registry.remove(sample_instance.instance_id)
        assert removed.instance_id == sample_instance.instance_id

        # Verify instance is gone
        assert await instance_registry.get(sample_instance.instance_id) is None
        assert await instance_registry.get_stats(sample_instance.instance_id) is None
        assert (
            await instance_registry.get_queue_info(sample_instance.instance_id) is None
        )

    async def test_remove_nonexistent_instance(self, instance_registry):
        """Test that removing non-existent instance raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await instance_registry.remove("nonexistent-id")

    async def test_get_existing_instance(self, instance_registry, sample_instance):
        """Test getting an existing instance."""
        await instance_registry.register(sample_instance)

        retrieved = await instance_registry.get(sample_instance.instance_id)
        assert retrieved is not None
        assert retrieved.instance_id == sample_instance.instance_id

    async def test_get_nonexistent_instance(self, instance_registry):
        """Test getting a non-existent instance returns None."""
        result = await instance_registry.get("nonexistent-id")
        assert result is None


"""
# ============================================================================
# List Operations Tests
"""
# ============================================================================


class TestListOperations:
    """Tests for listing instances."""

    async def test_list_all_empty(self, instance_registry):
        """Test listing instances from empty registry."""
        instances = await instance_registry.list_all()
        assert instances == []

    async def test_list_all_with_instances(self, instance_registry, sample_instances):
        """Test listing all instances."""
        for instance in sample_instances:
            await instance_registry.register(instance)

        all_instances = await instance_registry.list_all()
        assert len(all_instances) == len(sample_instances)

        instance_ids = [i.instance_id for i in all_instances]
        for instance in sample_instances:
            assert instance.instance_id in instance_ids

    async def test_list_filtered_by_model_id(self, instance_registry):
        """Test filtering instances by model_id."""
        # Register instances with different model_ids
        platform_info = {
            "software_name": "docker",
            "software_version": "20.10",
            "hardware_name": "test-hardware",
        }
        inst1 = Instance(
            instance_id="inst-1",
            model_id="model-a",
            endpoint="http://1",
            platform_info=platform_info,
        )
        inst2 = Instance(
            instance_id="inst-2",
            model_id="model-b",
            endpoint="http://2",
            platform_info=platform_info,
        )
        inst3 = Instance(
            instance_id="inst-3",
            model_id="model-a",
            endpoint="http://3",
            platform_info=platform_info,
        )

        await instance_registry.register(inst1)
        await instance_registry.register(inst2)
        await instance_registry.register(inst3)

        # Filter for model-a
        filtered = await instance_registry.list_all(model_id="model-a")
        assert len(filtered) == 2
        assert all(i.model_id == "model-a" for i in filtered)

        # Filter for model-b
        filtered = await instance_registry.list_all(model_id="model-b")
        assert len(filtered) == 1
        assert filtered[0].model_id == "model-b"

    async def test_list_filtered_no_matches(self, instance_registry, sample_instances):
        """Test filtering with no matching instances."""
        for instance in sample_instances:
            await instance_registry.register(instance)

        filtered = await instance_registry.list_all(model_id="nonexistent-model")
        assert filtered == []


"""
# ============================================================================
# Queue Info Management Tests
"""
# ============================================================================


class TestQueueInfoManagement:
    """Tests for queue information management."""

    async def test_get_queue_info_existing(self, instance_registry, sample_instance):
        """Test getting queue info for existing instance."""
        await instance_registry.register(sample_instance)

        queue_info = await instance_registry.get_queue_info(sample_instance.instance_id)
        assert queue_info is not None
        assert queue_info.instance_id == sample_instance.instance_id

    async def test_get_queue_info_nonexistent(self, instance_registry):
        """Test getting queue info for non-existent instance returns None."""
        queue_info = await instance_registry.get_queue_info("nonexistent-id")
        assert queue_info is None

    async def test_update_queue_info_existing(self, instance_registry, sample_instance):
        """Test updating queue info for existing instance."""
        await instance_registry.register(sample_instance)

        new_queue_info = InstanceQueueProbabilistic(
            instance_id=sample_instance.instance_id,
            quantiles=[0.5, 0.9],
            values=[50.0, 100.0],
        )

        await instance_registry.update_queue_info(
            sample_instance.instance_id, new_queue_info
        )

        retrieved = await instance_registry.get_queue_info(sample_instance.instance_id)
        assert retrieved is not None
        assert len(retrieved.quantiles) == 2
        assert retrieved.values == [50.0, 100.0]

    async def test_update_queue_info_nonexistent(self, instance_registry):
        """Test updating queue info for non-existent instance does nothing."""
        # Should not raise error
        queue_info = InstanceQueueBase(instance_id="nonexistent")
        await instance_registry.update_queue_info("nonexistent", queue_info)

        # Verify it wasn't added
        retrieved = await instance_registry.get_queue_info("nonexistent")
        assert retrieved is None

    async def test_update_queue_info_with_base_class(
        self, instance_registry, sample_instance
    ):
        """Test updating with InstanceQueueBase instead of Probabilistic."""
        await instance_registry.register(sample_instance)

        base_queue_info = InstanceQueueBase(instance_id=sample_instance.instance_id)
        await instance_registry.update_queue_info(
            sample_instance.instance_id, base_queue_info
        )

        retrieved = await instance_registry.get_queue_info(sample_instance.instance_id)
        assert retrieved is not None
        assert isinstance(retrieved, InstanceQueueBase)


"""
# ============================================================================
# Stats Management Tests
"""
# ============================================================================


class TestStatsManagement:
    """Tests for instance statistics management."""

    async def test_get_stats_existing(self, instance_registry, sample_instance):
        """Test getting stats for existing instance."""
        await instance_registry.register(sample_instance)

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats is not None
        assert isinstance(stats, InstanceStats)

    async def test_get_stats_nonexistent(self, instance_registry):
        """Test getting stats for non-existent instance returns None."""
        stats = await instance_registry.get_stats("nonexistent-id")
        assert stats is None

    async def test_increment_pending(self, instance_registry, sample_instance):
        """Test incrementing pending task count."""
        await instance_registry.register(sample_instance)

        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.increment_pending(sample_instance.instance_id)

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 2

    async def test_decrement_pending(self, instance_registry, sample_instance):
        """Test decrementing pending task count."""
        await instance_registry.register(sample_instance)

        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.decrement_pending(sample_instance.instance_id)

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 2

    async def test_decrement_pending_floor_at_zero(
        self, instance_registry, sample_instance
    ):
        """Test that decrementing pending doesn't go below zero."""
        await instance_registry.register(sample_instance)

        # Decrement when already at 0
        await instance_registry.decrement_pending(sample_instance.instance_id)
        await instance_registry.decrement_pending(sample_instance.instance_id)

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 0

    async def test_increment_completed(self, instance_registry, sample_instance):
        """Test incrementing completed task count."""
        await instance_registry.register(sample_instance)

        await instance_registry.increment_completed(sample_instance.instance_id)
        await instance_registry.increment_completed(sample_instance.instance_id)
        await instance_registry.increment_completed(sample_instance.instance_id)

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.completed_tasks == 3

    async def test_increment_failed(self, instance_registry, sample_instance):
        """Test incrementing failed task count."""
        await instance_registry.register(sample_instance)

        await instance_registry.increment_failed(sample_instance.instance_id)

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.failed_tasks == 1

    async def test_stats_operations_on_nonexistent_instance(self, instance_registry):
        """Test that stats operations on non-existent instance don't raise errors."""
        # Should not raise errors
        await instance_registry.increment_pending("nonexistent")
        await instance_registry.decrement_pending("nonexistent")
        await instance_registry.increment_completed("nonexistent")
        await instance_registry.increment_failed("nonexistent")

    async def test_combined_stats_operations(self, instance_registry, sample_instance):
        """Test multiple stats operations together."""
        await instance_registry.register(sample_instance)

        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.decrement_pending(sample_instance.instance_id)
        await instance_registry.increment_completed(sample_instance.instance_id)
        await instance_registry.increment_failed(sample_instance.instance_id)

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 2
        assert stats.completed_tasks == 1
        assert stats.failed_tasks == 1

    async def test_get_drain_status_no_stats(self, instance_registry, sample_instance):
        """Test get_drain_status when stats don't exist (line 264)."""
        await instance_registry.register(sample_instance)

        # Remove stats to simulate missing stats
        instance_registry._stats.pop(sample_instance.instance_id, None)

        # Get drain status - should return default values
        info = await instance_registry.get_drain_status(sample_instance.instance_id)
        assert info["instance_id"] == sample_instance.instance_id
        assert info["status"] == sample_instance.status
        assert info["pending_tasks"] == 0
        assert info["running_tasks"] == 0
        assert info["can_remove"] is True
        assert "drain_initiated_at" in info


"""
# ============================================================================
# Count Operations Tests
"""
# ============================================================================


class TestCountOperations:
    """Tests for instance counting."""

    async def test_get_total_count_empty(self, instance_registry):
        """Test total count on empty registry."""
        assert await instance_registry.get_total_count() == 0

    async def test_get_total_count_with_instances(
        self, instance_registry, sample_instances
    ):
        """Test total count with instances."""
        for instance in sample_instances:
            await instance_registry.register(instance)

        assert await instance_registry.get_total_count() == len(sample_instances)

    async def test_get_total_count_after_removal(
        self, instance_registry, sample_instances
    ):
        """Test total count after removing instances."""
        for instance in sample_instances:
            await instance_registry.register(instance)

        await instance_registry.remove(sample_instances[0].instance_id)

        assert await instance_registry.get_total_count() == len(sample_instances) - 1

    async def test_get_active_count(self, instance_registry, sample_instances):
        """Test active count (currently same as total)."""
        for instance in sample_instances:
            await instance_registry.register(instance)

        # Currently, active count = total count
        assert (
            await instance_registry.get_active_count()
            == await instance_registry.get_total_count()
        )


"""
# ============================================================================
# Thread Safety Tests
"""
# ============================================================================


@pytest.mark.skip(reason="ThreadSafety tests need async rewrite")
class TestThreadSafety:
    """Tests for thread-safe operations - DISABLED pending async rewrite."""

    pass
