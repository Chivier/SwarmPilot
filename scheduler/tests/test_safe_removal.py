"""Unit and integration tests for safe instance removal (drain/remove flow).

Tests the complete lifecycle:
1. Draining instances (stop accepting new tasks)
2. Waiting for pending tasks to complete
3. Safe removal after tasks complete
4. Task assignment exclusion for draining instances
"""


import pytest

from src.model import Instance, InstanceStatus

# ============================================================================
# Unit Tests for InstanceRegistry Drain Methods
# ============================================================================


class TestInstanceDraining:
    """Tests for instance draining functionality."""

    async def test_start_draining_active_instance(
        self, instance_registry, sample_instance
    ):
        """Test starting drain on an ACTIVE instance."""
        await instance_registry.register(sample_instance)

        # Start draining
        drained = await instance_registry.start_draining(
            sample_instance.instance_id
        )

        assert drained.status == InstanceStatus.DRAINING
        assert drained.drain_initiated_at is not None
        assert isinstance(drained.drain_initiated_at, str)

        # Verify timestamp format (ISO 8601 with Z)
        assert drained.drain_initiated_at.endswith("Z")

    async def test_start_draining_nonexistent_instance(self, instance_registry):
        """Test draining non-existent instance raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await instance_registry.start_draining("nonexistent-id")

    async def test_start_draining_already_draining(
        self, instance_registry, sample_instance
    ):
        """Test draining already DRAINING instance raises ValueError."""
        await instance_registry.register(sample_instance)
        await instance_registry.start_draining(sample_instance.instance_id)

        with pytest.raises(ValueError, match=r"already in.*state"):
            await instance_registry.start_draining(sample_instance.instance_id)

    async def test_get_drain_status_active_instance(
        self, instance_registry, sample_instance
    ):
        """Test getting drain status for ACTIVE instance."""
        await instance_registry.register(sample_instance)

        status = await instance_registry.get_drain_status(
            sample_instance.instance_id
        )

        assert status["instance_id"] == sample_instance.instance_id
        assert status["status"] == InstanceStatus.ACTIVE
        assert status["pending_tasks"] == 0
        assert status["running_tasks"] == 0
        assert (
            status["can_remove"] is False
        )  # ACTIVE instances cannot be removed
        assert status["drain_initiated_at"] is None

    async def test_get_drain_status_draining_no_tasks(
        self, instance_registry, sample_instance
    ):
        """Test drain status for DRAINING instance with no pending tasks."""
        await instance_registry.register(sample_instance)
        await instance_registry.start_draining(sample_instance.instance_id)

        status = await instance_registry.get_drain_status(
            sample_instance.instance_id
        )

        assert status["status"] == InstanceStatus.DRAINING
        assert status["pending_tasks"] == 0
        assert (
            status["can_remove"] is True
        )  # Can remove when draining with no tasks
        assert status["drain_initiated_at"] is not None

    async def test_get_drain_status_draining_with_tasks(
        self, instance_registry, sample_instance
    ):
        """Test drain status for DRAINING instance with pending tasks."""
        await instance_registry.register(sample_instance)

        # Add pending tasks
        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.increment_pending(sample_instance.instance_id)

        await instance_registry.start_draining(sample_instance.instance_id)

        status = await instance_registry.get_drain_status(
            sample_instance.instance_id
        )

        assert status["status"] == InstanceStatus.DRAINING
        assert status["pending_tasks"] == 2
        assert (
            status["can_remove"] is False
        )  # Cannot remove while tasks pending

    async def test_get_drain_status_nonexistent_instance(
        self, instance_registry
    ):
        """Test getting drain status for non-existent instance raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await instance_registry.get_drain_status("nonexistent-id")


class TestListActive:
    """Tests for list_active() method."""

    async def test_list_active_excludes_draining(
        self, instance_registry, sample_instances
    ):
        """Test that list_active() excludes DRAINING instances."""
        # Register 3 instances
        for instance in sample_instances:
            await instance_registry.register(instance)

        # Drain one instance
        await instance_registry.start_draining(sample_instances[0].instance_id)

        # List active instances
        active = await instance_registry.list_active()

        assert len(active) == 2
        assert sample_instances[0].instance_id not in [
            i.instance_id for i in active
        ]
        assert sample_instances[1].instance_id in [
            i.instance_id for i in active
        ]
        assert sample_instances[2].instance_id in [
            i.instance_id for i in active
        ]

    async def test_list_active_with_model_filter(self, instance_registry):
        """Test list_active() with model_id filter."""
        # Create instances with different models
        instance1 = Instance(
            instance_id="inst-1",
            model_id="model-a",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "test",
                "software_version": "1.0",
                "hardware_name": "hw1",
            },
        )
        instance2 = Instance(
            instance_id="inst-2",
            model_id="model-b",
            endpoint="http://localhost:8002",
            platform_info={
                "software_name": "test",
                "software_version": "1.0",
                "hardware_name": "hw2",
            },
        )
        instance3 = Instance(
            instance_id="inst-3",
            model_id="model-a",
            endpoint="http://localhost:8003",
            platform_info={
                "software_name": "test",
                "software_version": "1.0",
                "hardware_name": "hw3",
            },
        )

        await instance_registry.register(instance1)
        await instance_registry.register(instance2)
        await instance_registry.register(instance3)

        # Drain one model-a instance
        await instance_registry.start_draining("inst-1")

        # List active model-a instances
        active = await instance_registry.list_active(model_id="model-a")

        assert len(active) == 1
        assert active[0].instance_id == "inst-3"

    async def test_list_active_all_draining(
        self, instance_registry, sample_instances
    ):
        """Test list_active() when all instances are draining."""
        for instance in sample_instances:
            await instance_registry.register(instance)
            await instance_registry.start_draining(instance.instance_id)

        active = await instance_registry.list_active()
        assert len(active) == 0


class TestSafeRemove:
    """Tests for safe_remove() method."""

    async def test_safe_remove_draining_no_tasks(
        self, instance_registry, sample_instance
    ):
        """Test safe removal of DRAINING instance with no pending tasks."""
        await instance_registry.register(sample_instance)
        await instance_registry.start_draining(sample_instance.instance_id)

        # Should succeed
        removed = await instance_registry.safe_remove(
            sample_instance.instance_id
        )

        assert removed.instance_id == sample_instance.instance_id
        assert await instance_registry.get(sample_instance.instance_id) is None

    async def test_safe_remove_active_instance_fails(
        self, instance_registry, sample_instance
    ):
        """Test safe removal fails for ACTIVE instance."""
        await instance_registry.register(sample_instance)

        with pytest.raises(ValueError, match="must be in DRAINING state"):
            await instance_registry.safe_remove(sample_instance.instance_id)

    async def test_safe_remove_with_pending_tasks_fails(
        self, instance_registry, sample_instance
    ):
        """Test safe removal fails when instance has pending tasks."""
        await instance_registry.register(sample_instance)
        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.start_draining(sample_instance.instance_id)

        with pytest.raises(ValueError, match="has 1 pending tasks"):
            await instance_registry.safe_remove(sample_instance.instance_id)

    async def test_safe_remove_nonexistent_instance(self, instance_registry):
        """Test safe removal of non-existent instance raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await instance_registry.safe_remove("nonexistent-id")

    async def test_safe_remove_cleans_up_all_data(
        self, instance_registry, sample_instance
    ):
        """Test safe removal removes instance, stats, and queue info."""
        await instance_registry.register(sample_instance)
        await instance_registry.start_draining(sample_instance.instance_id)

        await instance_registry.safe_remove(sample_instance.instance_id)

        # Verify all data is cleaned up
        assert await instance_registry.get(sample_instance.instance_id) is None
        assert (
            await instance_registry.get_stats(sample_instance.instance_id)
            is None
        )
        assert (
            await instance_registry.get_queue_info(sample_instance.instance_id)
            is None
        )


class TestDrainWorkflow:
    """Tests for complete drain → wait → remove workflow."""

    async def test_complete_drain_workflow(
        self, instance_registry, sample_instance
    ):
        """Test the complete workflow: drain → monitor → remove."""
        # 1. Register instance
        await instance_registry.register(sample_instance)

        # 2. Add pending tasks
        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.increment_pending(sample_instance.instance_id)

        # 3. Start draining
        await instance_registry.start_draining(sample_instance.instance_id)

        # 4. Check status - cannot remove yet
        status = await instance_registry.get_drain_status(
            sample_instance.instance_id
        )
        assert status["can_remove"] is False
        assert status["pending_tasks"] == 2

        # 5. Simulate task completion
        await instance_registry.decrement_pending(sample_instance.instance_id)
        await instance_registry.increment_completed(sample_instance.instance_id)

        status = await instance_registry.get_drain_status(
            sample_instance.instance_id
        )
        assert status["can_remove"] is False  # Still one task
        assert status["pending_tasks"] == 1

        # 6. Complete last task
        await instance_registry.decrement_pending(sample_instance.instance_id)
        await instance_registry.increment_completed(sample_instance.instance_id)

        status = await instance_registry.get_drain_status(
            sample_instance.instance_id
        )
        assert status["can_remove"] is True  # Now safe to remove
        assert status["pending_tasks"] == 0

        # 7. Safe removal
        removed = await instance_registry.safe_remove(
            sample_instance.instance_id
        )
        assert removed.instance_id == sample_instance.instance_id

    async def test_drain_timestamp_persists(
        self, instance_registry, sample_instance
    ):
        """Test that drain_initiated_at timestamp persists through status checks."""
        await instance_registry.register(sample_instance)

        # Start draining
        await instance_registry.start_draining(sample_instance.instance_id)
        drained = await instance_registry.get(sample_instance.instance_id)
        original_timestamp = drained.drain_initiated_at

        # Check status multiple times
        for _ in range(3):
            status = await instance_registry.get_drain_status(
                sample_instance.instance_id
            )
            assert status["drain_initiated_at"] == original_timestamp

    async def test_list_all_vs_list_active_after_drain(
        self, instance_registry, sample_instances
    ):
        """Test difference between list_all and list_active after draining."""
        for instance in sample_instances:
            await instance_registry.register(instance)

        # Drain two instances
        await instance_registry.start_draining(sample_instances[0].instance_id)
        await instance_registry.start_draining(sample_instances[1].instance_id)

        all_instances = await instance_registry.list_all()
        active_instances = await instance_registry.list_active()

        assert len(all_instances) == 3
        assert len(active_instances) == 1
        assert (
            active_instances[0].instance_id == sample_instances[2].instance_id
        )


# ============================================================================
# Integration Test for Task Assignment Exclusion
# ============================================================================


class TestTaskAssignmentExclusion:
    """Integration tests verifying that DRAINING instances don't receive new tasks.

    Note: These tests verify the instance_registry.list_active() behavior.
    Full API-level testing is in test_api.py.
    """

    async def test_draining_instances_excluded_from_active_list(
        self, instance_registry
    ):
        """Test that draining instances are excluded from task assignment pool."""
        # Create instances
        instances = [
            Instance(
                instance_id=f"instance-{i}",
                model_id="test-model",
                endpoint=f"http://localhost:800{i}",
                platform_info={
                    "software_name": "test",
                    "software_version": "1.0",
                    "hardware_name": f"hw{i}",
                },
            )
            for i in range(5)
        ]

        # Register all instances
        for instance in instances:
            await instance_registry.register(instance)

        # Initially all should be available
        available = await instance_registry.list_active(model_id="test-model")
        assert len(available) == 5

        # Drain 2 instances
        await instance_registry.start_draining("instance-0")
        await instance_registry.start_draining("instance-1")

        # Now only 3 should be available
        available = await instance_registry.list_active(model_id="test-model")
        assert len(available) == 3

        # Verify the correct ones are available
        available_ids = {i.instance_id for i in available}
        assert "instance-0" not in available_ids
        assert "instance-1" not in available_ids
        assert "instance-2" in available_ids
        assert "instance-3" in available_ids
        assert "instance-4" in available_ids

    async def test_multiple_drain_cycles(self, instance_registry):
        """Test multiple drain/remove cycles don't cause issues."""
        instance1 = Instance(
            instance_id="inst-1",
            model_id="model-a",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "test",
                "software_version": "1.0",
                "hardware_name": "hw1",
            },
        )
        instance2 = Instance(
            instance_id="inst-2",
            model_id="model-a",
            endpoint="http://localhost:8002",
            platform_info={
                "software_name": "test",
                "software_version": "1.0",
                "hardware_name": "hw2",
            },
        )

        # Cycle 1: Register, drain, remove instance1
        await instance_registry.register(instance1)
        await instance_registry.start_draining("inst-1")
        await instance_registry.safe_remove("inst-1")

        # Cycle 2: Register, drain, remove instance2
        await instance_registry.register(instance2)
        await instance_registry.start_draining("inst-2")
        await instance_registry.safe_remove("inst-2")

        # Verify both are gone
        assert await instance_registry.get("inst-1") is None
        assert await instance_registry.get("inst-2") is None
        assert len(await instance_registry.list_all()) == 0

    async def test_concurrent_drain_multiple_instances(
        self, instance_registry, sample_instances
    ):
        """Test draining multiple instances concurrently."""
        for instance in sample_instances:
            await instance_registry.register(instance)

        # Drain all instances at once
        for instance in sample_instances:
            await instance_registry.start_draining(instance.instance_id)

        # Verify all are draining
        for instance in sample_instances:
            status = await instance_registry.get_drain_status(
                instance.instance_id
            )
            assert status["status"] == InstanceStatus.DRAINING

        # No instances should be available for new tasks
        active = await instance_registry.list_active()
        assert len(active) == 0

        # But all should still exist in registry
        all_instances = await instance_registry.list_all()
        assert len(all_instances) == 3
