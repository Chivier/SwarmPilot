"""Tests for /task/schedule_info endpoint.

This endpoint returns scheduling information showing which instance each task
was assigned to.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from swarmpilot.scheduler.models import TaskStatus


class TestTaskScheduleInfoEndpoint:
    """Tests for GET /task/schedule_info endpoint."""

    # ========================================================================
    # Basic Tests
    # ========================================================================

    def test_schedule_info_empty_registry(self, test_client):
        """Test schedule_info with no tasks returns empty list."""
        response = test_client.get("/v1/task/schedule_info")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["count"] == 0
        assert data["tasks"] == []

    def test_schedule_info_single_task(
        self, test_client, task_registry, instance_registry
    ):
        """Test schedule_info with a single task."""
        from swarmpilot.scheduler.models import Instance

        # Register instance directly
        async def setup():
            await instance_registry.register(
                Instance(
                    instance_id="inst-1",
                    model_id="test-model",
                    endpoint="http://localhost:8001",
                    platform_info={
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "gpu",
                    },
                )
            )

            # Create task directly with assigned_instance
            await task_registry.create_task(
                task_id="task-1",
                model_id="test-model",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-1",
            )

        asyncio.run(setup())

        # Get schedule info
        response = test_client.get("/v1/task/schedule_info")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["tasks"]) == 1

        task_info = data["tasks"][0]
        assert task_info["task_id"] == "task-1"
        assert task_info["model_id"] == "test-model"
        assert task_info["assigned_instance"] == "inst-1"
        assert task_info["status"] == "pending"

    def test_schedule_info_multiple_tasks(
        self, test_client, task_registry, instance_registry
    ):
        """Test schedule_info with multiple tasks across instances."""
        from swarmpilot.scheduler.models import Instance

        async def setup():
            # Register two instances
            for i in [1, 2]:
                await instance_registry.register(
                    Instance(
                        instance_id=f"inst-{i}",
                        model_id="test-model",
                        endpoint=f"http://localhost:800{i}",
                        platform_info={
                            "software_name": "docker",
                            "software_version": "20.10",
                            "hardware_name": "gpu",
                        },
                    )
                )

            # Create tasks distributed across instances
            for i in range(1, 5):
                instance = "inst-1" if i % 2 == 1 else "inst-2"
                await task_registry.create_task(
                    task_id=f"task-{i}",
                    model_id="test-model",
                    task_input={"prompt": f"test {i}"},
                    metadata={},
                    assigned_instance=instance,
                )

        asyncio.run(setup())

        # Get schedule info
        response = test_client.get("/v1/task/schedule_info")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["count"] == 4
        assert len(data["tasks"]) == 4

        # Verify all tasks have assigned instances
        for task in data["tasks"]:
            assert task["task_id"].startswith("task-")
            assert task["assigned_instance"] in ["inst-1", "inst-2"]

    # ========================================================================
    # Filtering Tests
    # ========================================================================

    def test_schedule_info_filter_by_model_id(
        self, test_client, task_registry, instance_registry
    ):
        """Test schedule_info filtering by model_id."""
        from swarmpilot.scheduler.models import Instance

        async def setup():
            # Register instances for different models
            await instance_registry.register(
                Instance(
                    instance_id="inst-a",
                    model_id="model-a",
                    endpoint="http://localhost:8001",
                    platform_info={
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "gpu",
                    },
                )
            )
            await instance_registry.register(
                Instance(
                    instance_id="inst-b",
                    model_id="model-b",
                    endpoint="http://localhost:8002",
                    platform_info={
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "gpu",
                    },
                )
            )

            # Create tasks for different models
            await task_registry.create_task(
                task_id="task-a1",
                model_id="model-a",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-a",
            )
            await task_registry.create_task(
                task_id="task-a2",
                model_id="model-a",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-a",
            )
            await task_registry.create_task(
                task_id="task-b1",
                model_id="model-b",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-b",
            )

        asyncio.run(setup())

        # Filter by model_id
        response = test_client.get("/v1/task/schedule_info?model_id=model-a")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["count"] == 2
        for task in data["tasks"]:
            assert task["model_id"] == "model-a"
            assert task["assigned_instance"] == "inst-a"

    def test_schedule_info_filter_by_instance_id(
        self, test_client, task_registry, instance_registry
    ):
        """Test schedule_info filtering by instance_id."""
        from swarmpilot.scheduler.models import Instance

        async def setup():
            # Register two instances
            for i in [1, 2]:
                await instance_registry.register(
                    Instance(
                        instance_id=f"inst-{i}",
                        model_id="test-model",
                        endpoint=f"http://localhost:800{i}",
                        platform_info={
                            "software_name": "docker",
                            "software_version": "20.10",
                            "hardware_name": "gpu",
                        },
                    )
                )

            # Create tasks: 3 on inst-1, 2 on inst-2
            for i in range(1, 4):
                await task_registry.create_task(
                    task_id=f"task-1-{i}",
                    model_id="test-model",
                    task_input={"prompt": f"test {i}"},
                    metadata={},
                    assigned_instance="inst-1",
                )
            for i in range(1, 3):
                await task_registry.create_task(
                    task_id=f"task-2-{i}",
                    model_id="test-model",
                    task_input={"prompt": f"test {i}"},
                    metadata={},
                    assigned_instance="inst-2",
                )

        asyncio.run(setup())

        # Filter by instance_id
        response = test_client.get("/v1/task/schedule_info?instance_id=inst-1")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["count"] == 3
        for task in data["tasks"]:
            assert task["assigned_instance"] == "inst-1"

    def test_schedule_info_filter_by_status(
        self, test_client, task_registry, instance_registry
    ):
        """Test schedule_info filtering by task status."""
        from swarmpilot.scheduler.models import Instance

        async def setup():
            # Register instance
            await instance_registry.register(
                Instance(
                    instance_id="inst-1",
                    model_id="test-model",
                    endpoint="http://localhost:8001",
                    platform_info={
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "gpu",
                    },
                )
            )

            # Create tasks with different statuses
            await task_registry.create_task(
                task_id="task-pending",
                model_id="test-model",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-1",
            )

            _task_running = await task_registry.create_task(
                task_id="task-running",
                model_id="test-model",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-1",
            )
            await task_registry.update_status(
                "task-running", TaskStatus.RUNNING
            )

            _task_completed = await task_registry.create_task(
                task_id="task-completed",
                model_id="test-model",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-1",
            )
            await task_registry.update_status(
                "task-completed", TaskStatus.COMPLETED
            )

        asyncio.run(setup())

        # Filter by status=pending
        response = test_client.get("/v1/task/schedule_info?status=pending")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert data["tasks"][0]["task_id"] == "task-pending"
        assert data["tasks"][0]["status"] == "pending"

    def test_schedule_info_filter_combined(
        self, test_client, task_registry, instance_registry
    ):
        """Test schedule_info with multiple filters combined."""
        from swarmpilot.scheduler.models import Instance

        async def setup():
            # Register instances for different models
            await instance_registry.register(
                Instance(
                    instance_id="inst-a",
                    model_id="model-a",
                    endpoint="http://localhost:8001",
                    platform_info={
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "gpu",
                    },
                )
            )
            await instance_registry.register(
                Instance(
                    instance_id="inst-b",
                    model_id="model-b",
                    endpoint="http://localhost:8002",
                    platform_info={
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "gpu",
                    },
                )
            )

            # Create tasks
            await task_registry.create_task(
                task_id="task-a1",
                model_id="model-a",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-a",
            )
            # Update to completed
            await task_registry.update_status("task-a1", TaskStatus.COMPLETED)

            await task_registry.create_task(
                task_id="task-a2",
                model_id="model-a",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-a",
            )
            # Keep pending

            await task_registry.create_task(
                task_id="task-b1",
                model_id="model-b",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-b",
            )
            await task_registry.update_status("task-b1", TaskStatus.COMPLETED)

        asyncio.run(setup())

        # Filter by model_id and status
        response = test_client.get(
            "/v1/task/schedule_info?model_id=model-a&status=completed"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert data["tasks"][0]["task_id"] == "task-a1"
        assert data["tasks"][0]["model_id"] == "model-a"
        assert data["tasks"][0]["status"] == "completed"

    # ========================================================================
    # Response Structure Tests
    # ========================================================================

    def test_schedule_info_response_fields(
        self, test_client, task_registry, instance_registry
    ):
        """Test that schedule_info response contains all required fields."""
        from swarmpilot.scheduler.models import Instance

        async def setup():
            await instance_registry.register(
                Instance(
                    instance_id="inst-1",
                    model_id="test-model",
                    endpoint="http://localhost:8001",
                    platform_info={
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "gpu",
                    },
                )
            )
            await task_registry.create_task(
                task_id="task-1",
                model_id="test-model",
                task_input={"prompt": "test"},
                metadata={"key": "value"},
                assigned_instance="inst-1",
            )

        asyncio.run(setup())

        response = test_client.get("/v1/task/schedule_info")
        assert response.status_code == 200

        data = response.json()

        # Check top-level fields
        assert "success" in data
        assert "count" in data
        assert "total" in data
        assert "tasks" in data

        # Check task fields
        task = data["tasks"][0]
        assert "task_id" in task
        assert "model_id" in task
        assert "status" in task
        assert "assigned_instance" in task
        assert "submitted_at" in task

    # ========================================================================
    # Pagination Tests
    # ========================================================================

    def test_schedule_info_pagination(
        self, test_client, task_registry, instance_registry
    ):
        """Test schedule_info with pagination parameters."""
        from swarmpilot.scheduler.models import Instance

        async def setup():
            await instance_registry.register(
                Instance(
                    instance_id="inst-1",
                    model_id="test-model",
                    endpoint="http://localhost:8001",
                    platform_info={
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "gpu",
                    },
                )
            )
            # Create 10 tasks
            for i in range(10):
                await task_registry.create_task(
                    task_id=f"task-{i:02d}",
                    model_id="test-model",
                    task_input={"prompt": f"test {i}"},
                    metadata={},
                    assigned_instance="inst-1",
                )

        asyncio.run(setup())

        # Test with limit
        response = test_client.get("/v1/task/schedule_info?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert data["total"] == 10
        assert len(data["tasks"]) == 3

        # Test with offset
        response = test_client.get("/v1/task/schedule_info?limit=3&offset=3")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert data["total"] == 10

    # ========================================================================
    # Edge Cases
    # ========================================================================

    def test_schedule_info_no_matching_filter(
        self, test_client, task_registry, instance_registry
    ):
        """Test schedule_info when filter matches no tasks."""
        from swarmpilot.scheduler.models import Instance

        async def setup():
            await instance_registry.register(
                Instance(
                    instance_id="inst-1",
                    model_id="test-model",
                    endpoint="http://localhost:8001",
                    platform_info={
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "gpu",
                    },
                )
            )
            await task_registry.create_task(
                task_id="task-1",
                model_id="test-model",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance="inst-1",
            )

        asyncio.run(setup())

        # Filter by non-existent model
        response = test_client.get(
            "/v1/task/schedule_info?model_id=non-existent"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["count"] == 0
        assert data["tasks"] == []

    def test_schedule_info_invalid_status(self, test_client):
        """Test schedule_info with invalid status filter returns 400."""
        response = test_client.get(
            "/v1/task/schedule_info?status=invalid_status"
        )
        assert response.status_code == 400

        data = response.json()
        assert data["detail"]["success"] is False
        assert "Invalid status" in data["detail"]["error"]


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from swarmpilot.scheduler.api import app

    return TestClient(app)


@pytest.fixture
def task_registry():
    """Get the global task registry from the app."""
    from swarmpilot.scheduler.api import task_registry

    return task_registry


@pytest.fixture
def instance_registry():
    """Get the global instance registry from the app."""
    from swarmpilot.scheduler.api import instance_registry

    return instance_registry
