"""Unit tests for PyLet integration (Phase 2).

Tests for:
- PyLetClient
- SchedulerClient
- InstanceManager
- DeploymentExecutor
- MigrationExecutor
"""

from unittest.mock import MagicMock, patch

import pytest

from swarmpilot.planner.pylet.client import (
    InstanceInfo,
    PartialDeploymentError,
    PartialDeploymentResult,
    PyLetClient,
)
from swarmpilot.planner.pylet.deployment_executor import (
    DeploymentExecutor,
    DeploymentPlan,
)
from swarmpilot.planner.pylet.instance_manager import (
    ManagedInstance,
    ManagedInstanceStatus,
)
from swarmpilot.planner.pylet.migration_executor import (
    MigrationExecutor,
    MigrationOperation,
    MigrationStatus,
)
from swarmpilot.planner.pylet.scheduler_client import SchedulerClient


class TestInstanceInfo:
    """Tests for InstanceInfo dataclass."""

    def test_from_pylet_instance(self):
        """Test creating InstanceInfo from PyLet instance data."""
        mock_instance = MagicMock()
        mock_instance.id = "test-id-123"
        mock_instance.endpoint = "192.168.1.100:8001"
        mock_instance.status = "RUNNING"
        mock_instance.labels = {
            "model_id": "Qwen/Qwen3-0.6B",
            "backend": "vllm",
            "managed_by": "swarmpilot",
        }

        info = InstanceInfo.from_pylet_instance(mock_instance)

        assert info.pylet_id == "test-id-123"
        assert info.model_id == "Qwen/Qwen3-0.6B"
        assert info.endpoint == "192.168.1.100:8001"
        assert info.status == "RUNNING"
        assert info.backend == "vllm"


class TestPyLetClient:
    """Tests for PyLetClient wrapper."""

    def test_init_not_initialized(self):
        """Test client starts not initialized."""
        client = PyLetClient("http://localhost:8000")
        assert not client.initialized

    @patch("swarmpilot.planner.pylet.client.pylet")
    def test_init_success(self, mock_pylet):
        """Test successful initialization."""
        client = PyLetClient("http://localhost:8000")
        client.init()

        mock_pylet.init.assert_called_once_with("http://localhost:8000")
        assert client.initialized

    @patch("swarmpilot.planner.pylet.client.pylet")
    def test_deploy_model_invalid_backend(self, mock_pylet):
        """Test deploy with invalid backend raises ValueError."""
        client = PyLetClient("http://localhost:8000")
        client.init()

        with pytest.raises(ValueError, match="Unsupported backend"):
            client.deploy_model("Qwen/Qwen3-0.6B", backend="invalid")

    @patch("swarmpilot.planner.pylet.client.pylet")
    def test_deploy_model_success(self, mock_pylet):
        """Test successful model deployment returns list."""
        mock_instance = MagicMock()
        mock_instance.id = "instance-123"
        mock_instance.endpoint = None
        mock_instance.status = "PENDING"
        mock_instance.labels = {
            "model_id": "Qwen/Qwen3-0.6B",
            "backend": "vllm",
            "managed_by": "swarmpilot",
        }
        mock_pylet.submit.return_value = mock_instance

        client = PyLetClient("http://localhost:8000")
        client.init()

        # deploy_model now always returns list[InstanceInfo]
        infos = client.deploy_model(
            model_id="Qwen/Qwen3-0.6B",
            backend="vllm",
            gpu_count=1,
        )

        assert isinstance(infos, list)
        assert len(infos) == 1
        assert infos[0].pylet_id == "instance-123"
        assert infos[0].model_id == "Qwen/Qwen3-0.6B"
        mock_pylet.submit.assert_called_once()

    @patch("swarmpilot.planner.pylet.client.pylet")
    def test_deploy_model_multiple(self, mock_pylet):
        """Test deploying multiple instances."""
        mock_instances = []
        for i in range(3):
            mock_inst = MagicMock()
            mock_inst.id = f"instance-{i}"
            mock_inst.endpoint = None
            mock_inst.status = "PENDING"
            mock_inst.labels = {
                "model_id": "model-a",
                "backend": "vllm",
                "managed_by": "swarmpilot",
            }
            mock_instances.append(mock_inst)

        mock_pylet.submit.side_effect = mock_instances

        client = PyLetClient("http://localhost:8000")
        client.init()

        infos = client.deploy_model("model-a", count=3)

        assert len(infos) == 3
        assert mock_pylet.submit.call_count == 3
        for i, info in enumerate(infos):
            assert info.pylet_id == f"instance-{i}"

    @patch("swarmpilot.planner.pylet.client.pylet")
    def test_deploy_model_partial_failure(self, mock_pylet):
        """Test that partial failures raise PartialDeploymentError."""
        from pylet.errors import PyletError as PyletSDKError

        mock_instance = MagicMock()
        mock_instance.id = "instance-0"
        mock_instance.endpoint = None
        mock_instance.status = "PENDING"
        mock_instance.labels = {
            "model_id": "model-a",
            "backend": "vllm",
            "managed_by": "swarmpilot",
        }

        # First succeeds, second fails, third succeeds
        mock_pylet.submit.side_effect = [
            mock_instance,
            PyletSDKError("Worker unavailable"),
            mock_instance,
        ]

        client = PyLetClient("http://localhost:8000")
        client.init()

        with pytest.raises(PartialDeploymentError) as exc_info:
            client.deploy_model("model-a", count=3)

        assert len(exc_info.value.result.succeeded) == 2
        assert len(exc_info.value.result.failed) == 1
        assert exc_info.value.result.failed[0][0] == 1  # Index 1 failed


class TestSchedulerClient:
    """Tests for SchedulerClient."""

    @patch("swarmpilot.planner.pylet.scheduler_client.httpx.Client")
    def test_register_instance_success(self, mock_client_class):
        """Test successful instance registration."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "Registered successfully"}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = SchedulerClient("http://localhost:8000")
        result = client.register_instance(
            instance_id="inst-1",
            model_id="Qwen/Qwen3-0.6B",
            endpoint="192.168.1.100:8001",
            backend="vllm",
        )

        assert result.success
        assert result.instance_id == "inst-1"
        assert result.model_id == "Qwen/Qwen3-0.6B"

    @patch("swarmpilot.planner.pylet.scheduler_client.httpx.Client")
    def test_drain_instance(self, mock_client_class):
        """Test draining an instance."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = SchedulerClient("http://localhost:8000")
        result = client.drain_instance("inst-1")

        assert result is True
        mock_client.post.assert_called_once()

    @patch("swarmpilot.planner.pylet.scheduler_client.httpx.Client")
    def test_health_check(self, mock_client_class):
        """Test scheduler health check."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = SchedulerClient("http://localhost:8000")
        result = client.health_check()

        assert result is True


class TestDeploymentPlan:
    """Tests for DeploymentPlan."""

    def test_from_diff_add_only(self):
        """Test plan creation with only adds."""
        current = {"model-a": 1}
        target = {"model-a": 3, "model-b": 2}

        plan = DeploymentPlan.from_diff(current, target)

        assert plan.total_adds == 4  # 2 for model-a, 2 for model-b
        assert plan.total_removes == 0

    def test_from_diff_remove_only(self):
        """Test plan creation with only removes."""
        current = {"model-a": 3, "model-b": 2}
        target = {"model-a": 1}

        plan = DeploymentPlan.from_diff(current, target)

        assert plan.total_adds == 0
        assert plan.total_removes == 4  # 2 from model-a, 2 from model-b

    def test_from_diff_mixed(self):
        """Test plan creation with adds and removes."""
        current = {"model-a": 2, "model-b": 3}
        target = {"model-a": 4, "model-b": 1}

        plan = DeploymentPlan.from_diff(current, target)

        assert plan.total_adds == 2  # for model-a
        assert plan.total_removes == 2  # for model-b

    def test_from_diff_no_changes(self):
        """Test plan creation with no changes."""
        current = {"model-a": 2}
        target = {"model-a": 2}

        plan = DeploymentPlan.from_diff(current, target)

        assert plan.total_adds == 0
        assert plan.total_removes == 0
        assert len(plan.actions) == 0


class TestManagedInstance:
    """Tests for ManagedInstance dataclass."""

    def test_is_active(self):
        """Test is_active property."""
        instance = ManagedInstance(
            pylet_id="test-id",
            instance_id="inst-1",
            model_id="model-a",
            status=ManagedInstanceStatus.ACTIVE,
        )
        assert instance.is_active

        instance.status = ManagedInstanceStatus.DEPLOYING
        assert not instance.is_active

    def test_is_terminal(self):
        """Test is_terminal property."""
        instance = ManagedInstance(
            pylet_id="test-id",
            instance_id="inst-1",
            model_id="model-a",
            status=ManagedInstanceStatus.TERMINATED,
        )
        assert instance.is_terminal

        instance.status = ManagedInstanceStatus.FAILED
        assert instance.is_terminal

        instance.status = ManagedInstanceStatus.ACTIVE
        assert not instance.is_terminal


class TestMigrationOperation:
    """Tests for MigrationOperation dataclass."""

    def test_duration(self):
        """Test duration calculation."""
        op = MigrationOperation(
            pylet_id="test-id",
            model_id="model-a",
            started_at=100.0,
            completed_at=110.0,
        )
        assert op.duration == 10.0

    def test_duration_incomplete(self):
        """Test duration returns None for incomplete operations."""
        op = MigrationOperation(
            pylet_id="test-id",
            model_id="model-a",
            started_at=100.0,
        )
        assert op.duration is None


class TestDeploymentExecutor:
    """Tests for DeploymentExecutor."""

    def test_plan(self):
        """Test creating a deployment plan."""
        mock_manager = MagicMock()
        mock_manager.instances = {}

        executor = DeploymentExecutor(mock_manager)
        plan = executor.plan(
            current_state={"model-a": 1},
            target_state={"model-a": 3},
        )

        assert plan.total_adds == 2
        assert plan.total_removes == 0

    def test_scale_model(self):
        """Test scaling a specific model using batch deployment."""
        from swarmpilot.planner.pylet.instance_manager import DeploymentResult

        mock_manager = MagicMock()
        mock_manager.instances = {}
        mock_manager.get_instances_by_model.return_value = []
        mock_manager.get_active_instances.return_value = []

        # Mock deploy_instances to return a DeploymentResult with deployed instances
        deployed_instance = ManagedInstance(
            pylet_id="new-id",
            instance_id="new-inst",
            model_id="model-a",
            status=ManagedInstanceStatus.DEPLOYING,
        )
        mock_manager.deploy_instances.return_value = DeploymentResult(
            model_id="model-a",
            requested_count=1,
            deployed=[deployed_instance],
            failed=[],
        )
        mock_manager.wait_instances_ready.return_value = [
            ManagedInstance(
                pylet_id="new-id",
                instance_id="new-inst",
                model_id="model-a",
                status=ManagedInstanceStatus.ACTIVE,
            )
        ]

        executor = DeploymentExecutor(mock_manager)
        result = executor.scale_model("model-a", target_count=1)

        assert result.success
        assert result.added_count == 1
        # Verify batch deployment was used
        mock_manager.deploy_instances.assert_called_once()


class TestMigrationExecutor:
    """Tests for MigrationExecutor."""

    def test_migrate_not_found(self):
        """Test migrating a non-existent instance."""
        mock_manager = MagicMock()
        mock_manager.get_instance.return_value = None

        executor = MigrationExecutor(mock_manager)
        result = executor.migrate("non-existent")

        assert result.status == MigrationStatus.FAILED
        assert "not found" in result.error.lower()

    def test_migrate_batch(self):
        """Test migrating multiple instances."""
        mock_manager = MagicMock()

        # First instance succeeds
        mock_instance = ManagedInstance(
            pylet_id="inst-1",
            instance_id="inst-1",
            model_id="model-a",
            status=ManagedInstanceStatus.ACTIVE,
        )
        mock_manager.get_instance.return_value = mock_instance
        mock_manager.terminate_instance.return_value = True
        mock_manager.deploy_instance.return_value = ManagedInstance(
            pylet_id="new-inst",
            instance_id="new-inst",
            model_id="model-a",
            status=ManagedInstanceStatus.DEPLOYING,
        )
        mock_manager.wait_instance_ready.return_value = ManagedInstance(
            pylet_id="new-inst",
            instance_id="new-inst",
            model_id="model-a",
            status=ManagedInstanceStatus.ACTIVE,
        )

        executor = MigrationExecutor(mock_manager)
        result = executor.migrate_batch(["inst-1"])

        assert result.success
        assert result.completed_count == 1


class TestAvailableInstanceStore:
    """Tests for AvailableInstanceStore with PyLet ID support."""

    @pytest.mark.asyncio
    async def test_add_and_get_pylet_instance(self):
        """Test adding and retrieving instance by PyLet ID."""
        from swarmpilot.planner.available_instance_store import (
            AvailableInstance,
            AvailableInstanceStore,
        )

        store = AvailableInstanceStore()

        instance = AvailableInstance(
            model_id="model-a",
            endpoint="192.168.1.100:8001",
            pylet_id="pylet-123",
            instance_id="inst-1",
        )

        await store.add_available_instance(instance)

        # Get by PyLet ID
        result = await store.get_instance_by_pylet_id("pylet-123")
        assert result is not None
        assert result.pylet_id == "pylet-123"
        assert result.model_id == "model-a"

    @pytest.mark.asyncio
    async def test_remove_by_pylet_id(self):
        """Test removing instance by PyLet ID."""
        from swarmpilot.planner.available_instance_store import (
            AvailableInstance,
            AvailableInstanceStore,
        )

        store = AvailableInstanceStore()

        instance = AvailableInstance(
            model_id="model-a",
            endpoint="192.168.1.100:8001",
            pylet_id="pylet-123",
        )

        await store.add_available_instance(instance)
        result = await store.remove_instance_by_pylet_id("pylet-123")

        assert result is True

        # Should not find it anymore
        found = await store.get_instance_by_pylet_id("pylet-123")
        assert found is None

    @pytest.mark.asyncio
    async def test_get_all_pylet_ids(self):
        """Test getting all PyLet IDs."""
        from swarmpilot.planner.available_instance_store import (
            AvailableInstance,
            AvailableInstanceStore,
        )

        store = AvailableInstanceStore()

        # Add instances with and without PyLet IDs
        await store.add_available_instance(
            AvailableInstance(
                model_id="model-a",
                endpoint="host1:8001",
                pylet_id="pylet-1",
            )
        )
        await store.add_available_instance(
            AvailableInstance(
                model_id="model-a",
                endpoint="host2:8001",
                pylet_id="pylet-2",
            )
        )
        await store.add_available_instance(
            AvailableInstance(
                model_id="model-b",
                endpoint="host3:8001",
                # No pylet_id
            )
        )

        pylet_ids = await store.get_all_pylet_ids()

        assert len(pylet_ids) == 2
        assert "pylet-1" in pylet_ids
        assert "pylet-2" in pylet_ids
