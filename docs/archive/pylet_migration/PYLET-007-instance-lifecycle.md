# PYLET-008: Instance Lifecycle Management

## Objective

Replace the planner's direct instance management with PyLet-based lifecycle management. The planner will use PyLet to start, monitor, and terminate model instances instead of managing them directly.

## Prerequisites

- PYLET-007 completed (PyLet client integration)
- SWorker wrapper fully implemented (Phase 1)

## Background

Currently, the planner manages instances through:
1. Direct HTTP calls to start/stop instances
2. Tracking instance state in memory
3. Health check polling

With PyLet integration, the planner will:
1. Submit instances via PyLet API
2. Track PyLet instance IDs
3. Use PyLet's status monitoring
4. Cancel instances via PyLet for graceful shutdown

## Files to Create/Modify

```
planner/
└── src/
    ├── instance_manager.py       # NEW: PyLet-based instance management
    ├── deployment_service.py     # MODIFY: Use instance_manager
    └── available_instance_store.py  # MODIFY: Track pylet IDs
```

## Implementation Steps

### Step 1: Create Instance Manager

Create `planner/src/instance_manager.py`:

```python
"""Instance lifecycle management using PyLet.

This module provides high-level instance management operations
that integrate with the PyLet cluster manager.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.pylet_client import (
    AsyncPyLetClient,
    InstanceInfo,
    get_pylet_client,
)


@dataclass
class ManagedInstance:
    """A model instance managed by the planner."""

    pylet_id: str
    instance_id: str
    model_id: str
    endpoint: Optional[str] = None
    status: str = "PENDING"
    registered: bool = False
    gpu_count: int = 1


class InstanceManager:
    """Manages model instances via PyLet.

    Provides operations for deploying, monitoring, and
    terminating model instances.
    """

    def __init__(
        self,
        pylet_client: Optional[AsyncPyLetClient] = None,
        model_commands: Optional[dict[str, str]] = None,
    ):
        """Initialize instance manager.

        Args:
            pylet_client: PyLet client instance.
            model_commands: Mapping of model_id to startup command.
        """
        self._client = pylet_client
        self._model_commands = model_commands or {}
        self._instances: dict[str, ManagedInstance] = {}
        self._lock = asyncio.Lock()

    def set_pylet_client(self, client: AsyncPyLetClient) -> None:
        """Set the PyLet client."""
        self._client = client

    def register_model_command(self, model_id: str, command: str) -> None:
        """Register a startup command for a model.

        Args:
            model_id: Model identifier.
            command: Command to start the model service.
        """
        self._model_commands[model_id] = command
        logger.info(f"Registered command for {model_id}")

    async def deploy_instances(
        self,
        model_id: str,
        count: int,
        gpu_count: int = 1,
    ) -> list[ManagedInstance]:
        """Deploy model instances.

        Args:
            model_id: Model to deploy.
            count: Number of instances.
            gpu_count: GPUs per instance.

        Returns:
            List of deployed managed instances.
        """
        if not self._client:
            raise RuntimeError("PyLet client not configured")

        command = self._model_commands.get(model_id)
        if not command:
            raise ValueError(f"No command registered for {model_id}")

        logger.info(f"Deploying {count} instances of {model_id}")

        # Deploy via PyLet
        infos = await self._client.deploy_model(
            model_id=model_id,
            model_command=command,
            count=count,
            gpu_count=gpu_count,
        )

        # Track instances
        managed = []
        async with self._lock:
            for info in infos:
                instance = ManagedInstance(
                    pylet_id=info.pylet_id,
                    instance_id=info.name,
                    model_id=model_id,
                    status=info.status,
                    gpu_count=gpu_count,
                )
                self._instances[info.pylet_id] = instance
                managed.append(instance)

        return managed

    async def wait_instances_ready(
        self,
        instances: list[ManagedInstance],
        timeout: float = 300.0,
    ) -> list[ManagedInstance]:
        """Wait for instances to become running.

        Args:
            instances: Instances to wait for.
            timeout: Maximum wait time per instance.

        Returns:
            Updated instances with endpoints.
        """
        if not self._client:
            raise RuntimeError("PyLet client not configured")

        # Convert to InstanceInfo for client
        infos = [
            InstanceInfo(
                pylet_id=inst.pylet_id,
                name=inst.instance_id,
                model_id=inst.model_id,
            )
            for inst in instances
        ]

        # Wait via client
        updated_infos = await self._client.wait_instances_ready(infos, timeout)

        # Update managed instances
        async with self._lock:
            for info in updated_infos:
                if info.pylet_id in self._instances:
                    inst = self._instances[info.pylet_id]
                    inst.status = info.status
                    inst.endpoint = info.endpoint

        return instances

    async def terminate_instances(
        self,
        model_id: str,
        count: int,
    ) -> list[ManagedInstance]:
        """Terminate instances of a model.

        Args:
            model_id: Model identifier.
            count: Number of instances to terminate.

        Returns:
            List of terminated instances.
        """
        if not self._client:
            raise RuntimeError("PyLet client not configured")

        # Find instances to terminate
        async with self._lock:
            candidates = [
                inst for inst in self._instances.values()
                if inst.model_id == model_id and inst.status == "RUNNING"
            ]

        if not candidates:
            logger.warning(f"No running instances of {model_id} to terminate")
            return []

        # Limit to requested count
        to_terminate = candidates[:count]

        # Convert to InstanceInfo
        infos = [
            InstanceInfo(
                pylet_id=inst.pylet_id,
                name=inst.instance_id,
                model_id=inst.model_id,
            )
            for inst in to_terminate
        ]

        # Cancel via PyLet (triggers graceful shutdown)
        await self._client.cancel_instances(infos, wait=True)

        # Remove from tracking
        async with self._lock:
            for inst in to_terminate:
                if inst.pylet_id in self._instances:
                    del self._instances[inst.pylet_id]

        logger.info(f"Terminated {len(to_terminate)} instances of {model_id}")
        return to_terminate

    async def get_model_instances(self, model_id: str) -> list[ManagedInstance]:
        """Get all instances of a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of managed instances.
        """
        async with self._lock:
            return [
                inst for inst in self._instances.values()
                if inst.model_id == model_id
            ]

    async def get_instance_count(self, model_id: str) -> int:
        """Get count of running instances for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Number of running instances.
        """
        async with self._lock:
            return sum(
                1 for inst in self._instances.values()
                if inst.model_id == model_id and inst.status == "RUNNING"
            )

    async def sync_with_pylet(self) -> None:
        """Sync instance state with PyLet.

        Queries PyLet for actual instance status and updates
        local tracking state.
        """
        if not self._client:
            return

        # Get all instances managed by planner
        pylet_instances = await self._client.list_model_instances_by_label(
            "managed_by", "swarmpilot-planner"
        )

        async with self._lock:
            # Update existing instances
            pylet_ids = {inst.pylet_id for inst in pylet_instances}

            for info in pylet_instances:
                if info.pylet_id in self._instances:
                    self._instances[info.pylet_id].status = info.status
                    self._instances[info.pylet_id].endpoint = info.endpoint

            # Remove instances that no longer exist in PyLet
            to_remove = [
                pid for pid in self._instances
                if pid not in pylet_ids
            ]
            for pid in to_remove:
                del self._instances[pid]

        logger.debug(f"Synced {len(pylet_instances)} instances from PyLet")


# Global instance manager
_instance_manager: Optional[InstanceManager] = None


def get_instance_manager() -> Optional[InstanceManager]:
    """Get the global instance manager."""
    return _instance_manager


def create_instance_manager(
    pylet_client: Optional[AsyncPyLetClient] = None,
) -> InstanceManager:
    """Create and set global instance manager."""
    global _instance_manager
    _instance_manager = InstanceManager(pylet_client=pylet_client)
    return _instance_manager
```

### Step 2: Update Available Instance Store

Modify `planner/src/available_instance_store.py` to track PyLet IDs:

```python
"""Available instance store with PyLet ID tracking."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InstanceRecord:
    """Record of an available instance."""

    instance_id: str
    model_id: str
    endpoint: str
    pylet_id: Optional[str] = None  # NEW: PyLet instance ID
    capacity: int = 1
    current_load: int = 0

    @property
    def available_capacity(self) -> int:
        """Calculate available capacity."""
        return max(0, self.capacity - self.current_load)


class AvailableInstanceStore:
    """Store for tracking available model instances."""

    def __init__(self):
        self._instances: dict[str, InstanceRecord] = {}

    def register(
        self,
        instance_id: str,
        model_id: str,
        endpoint: str,
        pylet_id: Optional[str] = None,
        capacity: int = 1,
    ) -> None:
        """Register an available instance.

        Args:
            instance_id: Instance identifier.
            model_id: Model served by instance.
            endpoint: HTTP endpoint.
            pylet_id: PyLet instance ID (for lifecycle management).
            capacity: Maximum concurrent requests.
        """
        record = InstanceRecord(
            instance_id=instance_id,
            model_id=model_id,
            endpoint=endpoint,
            pylet_id=pylet_id,
            capacity=capacity,
        )
        self._instances[instance_id] = record

    def deregister(self, instance_id: str) -> Optional[InstanceRecord]:
        """Deregister an instance.

        Args:
            instance_id: Instance to remove.

        Returns:
            Removed record if found.
        """
        return self._instances.pop(instance_id, None)

    def get_by_pylet_id(self, pylet_id: str) -> Optional[InstanceRecord]:
        """Get instance record by PyLet ID.

        Args:
            pylet_id: PyLet instance identifier.

        Returns:
            Instance record if found.
        """
        for record in self._instances.values():
            if record.pylet_id == pylet_id:
                return record
        return None

    def get_model_instances(self, model_id: str) -> list[InstanceRecord]:
        """Get all instances for a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of instance records.
        """
        return [
            record for record in self._instances.values()
            if record.model_id == model_id
        ]
```

### Step 3: Create Tests

Create `planner/tests/test_instance_manager.py`:

```python
"""Tests for instance lifecycle management."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.instance_manager import (
    InstanceManager,
    ManagedInstance,
    create_instance_manager,
)
from src.pylet_client import InstanceInfo


class TestInstanceManager:
    """Tests for InstanceManager."""

    @pytest.fixture
    def mock_client(self):
        """Create mock PyLet client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create instance manager with mock client."""
        mgr = InstanceManager(pylet_client=mock_client)
        mgr.register_model_command("test-model", "python serve.py")
        return mgr

    @pytest.mark.asyncio
    async def test_deploy_instances(self, manager, mock_client):
        """Test deploying instances via PyLet."""
        mock_client.deploy_model.return_value = [
            InstanceInfo(
                pylet_id="pylet-1",
                name="test-model-0",
                model_id="test-model",
                status="PENDING",
            ),
            InstanceInfo(
                pylet_id="pylet-2",
                name="test-model-1",
                model_id="test-model",
                status="PENDING",
            ),
        ]

        instances = await manager.deploy_instances("test-model", count=2)

        assert len(instances) == 2
        assert instances[0].pylet_id == "pylet-1"
        assert instances[1].pylet_id == "pylet-2"
        mock_client.deploy_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_instances(self, manager, mock_client):
        """Test terminating instances."""
        # Setup: deploy first
        mock_client.deploy_model.return_value = [
            InstanceInfo(
                pylet_id="pylet-1",
                name="test-model-0",
                model_id="test-model",
                status="RUNNING",
            ),
        ]
        await manager.deploy_instances("test-model", count=1)

        # Mark as running
        manager._instances["pylet-1"].status = "RUNNING"

        # Terminate
        terminated = await manager.terminate_instances("test-model", count=1)

        assert len(terminated) == 1
        assert terminated[0].pylet_id == "pylet-1"
        mock_client.cancel_instances.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_command_registered(self, manager, mock_client):
        """Test error when no command registered."""
        with pytest.raises(ValueError, match="No command registered"):
            await manager.deploy_instances("unknown-model", count=1)
```

## Test Strategy

### Unit Tests

```bash
cd planner
uv run pytest tests/test_instance_manager.py -v
```

### Integration with PyLet

```bash
# Start PyLet cluster
cd /home/yanweiye/Projects/pylet
pylet start &
pylet start --head localhost:8000 &

# Test instance lifecycle
cd planner
uv run python -c "
import asyncio
from src.instance_manager import create_instance_manager
from src.pylet_client_async import AsyncPyLetClient

async def test():
    client = AsyncPyLetClient()
    await client.init()

    mgr = create_instance_manager(client)
    mgr.register_model_command('test', 'sleep 3600')

    instances = await mgr.deploy_instances('test', count=1)
    print(f'Deployed: {instances}')

    await asyncio.sleep(5)
    await mgr.terminate_instances('test', count=1)
    print('Terminated')

asyncio.run(test())
"
```

## Acceptance Criteria

- [ ] InstanceManager created with PyLet integration
- [ ] Deploy instances via PyLet submit
- [ ] Terminate instances via PyLet cancel
- [ ] Track PyLet IDs in instance store
- [ ] Sync state with PyLet cluster
- [ ] All tests pass

## Next Steps

Proceed to [PYLET-009](PYLET-009-deployment-strategy.md) for deployment strategy integration.

## Code References

- Current deployment service: [planner/src/deployment_service.py](../../planner/src/deployment_service.py)
- Instance store: [planner/src/available_instance_store.py](../../planner/src/available_instance_store.py)
