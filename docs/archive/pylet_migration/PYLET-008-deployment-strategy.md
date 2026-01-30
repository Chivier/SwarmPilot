# PYLET-009: Deployment Strategy Integration

## Objective

Integrate the SwarmOptimizer's deployment decisions with PyLet-based instance management. The optimizer's output (add/remove instance decisions) will be executed using PyLet's submit/cancel operations.

## Prerequisites

- PYLET-007 completed (PyLet client integration)
- PYLET-008 completed (Instance lifecycle management)
- Understanding of SwarmOptimizer algorithm

## Background

The SwarmOptimizer computes optimal instance deployments based on:
1. Current workload distribution
2. Available resources (GPU capacity)
3. Request latency requirements

The optimizer outputs deployment changes:
```python
{
    "add": [("model_a", 2), ("model_b", 1)],
    "remove": [("model_c", 1)]
}
```

This task integrates these decisions with PyLet execution.

## Files to Create/Modify

```
planner/
└── src/
    ├── deployment_executor.py    # NEW: Execute deployment changes
    ├── deployment_service.py     # MODIFY: Use executor
    └── core/
        └── swarm_optimizer.py    # REVIEW: No changes needed
```

## Implementation Steps

### Step 1: Create Deployment Executor

Create `planner/src/deployment_executor.py`:

```python
"""Deployment execution using PyLet.

This module translates optimizer decisions into PyLet operations.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.instance_manager import InstanceManager, ManagedInstance


@dataclass
class DeploymentChange:
    """A single deployment change."""

    model_id: str
    count: int
    action: str  # "add" or "remove"


@dataclass
class DeploymentResult:
    """Result of a deployment execution."""

    successful_adds: list[ManagedInstance]
    successful_removes: list[ManagedInstance]
    failed_adds: list[tuple[str, int, Exception]]  # (model_id, count, error)
    failed_removes: list[tuple[str, int, Exception]]

    @property
    def all_successful(self) -> bool:
        """Check if all operations succeeded."""
        return not self.failed_adds and not self.failed_removes


class DeploymentExecutor:
    """Executes deployment changes via PyLet.

    Translates optimizer decisions into instance lifecycle operations.
    """

    def __init__(
        self,
        instance_manager: InstanceManager,
        parallel_deploys: int = 5,
        deploy_timeout: float = 300.0,
    ):
        """Initialize deployment executor.

        Args:
            instance_manager: Instance lifecycle manager.
            parallel_deploys: Max concurrent deployments.
            deploy_timeout: Timeout for instance startup.
        """
        self._manager = instance_manager
        self._parallel_deploys = parallel_deploys
        self._deploy_timeout = deploy_timeout
        self._semaphore = asyncio.Semaphore(parallel_deploys)

    async def execute(
        self,
        changes: dict[str, list[tuple[str, int]]],
    ) -> DeploymentResult:
        """Execute deployment changes.

        Args:
            changes: Dict with "add" and "remove" lists of (model_id, count).

        Returns:
            DeploymentResult with success/failure details.
        """
        adds = changes.get("add", [])
        removes = changes.get("remove", [])

        logger.info(
            f"Executing deployment: {len(adds)} adds, {len(removes)} removes"
        )

        result = DeploymentResult(
            successful_adds=[],
            successful_removes=[],
            failed_adds=[],
            failed_removes=[],
        )

        # Execute removes first to free resources
        if removes:
            await self._execute_removes(removes, result)

        # Then execute adds
        if adds:
            await self._execute_adds(adds, result)

        return result

    async def _execute_removes(
        self,
        removes: list[tuple[str, int]],
        result: DeploymentResult,
    ) -> None:
        """Execute instance removals.

        Args:
            removes: List of (model_id, count) to remove.
            result: Result object to populate.
        """
        tasks = []
        for model_id, count in removes:
            task = asyncio.create_task(
                self._remove_instances(model_id, count)
            )
            tasks.append((model_id, count, task))

        for model_id, count, task in tasks:
            try:
                removed = await task
                result.successful_removes.extend(removed)
                logger.info(f"Removed {len(removed)} instances of {model_id}")
            except Exception as e:
                logger.error(f"Failed to remove {count} instances of {model_id}: {e}")
                result.failed_removes.append((model_id, count, e))

    async def _execute_adds(
        self,
        adds: list[tuple[str, int]],
        result: DeploymentResult,
    ) -> None:
        """Execute instance additions.

        Args:
            adds: List of (model_id, count) to add.
            result: Result object to populate.
        """
        tasks = []
        for model_id, count in adds:
            task = asyncio.create_task(
                self._add_instances(model_id, count)
            )
            tasks.append((model_id, count, task))

        for model_id, count, task in tasks:
            try:
                added = await task
                result.successful_adds.extend(added)
                logger.info(f"Added {len(added)} instances of {model_id}")
            except Exception as e:
                logger.error(f"Failed to add {count} instances of {model_id}: {e}")
                result.failed_adds.append((model_id, count, e))

    async def _add_instances(
        self,
        model_id: str,
        count: int,
    ) -> list[ManagedInstance]:
        """Add instances with concurrency control.

        Args:
            model_id: Model to deploy.
            count: Number of instances.

        Returns:
            List of deployed instances.
        """
        async with self._semaphore:
            instances = await self._manager.deploy_instances(model_id, count)

            # Wait for instances to be ready
            ready = await self._manager.wait_instances_ready(
                instances,
                timeout=self._deploy_timeout,
            )

            # Filter to only running instances
            return [inst for inst in ready if inst.status == "RUNNING"]

    async def _remove_instances(
        self,
        model_id: str,
        count: int,
    ) -> list[ManagedInstance]:
        """Remove instances.

        Args:
            model_id: Model identifier.
            count: Number to remove.

        Returns:
            List of terminated instances.
        """
        return await self._manager.terminate_instances(model_id, count)

    async def reconcile(
        self,
        target_deployment: dict[str, int],
    ) -> DeploymentResult:
        """Reconcile current state with target deployment.

        Args:
            target_deployment: Dict of model_id -> desired count.

        Returns:
            DeploymentResult from reconciliation.
        """
        # Get current counts
        current = {}
        for model_id in target_deployment:
            current[model_id] = await self._manager.get_instance_count(model_id)

        # Compute changes
        changes = {"add": [], "remove": []}

        for model_id, target_count in target_deployment.items():
            current_count = current.get(model_id, 0)
            diff = target_count - current_count

            if diff > 0:
                changes["add"].append((model_id, diff))
            elif diff < 0:
                changes["remove"].append((model_id, abs(diff)))

        logger.info(f"Reconciliation changes: {changes}")
        return await self.execute(changes)
```

### Step 2: Integrate with Deployment Service

Modify `planner/src/deployment_service.py`:

```python
"""Deployment service with PyLet integration."""

import asyncio
from typing import Optional

from loguru import logger

from src.core.swarm_optimizer import SwarmOptimizer
from src.deployment_executor import DeploymentExecutor, DeploymentResult
from src.instance_manager import InstanceManager, create_instance_manager
from src.pylet_client_async import AsyncPyLetClient


class DeploymentService:
    """Service for managing model deployments.

    Coordinates between the optimizer and PyLet-based execution.
    """

    def __init__(
        self,
        optimizer: SwarmOptimizer,
        instance_manager: Optional[InstanceManager] = None,
        pylet_client: Optional[AsyncPyLetClient] = None,
    ):
        """Initialize deployment service.

        Args:
            optimizer: Swarm optimizer instance.
            instance_manager: Instance lifecycle manager.
            pylet_client: PyLet client for cluster operations.
        """
        self._optimizer = optimizer
        self._instance_manager = instance_manager or create_instance_manager()
        self._pylet_client = pylet_client
        self._executor: Optional[DeploymentExecutor] = None
        self._running = False

    async def start(self) -> None:
        """Start the deployment service."""
        if self._running:
            return

        # Initialize PyLet connection
        if self._pylet_client:
            await self._pylet_client.init()
            self._instance_manager.set_pylet_client(self._pylet_client)

        # Create executor
        self._executor = DeploymentExecutor(self._instance_manager)

        self._running = True
        logger.info("Deployment service started")

    async def stop(self) -> None:
        """Stop the deployment service."""
        if not self._running:
            return

        self._running = False

        if self._pylet_client:
            await self._pylet_client.shutdown()

        logger.info("Deployment service stopped")

    def register_model(self, model_id: str, command: str) -> None:
        """Register a model with its startup command.

        Args:
            model_id: Model identifier.
            command: Command to start model service.
        """
        self._instance_manager.register_model_command(model_id, command)

    async def optimize_and_deploy(
        self,
        workload: dict,
        capacity: dict,
    ) -> DeploymentResult:
        """Run optimization and execute deployment changes.

        Args:
            workload: Current workload distribution.
            capacity: Available capacity matrix.

        Returns:
            DeploymentResult from execution.
        """
        if not self._executor:
            raise RuntimeError("Service not started")

        # Run optimizer
        changes = self._optimizer.compute_changes(
            current_deployment=await self._get_current_deployment(),
            workload=workload,
            capacity=capacity,
        )

        logger.info(f"Optimizer computed changes: {changes}")

        # Execute changes
        return await self._executor.execute(changes)

    async def reconcile_deployment(
        self,
        target: dict[str, int],
    ) -> DeploymentResult:
        """Reconcile to target deployment.

        Args:
            target: Target instance counts per model.

        Returns:
            DeploymentResult from reconciliation.
        """
        if not self._executor:
            raise RuntimeError("Service not started")

        return await self._executor.reconcile(target)

    async def _get_current_deployment(self) -> dict[str, int]:
        """Get current instance counts per model."""
        deployment = {}
        for model_id in self._instance_manager._model_commands:
            count = await self._instance_manager.get_instance_count(model_id)
            deployment[model_id] = count
        return deployment
```

### Step 3: Create Tests

Create `planner/tests/test_deployment_executor.py`:

```python
"""Tests for deployment executor."""

import pytest
from unittest.mock import AsyncMock

from src.deployment_executor import DeploymentExecutor, DeploymentChange
from src.instance_manager import InstanceManager, ManagedInstance


class TestDeploymentExecutor:
    """Tests for DeploymentExecutor."""

    @pytest.fixture
    def mock_manager(self):
        """Create mock instance manager."""
        manager = AsyncMock(spec=InstanceManager)
        manager.deploy_instances.return_value = [
            ManagedInstance(
                pylet_id="p1",
                instance_id="inst-1",
                model_id="model-a",
                status="RUNNING",
            )
        ]
        manager.wait_instances_ready.return_value = [
            ManagedInstance(
                pylet_id="p1",
                instance_id="inst-1",
                model_id="model-a",
                status="RUNNING",
            )
        ]
        manager.terminate_instances.return_value = [
            ManagedInstance(
                pylet_id="p2",
                instance_id="inst-2",
                model_id="model-b",
                status="CANCELLED",
            )
        ]
        return manager

    @pytest.fixture
    def executor(self, mock_manager):
        """Create executor with mock manager."""
        return DeploymentExecutor(mock_manager)

    @pytest.mark.asyncio
    async def test_execute_add(self, executor, mock_manager):
        """Test executing add operations."""
        changes = {"add": [("model-a", 1)], "remove": []}
        result = await executor.execute(changes)

        assert len(result.successful_adds) == 1
        assert result.successful_adds[0].model_id == "model-a"
        mock_manager.deploy_instances.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_remove(self, executor, mock_manager):
        """Test executing remove operations."""
        changes = {"add": [], "remove": [("model-b", 1)]}
        result = await executor.execute(changes)

        assert len(result.successful_removes) == 1
        mock_manager.terminate_instances.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_removes_before_adds(self, executor, mock_manager):
        """Test that removes execute before adds."""
        call_order = []

        async def track_deploy(*args, **kwargs):
            call_order.append("deploy")
            return [ManagedInstance("p", "i", "m", status="RUNNING")]

        async def track_terminate(*args, **kwargs):
            call_order.append("terminate")
            return [ManagedInstance("p", "i", "m")]

        mock_manager.deploy_instances.side_effect = track_deploy
        mock_manager.terminate_instances.side_effect = track_terminate

        changes = {
            "add": [("model-a", 1)],
            "remove": [("model-b", 1)],
        }
        await executor.execute(changes)

        assert call_order == ["terminate", "deploy"]

    @pytest.mark.asyncio
    async def test_reconcile(self, executor, mock_manager):
        """Test reconciliation to target state."""
        mock_manager.get_instance_count.return_value = 1

        result = await executor.reconcile({
            "model-a": 2,  # Need 1 more
            "model-b": 0,  # Need to remove 1
        })

        assert result.all_successful or len(result.failed_adds) == 0
```

## Test Strategy

### Unit Tests

```bash
cd planner
uv run pytest tests/test_deployment_executor.py -v
```

### Integration Test

```python
async def test_full_deployment_flow():
    """Test complete deployment flow."""
    # 1. Initialize PyLet connection
    client = AsyncPyLetClient()
    await client.init()

    # 2. Create service
    optimizer = SwarmOptimizer()
    service = DeploymentService(optimizer, pylet_client=client)
    await service.start()

    # 3. Register models
    service.register_model("model-a", "vllm serve model-a")
    service.register_model("model-b", "vllm serve model-b")

    # 4. Execute deployment
    result = await service.reconcile_deployment({
        "model-a": 2,
        "model-b": 1,
    })

    assert result.all_successful
    await service.stop()
```

## Acceptance Criteria

- [ ] DeploymentExecutor created
- [ ] Removes execute before adds (resource freeing)
- [ ] Parallel deployment with concurrency control
- [ ] Reconciliation computes correct changes
- [ ] Integration with SwarmOptimizer output
- [ ] All tests pass

## Next Steps

Proceed to [PYLET-010](PYLET-010-migration-optimizer.md) for migration optimizer updates.

## Code References

- SwarmOptimizer: [planner/src/core/swarm_optimizer.py](../../planner/src/core/swarm_optimizer.py)
- Deployment service: [planner/src/deployment_service.py](../../planner/src/deployment_service.py)
