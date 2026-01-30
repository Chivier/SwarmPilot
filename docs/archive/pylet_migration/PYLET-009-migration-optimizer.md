# PYLET-010: Migration Optimizer Update

## Objective

Update the MigrationOptimizer to use PyLet for live migration operations. When the optimizer determines instances should be migrated between nodes, it will use PyLet's cancel and resubmit pattern.

## Prerequisites

- PYLET-008 completed (Instance lifecycle)
- PYLET-009 completed (Deployment strategy)
- Understanding of current migration logic

## Background

The MigrationOptimizer handles scenarios where:
1. Instances need to move to different nodes (load balancing)
2. Nodes are being drained (maintenance)
3. Better resource allocation is available

With PyLet, migration becomes:
1. Cancel instance on current worker
2. Submit new instance (PyLet schedules to available worker)
3. Wait for new instance to be running
4. Verify scheduler registration

## Files to Create/Modify

```
planner/
└── src/
    ├── migration_executor.py     # NEW: PyLet-based migration
    └── migration_optimizer.py    # MODIFY: Use migration executor
```

## Implementation Steps

### Step 1: Create Migration Executor

Create `planner/src/migration_executor.py`:

```python
"""Migration execution using PyLet.

Handles live migration of model instances between workers.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.instance_manager import InstanceManager, ManagedInstance
from src.pylet_client import InstanceInfo


@dataclass
class MigrationPlan:
    """Plan for migrating an instance."""

    source_pylet_id: str
    model_id: str
    instance_id: str
    reason: str  # "load_balance", "drain", "optimization"
    target_worker: Optional[str] = None  # Optional preferred worker


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    plan: MigrationPlan
    success: bool
    old_instance: Optional[ManagedInstance] = None
    new_instance: Optional[ManagedInstance] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class MigrationExecutor:
    """Executes instance migrations via PyLet.

    Uses cancel-and-resubmit pattern for live migration.
    """

    def __init__(
        self,
        instance_manager: InstanceManager,
        migration_timeout: float = 300.0,
        drain_timeout: float = 60.0,
    ):
        """Initialize migration executor.

        Args:
            instance_manager: Instance lifecycle manager.
            migration_timeout: Timeout for new instance startup.
            drain_timeout: Timeout for graceful shutdown.
        """
        self._manager = instance_manager
        self._migration_timeout = migration_timeout
        self._drain_timeout = drain_timeout

    async def migrate(self, plan: MigrationPlan) -> MigrationResult:
        """Execute a single migration.

        Args:
            plan: Migration plan to execute.

        Returns:
            MigrationResult with outcome details.
        """
        import time

        start_time = time.time()
        result = MigrationResult(plan=plan, success=False)

        try:
            # Step 1: Deploy replacement instance first
            logger.info(f"Migrating {plan.instance_id}: deploying replacement")

            new_instances = await self._manager.deploy_instances(
                model_id=plan.model_id,
                count=1,
            )

            if not new_instances:
                result.error = "Failed to deploy replacement instance"
                return result

            new_instance = new_instances[0]

            # Step 2: Wait for new instance to be ready
            logger.info(f"Waiting for replacement {new_instance.instance_id}")

            ready = await self._manager.wait_instances_ready(
                [new_instance],
                timeout=self._migration_timeout,
            )

            if not ready or ready[0].status != "RUNNING":
                result.error = "Replacement instance failed to start"
                # Cleanup failed replacement
                await self._cleanup_instance(new_instance)
                return result

            result.new_instance = ready[0]

            # Step 3: Terminate old instance (graceful shutdown)
            logger.info(f"Terminating old instance {plan.instance_id}")

            old_instances = await self._manager.get_model_instances(plan.model_id)
            old_instance = next(
                (i for i in old_instances if i.pylet_id == plan.source_pylet_id),
                None,
            )

            if old_instance:
                terminated = await self._manager.terminate_instances(
                    plan.model_id,
                    count=1,
                )
                if terminated:
                    result.old_instance = terminated[0]

            result.success = True
            logger.info(
                f"Migration complete: {plan.instance_id} -> {new_instance.instance_id}"
            )

        except Exception as e:
            logger.error(f"Migration failed for {plan.instance_id}: {e}")
            result.error = str(e)

        result.duration_seconds = time.time() - start_time
        return result

    async def migrate_batch(
        self,
        plans: list[MigrationPlan],
        parallel: int = 3,
    ) -> list[MigrationResult]:
        """Execute multiple migrations.

        Args:
            plans: List of migration plans.
            parallel: Max concurrent migrations.

        Returns:
            List of migration results.
        """
        semaphore = asyncio.Semaphore(parallel)

        async def migrate_with_limit(plan: MigrationPlan) -> MigrationResult:
            async with semaphore:
                return await self.migrate(plan)

        tasks = [migrate_with_limit(p) for p in plans]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    MigrationResult(
                        plan=plans[i],
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def drain_worker(
        self,
        worker_id: str,
        model_ids: Optional[list[str]] = None,
    ) -> list[MigrationResult]:
        """Drain all instances from a worker.

        Args:
            worker_id: Worker to drain.
            model_ids: Optional filter for specific models.

        Returns:
            List of migration results.
        """
        # Get instances on worker
        instances_to_migrate = []

        for model_id, command in self._manager._model_commands.items():
            if model_ids and model_id not in model_ids:
                continue

            instances = await self._manager.get_model_instances(model_id)
            for inst in instances:
                # Check if instance is on target worker
                # (This requires querying PyLet for instance location)
                instances_to_migrate.append(inst)

        # Create migration plans
        plans = [
            MigrationPlan(
                source_pylet_id=inst.pylet_id,
                model_id=inst.model_id,
                instance_id=inst.instance_id,
                reason="drain",
            )
            for inst in instances_to_migrate
        ]

        logger.info(f"Draining worker {worker_id}: {len(plans)} instances")
        return await self.migrate_batch(plans)

    async def _cleanup_instance(self, instance: ManagedInstance) -> None:
        """Cleanup a failed instance."""
        try:
            await self._manager.terminate_instances(
                instance.model_id,
                count=1,
            )
        except Exception as e:
            logger.warning(f"Cleanup failed for {instance.instance_id}: {e}")
```

### Step 2: Update Migration Optimizer

Modify `planner/src/migration_optimizer.py`:

```python
"""Migration optimizer with PyLet integration."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.migration_executor import (
    MigrationExecutor,
    MigrationPlan,
    MigrationResult,
)
from src.instance_manager import InstanceManager


@dataclass
class MigrationDecision:
    """Decision about whether to migrate."""

    should_migrate: bool
    plans: list[MigrationPlan]
    reason: str
    expected_improvement: float = 0.0


class MigrationOptimizer:
    """Optimizes instance placement via migration.

    Analyzes current deployment and recommends migrations
    for better resource utilization or load balancing.
    """

    def __init__(
        self,
        instance_manager: InstanceManager,
        migration_threshold: float = 0.2,
        max_concurrent_migrations: int = 3,
    ):
        """Initialize migration optimizer.

        Args:
            instance_manager: Instance lifecycle manager.
            migration_threshold: Min improvement to trigger migration.
            max_concurrent_migrations: Max parallel migrations.
        """
        self._manager = instance_manager
        self._threshold = migration_threshold
        self._max_concurrent = max_concurrent_migrations
        self._executor = MigrationExecutor(instance_manager)

    async def analyze_and_migrate(
        self,
        worker_loads: dict[str, float],
        model_latencies: dict[str, float],
    ) -> list[MigrationResult]:
        """Analyze deployment and execute beneficial migrations.

        Args:
            worker_loads: Worker ID to load percentage mapping.
            model_latencies: Model ID to average latency mapping.

        Returns:
            List of migration results.
        """
        decision = await self._analyze(worker_loads, model_latencies)

        if not decision.should_migrate:
            logger.info(f"No migrations needed: {decision.reason}")
            return []

        logger.info(
            f"Executing {len(decision.plans)} migrations: {decision.reason}"
        )

        return await self._executor.migrate_batch(
            decision.plans,
            parallel=self._max_concurrent,
        )

    async def _analyze(
        self,
        worker_loads: dict[str, float],
        model_latencies: dict[str, float],
    ) -> MigrationDecision:
        """Analyze current state and create migration plans.

        Args:
            worker_loads: Worker load percentages.
            model_latencies: Model latencies.

        Returns:
            MigrationDecision with plans if migration beneficial.
        """
        plans = []

        # Strategy 1: Load balancing
        if worker_loads:
            max_load = max(worker_loads.values())
            min_load = min(worker_loads.values())
            imbalance = max_load - min_load

            if imbalance > self._threshold:
                # Find overloaded workers
                overloaded = [
                    w for w, load in worker_loads.items()
                    if load > 0.8
                ]

                for worker_id in overloaded:
                    # Get instances on this worker and plan migrations
                    # (Simplified - actual implementation needs PyLet query)
                    pass

        # Strategy 2: Latency optimization
        for model_id, latency in model_latencies.items():
            if latency > 2.0:  # High latency threshold
                # Consider migrating to less loaded worker
                pass

        if not plans:
            return MigrationDecision(
                should_migrate=False,
                plans=[],
                reason="Current deployment is optimal",
            )

        return MigrationDecision(
            should_migrate=True,
            plans=plans,
            reason=f"Load imbalance detected",
            expected_improvement=0.15,
        )

    async def drain_worker(
        self,
        worker_id: str,
    ) -> list[MigrationResult]:
        """Drain a worker for maintenance.

        Args:
            worker_id: Worker to drain.

        Returns:
            Migration results.
        """
        return await self._executor.drain_worker(worker_id)

    async def rebalance(self) -> list[MigrationResult]:
        """Force rebalancing of all instances.

        Returns:
            Migration results.
        """
        # Get current distribution
        # Compute optimal distribution
        # Create migration plans
        # Execute migrations

        logger.info("Rebalancing not yet implemented")
        return []
```

### Step 3: Create Tests

Create `planner/tests/test_migration_executor.py`:

```python
"""Tests for migration executor."""

import pytest
from unittest.mock import AsyncMock

from src.migration_executor import (
    MigrationExecutor,
    MigrationPlan,
)
from src.instance_manager import InstanceManager, ManagedInstance


class TestMigrationExecutor:
    """Tests for MigrationExecutor."""

    @pytest.fixture
    def mock_manager(self):
        """Create mock instance manager."""
        manager = AsyncMock(spec=InstanceManager)
        manager._model_commands = {"model-a": "cmd"}
        return manager

    @pytest.fixture
    def executor(self, mock_manager):
        """Create executor with mock manager."""
        return MigrationExecutor(mock_manager)

    @pytest.mark.asyncio
    async def test_successful_migration(self, executor, mock_manager):
        """Test successful instance migration."""
        # Setup: new instance deploys and starts
        mock_manager.deploy_instances.return_value = [
            ManagedInstance(
                pylet_id="new-1",
                instance_id="new-inst",
                model_id="model-a",
                status="PENDING",
            )
        ]
        mock_manager.wait_instances_ready.return_value = [
            ManagedInstance(
                pylet_id="new-1",
                instance_id="new-inst",
                model_id="model-a",
                status="RUNNING",
            )
        ]
        mock_manager.get_model_instances.return_value = [
            ManagedInstance(
                pylet_id="old-1",
                instance_id="old-inst",
                model_id="model-a",
                status="RUNNING",
            )
        ]
        mock_manager.terminate_instances.return_value = [
            ManagedInstance(
                pylet_id="old-1",
                instance_id="old-inst",
                model_id="model-a",
                status="CANCELLED",
            )
        ]

        plan = MigrationPlan(
            source_pylet_id="old-1",
            model_id="model-a",
            instance_id="old-inst",
            reason="load_balance",
        )

        result = await executor.migrate(plan)

        assert result.success
        assert result.new_instance is not None
        assert result.new_instance.status == "RUNNING"
        assert result.old_instance is not None

    @pytest.mark.asyncio
    async def test_migration_failure_rollback(self, executor, mock_manager):
        """Test migration rollback on failure."""
        mock_manager.deploy_instances.return_value = [
            ManagedInstance(
                pylet_id="new-1",
                instance_id="new-inst",
                model_id="model-a",
                status="PENDING",
            )
        ]
        mock_manager.wait_instances_ready.return_value = [
            ManagedInstance(
                pylet_id="new-1",
                instance_id="new-inst",
                model_id="model-a",
                status="FAILED",  # Startup failed
            )
        ]

        plan = MigrationPlan(
            source_pylet_id="old-1",
            model_id="model-a",
            instance_id="old-inst",
            reason="load_balance",
        )

        result = await executor.migrate(plan)

        assert not result.success
        assert "failed to start" in result.error.lower()
        # Cleanup should be called
        mock_manager.terminate_instances.assert_called()

    @pytest.mark.asyncio
    async def test_batch_migration(self, executor, mock_manager):
        """Test parallel batch migrations."""
        mock_manager.deploy_instances.return_value = [
            ManagedInstance("new", "inst", "model-a", status="PENDING")
        ]
        mock_manager.wait_instances_ready.return_value = [
            ManagedInstance("new", "inst", "model-a", status="RUNNING")
        ]
        mock_manager.get_model_instances.return_value = []
        mock_manager.terminate_instances.return_value = []

        plans = [
            MigrationPlan(f"old-{i}", "model-a", f"inst-{i}", "test")
            for i in range(5)
        ]

        results = await executor.migrate_batch(plans, parallel=2)

        assert len(results) == 5
        assert all(r.success for r in results)
```

## Test Strategy

### Unit Tests

```bash
cd planner
uv run pytest tests/test_migration_executor.py -v
```

### Integration Test

```python
async def test_live_migration():
    """Test live migration with PyLet."""
    # 1. Deploy initial instance
    # 2. Create migration plan
    # 3. Execute migration
    # 4. Verify new instance running
    # 5. Verify old instance terminated
    # 6. Verify no task loss
```

## Acceptance Criteria

- [ ] MigrationExecutor created with cancel-resubmit pattern
- [ ] Deploy replacement before terminating original
- [ ] Rollback on migration failure
- [ ] Batch migrations with concurrency control
- [ ] Worker drain functionality
- [ ] All tests pass

## Next Steps

Proceed to [PYLET-011](PYLET-011-state-tracking.md) for state tracking implementation.

## Code References

- Current migration optimizer: [planner/src/migration_optimizer.py](../../planner/src/migration_optimizer.py)
- Instance manager: [planner/src/instance_manager.py](../../planner/src/instance_manager.py)
