"""Migration execution via cancel-and-resubmit (PYLET-010).

This module provides the MigrationExecutor class that handles instance
migrations between workers. Since PyLet doesn't support live migration,
we use a cancel-and-resubmit pattern: terminate the instance on the source
worker and deploy a new one on the target worker.

Example:
    from swarmpilot.planner.pylet.migration_executor import MigrationExecutor

    executor = MigrationExecutor(instance_manager)

    # Migrate a single instance
    result = executor.migrate(
        pylet_id="abc123",
        target_worker="worker-2",
    )

    # Drain a worker
    results = executor.drain_worker(
        worker_id="worker-1",
        target_worker="worker-2",
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from swarmpilot.planner.pylet.instance_manager import (
    InstanceManager,
    ManagedInstance,
    ManagedInstanceStatus,
)


class MigrationStatus(str, Enum):
    """Status of a migration operation."""

    PENDING = "pending"
    DRAINING = "draining"  # Draining from scheduler
    TERMINATING = "terminating"  # Terminating old instance
    DEPLOYING = "deploying"  # Deploying new instance
    REGISTERING = "registering"  # Registering new instance
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MigrationPlan:
    """Plan for migrating instances.

    Attributes:
        migrations: List of migration operations.
        source_worker: Optional source worker to drain.
        target_worker: Optional target worker for deployments.
    """

    migrations: list[MigrationOperation]
    source_worker: str | None = None
    target_worker: str | None = None

    @property
    def total_count(self) -> int:
        """Total number of migrations."""
        return len(self.migrations)


@dataclass
class MigrationOperation:
    """A single migration operation.

    Attributes:
        pylet_id: PyLet ID of instance to migrate.
        model_id: Model identifier.
        source_worker: Source worker (if known).
        target_worker: Target worker for new instance.
        status: Current migration status.
        new_pylet_id: PyLet ID of new instance (after migration).
        error: Error message if failed.
        started_at: Migration start timestamp.
        completed_at: Migration completion timestamp.
    """

    pylet_id: str
    model_id: str
    source_worker: str | None = None
    target_worker: str | None = None
    status: MigrationStatus = MigrationStatus.PENDING
    new_pylet_id: str | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None

    @property
    def duration(self) -> float | None:
        """Get migration duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class MigrationResult:
    """Result of migration execution.

    Attributes:
        success: Whether all migrations succeeded.
        completed: List of successfully completed migrations.
        failed: List of failed migrations.
        total_duration: Total time for all migrations.
    """

    success: bool
    completed: list[MigrationOperation]
    failed: list[MigrationOperation]
    total_duration: float = 0.0

    @property
    def completed_count(self) -> int:
        """Number of successful migrations."""
        return len(self.completed)

    @property
    def failed_count(self) -> int:
        """Number of failed migrations."""
        return len(self.failed)


class MigrationExecutor:
    """Executes instance migrations via cancel-and-resubmit.

    Since PyLet doesn't support live migration, this class implements
    a cancel-and-resubmit pattern:
    1. Drain the instance from scheduler
    2. Terminate the old instance
    3. Deploy a new instance on the target worker
    4. Register the new instance with scheduler

    Attributes:
        instance_manager: InstanceManager for deployment operations.
        default_backend: Default model backend.
        default_gpu_count: Default GPU count per instance.
    """

    def __init__(
        self,
        instance_manager: InstanceManager,
        default_backend: str = "vllm",
        default_gpu_count: int = 1,
    ):
        """Initialize migration executor.

        Args:
            instance_manager: InstanceManager for operations.
            default_backend: Default model backend.
            default_gpu_count: Default GPUs per instance.
        """
        self.instance_manager = instance_manager
        self.default_backend = default_backend
        self.default_gpu_count = default_gpu_count

    def migrate(
        self,
        pylet_id: str,
        target_worker: str | None = None,
        wait_for_ready: bool = True,
        ready_timeout: float = 300.0,
    ) -> MigrationOperation:
        """Migrate a single instance to a new worker.

        Args:
            pylet_id: PyLet ID of instance to migrate.
            target_worker: Target worker ID (None for any available).
            wait_for_ready: Whether to wait for new instance to be ready.
            ready_timeout: Timeout for new instance readiness.

        Returns:
            MigrationOperation with result.
        """
        # Get the managed instance
        managed = self.instance_manager.get_instance(pylet_id)
        if not managed:
            op = MigrationOperation(
                pylet_id=pylet_id,
                model_id="unknown",
                status=MigrationStatus.FAILED,
                error="Instance not found",
            )
            return op

        op = MigrationOperation(
            pylet_id=pylet_id,
            model_id=managed.model_id,
            source_worker=managed.worker_id,
            target_worker=target_worker,
            started_at=time.time(),
        )

        try:
            # Step 1: Drain and terminate old instance
            op.status = MigrationStatus.DRAINING
            logger.info(
                f"Migrating {pylet_id} ({managed.model_id}) "
                f"to worker {target_worker or 'any'}"
            )

            op.status = MigrationStatus.TERMINATING
            success = self.instance_manager.terminate_instance(
                pylet_id, wait_for_drain=True
            )
            if not success:
                op.status = MigrationStatus.FAILED
                op.error = "Failed to terminate old instance"
                op.completed_at = time.time()
                return op

            # Step 2: Deploy new instance
            op.status = MigrationStatus.DEPLOYING
            new_instance = self.instance_manager.deploy_instance(
                model_id=managed.model_id,
                backend=managed.backend,
                gpu_count=managed.gpu_count,
                target_worker=target_worker,
            )
            op.new_pylet_id = new_instance.pylet_id

            # Step 3: Wait for ready and register
            if wait_for_ready:
                op.status = MigrationStatus.REGISTERING
                self.instance_manager.wait_instance_ready(
                    new_instance.pylet_id,
                    timeout=ready_timeout,
                )

            op.status = MigrationStatus.COMPLETED
            op.completed_at = time.time()
            logger.info(
                f"Migration completed: {pylet_id} -> {new_instance.pylet_id} "
                f"(duration: {op.duration:.1f}s)"
            )

        except Exception as e:
            op.status = MigrationStatus.FAILED
            op.error = str(e)
            op.completed_at = time.time()
            logger.opt(exception=True).error(
                f"Migration failed for {pylet_id}: {e}"
            )

        return op

    def migrate_batch(
        self,
        pylet_ids: list[str],
        target_worker: str | None = None,
        wait_for_ready: bool = True,
    ) -> MigrationResult:
        """Migrate multiple instances.

        Args:
            pylet_ids: List of PyLet IDs to migrate.
            target_worker: Target worker (None for any available).
            wait_for_ready: Whether to wait for new instances.

        Returns:
            MigrationResult with completed and failed migrations.
        """
        start_time = time.time()
        completed = []
        failed = []

        for pylet_id in pylet_ids:
            op = self.migrate(
                pylet_id,
                target_worker=target_worker,
                wait_for_ready=wait_for_ready,
            )
            if op.status == MigrationStatus.COMPLETED:
                completed.append(op)
            else:
                failed.append(op)

        total_duration = time.time() - start_time

        return MigrationResult(
            success=len(failed) == 0,
            completed=completed,
            failed=failed,
            total_duration=total_duration,
        )

    def drain_worker(
        self,
        worker_id: str,
        target_worker: str | None = None,
        wait_for_ready: bool = True,
    ) -> MigrationResult:
        """Drain all instances from a worker.

        Migrates all instances on the specified worker to other workers.

        Args:
            worker_id: Worker ID to drain.
            target_worker: Target worker (None for any available).
            wait_for_ready: Whether to wait for new instances.

        Returns:
            MigrationResult with completed and failed migrations.
        """
        logger.info(
            f"Draining worker {worker_id} to {target_worker or 'any worker'}"
        )

        # Find all instances on the worker
        instances_to_migrate = [
            i
            for i in self.instance_manager.instances.values()
            if i.worker_id == worker_id
            and i.status == ManagedInstanceStatus.ACTIVE
        ]

        if not instances_to_migrate:
            logger.info(f"No instances to drain from worker {worker_id}")
            return MigrationResult(
                success=True,
                completed=[],
                failed=[],
                total_duration=0.0,
            )

        pylet_ids = [i.pylet_id for i in instances_to_migrate]
        logger.info(f"Found {len(pylet_ids)} instances to migrate")

        return self.migrate_batch(
            pylet_ids,
            target_worker=target_worker,
            wait_for_ready=wait_for_ready,
        )

    def rebalance(
        self,
        target_distribution: dict[str, int],
        wait_for_ready: bool = True,
    ) -> MigrationResult:
        """Rebalance instances across workers.

        Migrates instances to achieve target distribution per worker.

        Args:
            target_distribution: Target instance count per worker.
            wait_for_ready: Whether to wait for new instances.

        Returns:
            MigrationResult with completed and failed migrations.
        """
        start_time = time.time()
        completed = []
        failed = []

        # Get current distribution
        current_distribution: dict[str, list[ManagedInstance]] = {}
        for instance in self.instance_manager.instances.values():
            if instance.status == ManagedInstanceStatus.ACTIVE:
                worker = instance.worker_id or "unknown"
                if worker not in current_distribution:
                    current_distribution[worker] = []
                current_distribution[worker].append(instance)

        # Calculate migrations needed
        for worker_id, current_instances in current_distribution.items():
            target_count = target_distribution.get(worker_id, 0)
            current_count = len(current_instances)

            if current_count > target_count:
                # Need to move instances away from this worker
                excess = current_count - target_count
                to_migrate = current_instances[:excess]

                # Find workers that need more instances
                for instance in to_migrate:
                    target_worker = None
                    for tw, tc in target_distribution.items():
                        tw_current = len(current_distribution.get(tw, []))
                        if tw_current < tc and tw != worker_id:
                            target_worker = tw
                            break

                    op = self.migrate(
                        instance.pylet_id,
                        target_worker=target_worker,
                        wait_for_ready=wait_for_ready,
                    )

                    if op.status == MigrationStatus.COMPLETED:
                        completed.append(op)
                    else:
                        failed.append(op)

        total_duration = time.time() - start_time

        return MigrationResult(
            success=len(failed) == 0,
            completed=completed,
            failed=failed,
            total_duration=total_duration,
        )
