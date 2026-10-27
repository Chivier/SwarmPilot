"""PyLet deployment service for optimizer integration (PYLET-013).

This module provides a high-level service that integrates PyLet operations
with the planner's optimizer loop. It translates optimization decisions
into PyLet deployments and manages the full instance lifecycle.

Example:
    from swarmpilot.planner.pylet.deployment_service import PyLetDeploymentService

    service = PyLetDeploymentService(
        pylet_head_url="http://pylet-head:8000",
        scheduler_url="http://scheduler:8001",
    )
    service.init()

    # Execute optimizer output
    result = service.apply_deployment(
        target_state={"model-a": 3, "model-b": 2},
    )
"""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from swarmpilot.planner.pylet.deployment_executor import (
    DeploymentExecutor,
    ExecutionResult,
)
from swarmpilot.planner.pylet.instance_manager import (
    InstanceManager,
    ManagedInstance,
)
from swarmpilot.planner.pylet.migration_executor import (
    MigrationExecutor,
    MigrationResult,
)

if TYPE_CHECKING:
    from swarmpilot.planner.available_instance_store import (
        AvailableInstanceStore,
    )


@dataclass
class DeploymentServiceResult:
    """Result of a deployment service operation.

    Attributes:
        success: Whether the operation succeeded.
        deployment_result: Result from DeploymentExecutor if scaling was done.
        migration_result: Result from MigrationExecutor if migration was done.
        active_instances: List of currently active instances.
        error: Error message if failed.
    """

    success: bool
    deployment_result: ExecutionResult | None = None
    migration_result: MigrationResult | None = None
    active_instances: list[ManagedInstance] = field(default_factory=list)
    error: str | None = None

    @property
    def total_added(self) -> int:
        """Total instances added."""
        if self.deployment_result:
            return self.deployment_result.added_count
        return 0

    @property
    def total_removed(self) -> int:
        """Total instances removed."""
        if self.deployment_result:
            return self.deployment_result.removed_count
        return 0

    @property
    def total_migrated(self) -> int:
        """Total instances migrated."""
        if self.migration_result:
            return self.migration_result.completed_count
        return 0


class PyLetDeploymentService:
    """High-level service for PyLet-based deployments.

    This service provides a simplified interface for the planner to manage
    model instances via PyLet. It handles:
    - Initial deployment of instances
    - Scaling up/down based on optimizer decisions
    - Migration between models (via cancel-and-resubmit)
    - Synchronization with AvailableInstanceStore

    Attributes:
        instance_manager: InstanceManager for lifecycle operations.
        deployment_executor: DeploymentExecutor for scaling operations.
        migration_executor: MigrationExecutor for migrations.
        default_backend: Default model backend.
        default_gpu_count: Default GPUs per instance.
        deploy_timeout: Timeout for instance deployment.
        drain_timeout: Timeout for instance drain.
    """

    def __init__(
        self,
        pylet_head_url: str,
        scheduler_url: str,
        default_backend: str = "vllm",
        default_gpu_count: int = 1,
        default_cpu_count: int = 1,
        deploy_timeout: float = 300.0,
        drain_timeout: float = 30.0,
        custom_command: str | None = None,
        reuse_cluster: bool = False,
        model_configs: dict[str, dict] | None = None,
    ):
        """Initialize PyLet deployment service.

        Args:
            pylet_head_url: URL of PyLet head node.
            scheduler_url: URL of scheduler.
            default_backend: Default model backend.
            default_gpu_count: Default GPUs per instance.
            default_cpu_count: Default CPU cores per instance.
            deploy_timeout: Timeout for instance deployment.
            drain_timeout: Timeout for instance drain.
            custom_command: Optional custom command template (overrides backend).
            reuse_cluster: If True, reuse existing PyLet connection.
            model_configs: Per-model overrides keyed by model_id.
                Each value may contain ``gpu_count`` and ``extra_args``.
        """
        self._instance_manager = InstanceManager(
            pylet_head_url=pylet_head_url,
            scheduler_url=scheduler_url,
            custom_command=custom_command,
            reuse_cluster=reuse_cluster,
        )
        self._deployment_executor: DeploymentExecutor | None = None
        self._migration_executor: MigrationExecutor | None = None
        self._default_backend = default_backend
        self._default_gpu_count = default_gpu_count
        self._default_cpu_count = default_cpu_count
        self._deploy_timeout = deploy_timeout
        self._drain_timeout = drain_timeout
        self._model_configs = model_configs or {}
        self._initialized = False
        # One reconcile at a time: the auto-optimizer applies plans from a
        # worker thread while SDK and API calls apply theirs, and two
        # executors working from the same "current" state would add or
        # remove the same machines twice.
        self._reconcile_lock = threading.RLock()
        # Replica count last asked for per model, so a replacement for a
        # failed instance never goes beyond it.
        self._desired: dict[str, int] = {}
        self._watcher: threading.Thread | None = None
        self._watcher_stop = threading.Event()
        self._replace_failed = True

    def init(self) -> None:
        """Initialize the service and connect to PyLet/scheduler.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            return

        self._instance_manager.init()
        self._deployment_executor = DeploymentExecutor(
            self._instance_manager,
            default_backend=self._default_backend,
            default_gpu_count=self._default_gpu_count,
            model_configs=self._model_configs,
        )
        self._migration_executor = MigrationExecutor(self._instance_manager)
        self._initialized = True
        logger.info("PyLetDeploymentService initialized")

    def close(self) -> None:
        """Close connections and cleanup."""
        self.stop_watcher()
        if self._instance_manager:
            self._instance_manager.close()
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    @property
    def instance_manager(self) -> InstanceManager:
        """Get the instance manager."""
        return self._instance_manager

    def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "PyLetDeploymentService not initialized. Call init() first."
            )

    def get_current_state(self) -> dict[str, int]:
        """Get current instance counts by model.

        Returns:
            Dict mapping model_id to count of active instances.
        """
        self._ensure_initialized()

        state: dict[str, int] = {}
        for instance in self._instance_manager.get_active_instances():
            model_id = instance.model_id
            state[model_id] = state.get(model_id, 0) + 1

        return state

    def spare_capacity(self) -> int:
        """Instances of the default GPU count that fit on free GPUs.

        A model pinned to one kind of hardware or using more GPUs may fit
        fewer; its extra replicas then wait in PyLet's queue and the
        deploy reports them failed.
        """
        self._ensure_initialized()
        per = max(1, self._default_gpu_count)
        return self._instance_manager.free_gpus() // per

    def free_slots(self) -> list[tuple[str, str]]:
        """Free GPUs, one entry each.

        Entries of the same worker add up to its free GPUs, so a caller
        that knows each model's GPU count (``model_requirements``) can
        tell which models still fit where.

        Returns:
            One (worker id, hardware name) pair per free GPU.
        """
        self._ensure_initialized()
        return self._instance_manager.free_slots(1)

    def model_requirements(self, model_id: str) -> tuple[int, str | None]:
        """GPUs one instance of a model takes, and its hardware pattern.

        Args:
            model_id: Model identifier.

        Returns:
            (GPU count, hardware name pattern or None for any hardware).
        """
        cfg = self._model_configs.get(model_id, {})
        gpus = int(cfg.get("gpu_count", self._default_gpu_count) or 1)
        return max(1, gpus), cfg.get("hardware") or None

    def apply_deployment(
        self,
        target_state: dict[str, int],
        wait_for_ready: bool = True,
        remove: Sequence[str] = (),
        place: dict[str, Sequence[str | None]] | None = None,
    ) -> DeploymentServiceResult:
        """Apply a deployment state computed by the optimizer.

        Computes the diff between current and target state, then executes
        the necessary add/remove operations via PyLet.

        Args:
            target_state: Target model counts {model_id: count}.
            wait_for_ready: Whether to wait for new instances to be ready.
            remove: PyLet IDs to remove first, even when a model keeps its
                count (it gets replacements).
            place: Workers for each model's new instances, in order.

        Returns:
            DeploymentServiceResult with operation details.
        """
        self._ensure_initialized()

        try:
            logger.info(f"Applying deployment: target={target_state}")

            # Execute via DeploymentExecutor
            with self._reconcile_lock:
                self._desired.update(target_state)
                result = self._deployment_executor.reconcile(
                    target_state=target_state,
                    wait_for_ready=wait_for_ready,
                    remove=remove,
                    place=place,
                )

            active = self._instance_manager.get_active_instances()

            return DeploymentServiceResult(
                success=result.success,
                deployment_result=result,
                active_instances=active,
            )

        except Exception as e:
            logger.opt(exception=True).error(f"Deployment failed: {e}")
            return DeploymentServiceResult(
                success=False,
                error=str(e),
            )

    def scale_model(
        self,
        model_id: str,
        target_count: int,
        wait_for_ready: bool = True,
    ) -> DeploymentServiceResult:
        """Scale a specific model to target count.

        Args:
            model_id: Model to scale.
            target_count: Target number of instances.
            wait_for_ready: Whether to wait for new instances.

        Returns:
            DeploymentServiceResult with operation details.
        """
        self._ensure_initialized()

        try:
            with self._reconcile_lock:
                self._desired[model_id] = target_count
                result = self._deployment_executor.scale_model(
                    model_id=model_id,
                    target_count=target_count,
                    wait_for_ready=wait_for_ready,
                )

            active = self._instance_manager.get_active_instances(model_id)

            return DeploymentServiceResult(
                success=result.success,
                deployment_result=result,
                active_instances=active,
            )

        except Exception as e:
            logger.opt(exception=True).error(
                f"Scale failed for {model_id}: {e}"
            )
            return DeploymentServiceResult(
                success=False,
                error=str(e),
            )

    def migrate_model(
        self,
        pylet_id: str,
        target_model_id: str | None = None,
    ) -> DeploymentServiceResult:
        """Migrate an instance (cancel and resubmit).

        Args:
            pylet_id: PyLet ID of instance to migrate.
            target_model_id: New model ID (or None to keep same model).

        Returns:
            DeploymentServiceResult with migration details.
        """
        self._ensure_initialized()

        try:
            with self._reconcile_lock:
                result = self._migration_executor.migrate(
                    pylet_id=pylet_id,
                    target_model_id=target_model_id,
                )

            active = self._instance_manager.get_active_instances()

            return DeploymentServiceResult(
                success=result.success,
                migration_result=result,
                active_instances=active,
            )

        except Exception as e:
            logger.opt(exception=True).error(
                f"Migration failed for {pylet_id}: {e}"
            )
            return DeploymentServiceResult(
                success=False,
                error=str(e),
            )

    def set_model_config(self, model_id: str, **overrides: object) -> None:
        """Set per-model deployment overrides (see ``model_configs``).

        Args:
            model_id: Model identifier.
            **overrides: ``backend``, ``gpu_count``, ``extra_args`` or
                ``hardware``; a None value removes the override.
        """
        self._ensure_initialized()
        configs = self._deployment_executor.model_configs
        cfg = dict(configs.get(model_id, {}))
        for key, value in overrides.items():
            if value is None:
                cfg.pop(key, None)
            else:
                cfg[key] = value
        configs[model_id] = cfg
        self._model_configs = configs

    # -- failed instances ---------------------------------------------------

    def check_instances(self) -> list[ManagedInstance]:
        """Remove instances whose engine died and start replacements.

        Replacements run on a separate thread, under the reconcile lock,
        so a slow engine start never holds up the next check.

        Returns:
            The instances that were removed.
        """
        self._ensure_initialized()
        reaped = self._instance_manager.reap_failed()
        if reaped and self._replace_failed:
            counts = Counter(m.model_id for m in reaped)
            threading.Thread(
                target=self._replace,
                args=(dict(counts),),
                name="replace-failed",
                daemon=True,
            ).start()
        return reaped

    def _replace(self, lost: dict[str, int]) -> None:
        """Deploy new instances for ones that failed.

        A model gets at most as many as it lost, and never more than its
        last requested replica count (a scale-down or terminate since the
        failure wins).
        """
        with self._reconcile_lock:
            for model_id, n in lost.items():
                current = sum(
                    not m.is_terminal
                    for m in self._instance_manager.get_instances_by_model(
                        model_id
                    )
                )
                desired = self._desired.get(model_id)
                need = n if desired is None else min(n, desired - current)
                if need <= 0:
                    continue
                logger.info(
                    f"[INSTANCE_REPLACE] model_id={model_id} count={need} "
                    f"(current={current}, desired={desired})"
                )
                try:
                    result = self._deployment_executor.execute(
                        {model_id: current},
                        {model_id: current + need},
                        ready_timeout=self._deploy_timeout,
                    )
                    if not result.success:
                        logger.warning(
                            f"Replacing {model_id} instances: "
                            f"{result.failed_count} failed"
                        )
                except Exception as e:
                    logger.opt(exception=True).error(
                        f"Replacing {model_id} instances failed: {e}"
                    )

    def start_watcher(self, interval: float, replace: bool = True) -> None:
        """Check for failed instances every ``interval`` seconds.

        Args:
            interval: Seconds between checks; 0 or less does nothing.
            replace: Deploy a replacement for each failed instance.
        """
        self._ensure_initialized()
        if interval <= 0 or self._watcher is not None:
            return
        self._replace_failed = replace
        self._watcher_stop.clear()

        def run() -> None:
            while not self._watcher_stop.wait(interval):
                try:
                    self.check_instances()
                except Exception as e:
                    logger.warning(f"Instance check failed: {e}")

        self._watcher = threading.Thread(
            target=run, name="instance-watcher", daemon=True
        )
        self._watcher.start()
        logger.info(f"Watching instances every {interval}s (replace={replace})")

    def stop_watcher(self) -> None:
        """Stop the instance watcher."""
        self._watcher_stop.set()
        if self._watcher is not None:
            self._watcher.join(timeout=10)
            self._watcher = None

    def sync_to_instance_store(
        self,
        instance_store: AvailableInstanceStore,
    ) -> int:
        """Sync active instances to AvailableInstanceStore.

        This registers all active PyLet-managed instances with the
        AvailableInstanceStore so the planner can use them for migrations.

        Args:
            instance_store: AvailableInstanceStore to sync to.

        Returns:
            Number of instances synced.
        """
        self._ensure_initialized()

        from swarmpilot.planner.available_instance_store import (
            AvailableInstance,
        )

        synced = 0
        for instance in self._instance_manager.get_active_instances():
            if instance.endpoint:
                available = AvailableInstance(
                    model_id=instance.model_id,
                    endpoint=instance.endpoint,
                    pylet_id=instance.pylet_id,
                    instance_id=instance.instance_id,
                )
                # Use sync wrapper for async method
                asyncio.get_event_loop().run_until_complete(
                    instance_store.add_available_instance(available)
                )
                synced += 1
                logger.debug(
                    f"Synced instance {instance.pylet_id} to store: "
                    f"{instance.model_id} @ {instance.endpoint}"
                )

        logger.info(f"Synced {synced} instances to AvailableInstanceStore")
        return synced

    async def sync_to_instance_store_async(
        self,
        instance_store: AvailableInstanceStore,
    ) -> int:
        """Async version of sync_to_instance_store.

        Args:
            instance_store: AvailableInstanceStore to sync to.

        Returns:
            Number of instances synced.
        """
        self._ensure_initialized()

        from swarmpilot.planner.available_instance_store import (
            AvailableInstance,
        )

        synced = 0
        for instance in self._instance_manager.get_active_instances():
            if instance.endpoint:
                available = AvailableInstance(
                    model_id=instance.model_id,
                    endpoint=instance.endpoint,
                    pylet_id=instance.pylet_id,
                    instance_id=instance.instance_id,
                )
                await instance_store.add_available_instance(available)
                synced += 1
                logger.debug(
                    f"Synced instance {instance.pylet_id} to store: "
                    f"{instance.model_id} @ {instance.endpoint}"
                )

        logger.info(f"Synced {synced} instances to AvailableInstanceStore")
        return synced

    def get_instances_by_model(self, model_id: str) -> list[ManagedInstance]:
        """Get all instances for a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of ManagedInstance objects.
        """
        self._ensure_initialized()
        return self._instance_manager.get_instances_by_model(model_id)

    def get_active_instances(
        self, model_id: str | None = None
    ) -> list[ManagedInstance]:
        """Get all active instances.

        Args:
            model_id: Optional filter by model ID.

        Returns:
            List of active ManagedInstance objects.
        """
        self._ensure_initialized()
        return self._instance_manager.get_active_instances(model_id)

    def terminate_all(self, wait_for_drain: bool = False) -> dict[str, bool]:
        """Terminate all managed instances.

        Args:
            wait_for_drain: Whether to wait for drain before termination.

        Returns:
            Dict mapping pylet_id to termination success.
        """
        self._ensure_initialized()
        # Cleared first, so a replacement waiting for the lock skips.
        self._desired = dict.fromkeys(self._desired, 0)
        with self._reconcile_lock:
            return self._instance_manager.terminate_all(wait_for_drain)

    def terminate_model(self, model_id: str) -> dict[str, bool]:
        """Terminate every instance of a model.

        Args:
            model_id: Model identifier.

        Returns:
            Dict mapping pylet_id to termination success.
        """
        self._ensure_initialized()
        # Cleared first, so a replacement waiting for the lock skips; one
        # already deploying finishes first and is terminated here too.
        self._desired[model_id] = 0
        with self._reconcile_lock:
            pylet_ids = [
                i.pylet_id
                for i in self._instance_manager.get_instances_by_model(model_id)
            ]
            return self._instance_manager.terminate_instances(pylet_ids)


# Global service singleton
_pylet_service: PyLetDeploymentService | None = None


def create_pylet_service(
    pylet_head_url: str,
    scheduler_url: str,
    default_backend: str = "vllm",
    default_gpu_count: int = 1,
    default_cpu_count: int = 1,
    deploy_timeout: float = 300.0,
    drain_timeout: float = 30.0,
    custom_command: str | None = None,
    reuse_cluster: bool = False,
    model_configs: dict[str, dict] | None = None,
) -> PyLetDeploymentService:
    """Create and initialize the global PyLet deployment service.

    Args:
        pylet_head_url: URL of PyLet head node.
        scheduler_url: URL of scheduler.
        default_backend: Default model backend.
        default_gpu_count: Default GPUs per instance.
        default_cpu_count: Default CPU cores per instance.
        deploy_timeout: Timeout for instance deployment.
        drain_timeout: Timeout for instance drain.
        custom_command: Optional custom command template (overrides backend).
        reuse_cluster: If True, reuse existing PyLet connection.
        model_configs: Per-model overrides keyed by model_id.
            Each value may contain ``gpu_count`` and ``extra_args``.

    Returns:
        Initialized PyLetDeploymentService.
    """
    global _pylet_service

    _pylet_service = PyLetDeploymentService(
        pylet_head_url=pylet_head_url,
        scheduler_url=scheduler_url,
        default_backend=default_backend,
        default_gpu_count=default_gpu_count,
        default_cpu_count=default_cpu_count,
        deploy_timeout=deploy_timeout,
        drain_timeout=drain_timeout,
        custom_command=custom_command,
        reuse_cluster=reuse_cluster,
        model_configs=model_configs,
    )
    _pylet_service.init()
    return _pylet_service


def get_pylet_service() -> PyLetDeploymentService:
    """Get the global PyLet deployment service.

    Returns:
        The global PyLetDeploymentService instance.

    Raises:
        RuntimeError: If service not created.
    """
    if _pylet_service is None:
        raise RuntimeError(
            "PyLet service not created. Call create_pylet_service() first."
        )
    return _pylet_service


def get_pylet_service_optional() -> PyLetDeploymentService | None:
    """Get the global PyLet service if available.

    Returns:
        PyLetDeploymentService or None if not created.
    """
    return _pylet_service
