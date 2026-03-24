"""Instance lifecycle manager (PYLET-008).

This module provides the InstanceManager class that orchestrates the complete
instance lifecycle: deploy via PyLet, wait for health, register with scheduler,
and cleanup on termination.

Example:
    from swarmpilot.planner.pylet.instance_manager import InstanceManager

    manager = InstanceManager(
        pylet_head_url="http://localhost:8000",
        scheduler_url="http://localhost:8001",
    )
    manager.init()

    # Deploy and register instances
    instances = manager.deploy_instances(
        model_id="Qwen/Qwen3-0.6B",
        count=2,
        gpu_count=1,
    )

    # Wait for all to be ready
    ready_instances = manager.wait_instances_ready(instances)

    # Terminate when done
    manager.terminate_instances([i.pylet_id for i in instances])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from swarmpilot.planner.pylet.client import PartialDeploymentError, PyLetClient
from swarmpilot.planner.pylet.scheduler_client import SchedulerClient
from swarmpilot.planner.scheduler_registry import get_scheduler_registry


class ManagedInstanceStatus(str, Enum):
    """Status of a managed instance in the planner's view."""

    DEPLOYING = "deploying"  # PyLet instance submitted, waiting for running
    WAITING_HEALTH = "waiting_health"  # Running, waiting for health check
    REGISTERING = "registering"  # Healthy, registering with scheduler
    ACTIVE = "active"  # Registered and serving traffic
    DRAINING = "draining"  # Stopping new tasks, completing existing
    TERMINATING = "terminating"  # Being terminated
    TERMINATED = "terminated"  # Fully terminated
    FAILED = "failed"  # Failed to deploy or register


@dataclass
class ManagedInstance:
    """A model instance managed by the planner.

    Tracks both PyLet and scheduler state for complete lifecycle management.

    Attributes:
        pylet_id: PyLet instance UUID.
        instance_id: Scheduler instance ID (may differ from pylet_id).
        model_id: Model identifier.
        endpoint: HTTP endpoint when running.
        backend: Model backend (vllm, sglang).
        status: Current lifecycle status.
        gpu_count: Number of GPUs allocated.
        worker_id: PyLet worker where instance runs.
        error: Error message if failed.
        created_at: Creation timestamp.
    """

    pylet_id: str
    instance_id: str  # For scheduler registration
    model_id: str
    endpoint: str | None = None
    backend: str = "vllm"
    status: ManagedInstanceStatus = ManagedInstanceStatus.DEPLOYING
    gpu_count: int = 1
    worker_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        """Check if instance is actively serving traffic."""
        return self.status == ManagedInstanceStatus.ACTIVE

    @property
    def is_terminal(self) -> bool:
        """Check if instance is in a terminal state."""
        return self.status in (
            ManagedInstanceStatus.TERMINATED,
            ManagedInstanceStatus.FAILED,
        )


@dataclass
class DeploymentResult:
    """Result of a deployment operation.

    Attributes:
        model_id: Model identifier.
        requested_count: Number of instances requested.
        deployed: List of successfully deployed instances.
        failed: List of (instance_id, error) tuples for failures.
    """

    model_id: str
    requested_count: int
    deployed: list[ManagedInstance]
    failed: list[tuple[str, str]]

    @property
    def success_count(self) -> int:
        """Number of successfully deployed instances."""
        return len(self.deployed)

    @property
    def all_succeeded(self) -> bool:
        """Whether all requested instances were deployed."""
        return self.success_count == self.requested_count


class InstanceManager:
    """Manages complete instance lifecycle via PyLet and scheduler.

    This class orchestrates:
    1. Deployment via PyLet
    2. Health check waiting
    3. Registration with scheduler
    4. Graceful termination with drain-then-remove

    Attributes:
        pylet_client: Client for PyLet operations.
        scheduler_client: Client for scheduler operations.
        instances: Dict of managed instances by pylet_id.
    """

    def __init__(
        self,
        pylet_head_url: str,
        scheduler_url: str,
        custom_command: str | None = None,
        reuse_cluster: bool = False,
    ):
        """Initialize instance manager.

        Args:
            pylet_head_url: URL of PyLet head node.
            scheduler_url: URL of default scheduler (fallback).
            custom_command: Optional custom command template for PyLet.
            reuse_cluster: If True, reuse existing PyLet connection.
        """
        self.pylet_client = PyLetClient(
            head_url=pylet_head_url,
            custom_command=custom_command,
            reuse_cluster=reuse_cluster,
        )
        self._default_scheduler_url = scheduler_url
        self.scheduler_client = SchedulerClient(scheduler_url)
        self._scheduler_clients: dict[str, SchedulerClient] = {}
        self._instances: dict[str, ManagedInstance] = {}
        self._initialized = False

    def init(self) -> None:
        """Initialize connections to PyLet and scheduler.

        Raises:
            RuntimeError: If PyLet is unavailable. Default scheduler health
                check is skipped if scheduler registry has entries (PYLET-024).
        """
        if self._initialized:
            return

        # Initialize PyLet client
        self.pylet_client.init()

        # Verify default scheduler is available (only if no registry entries)
        registry = get_scheduler_registry()
        if len(registry) == 0:
            if not self.scheduler_client.health_check():
                raise RuntimeError(
                    f"Scheduler not available at "
                    f"{self.scheduler_client.scheduler_url}"
                )
        else:
            logger.info(
                f"Scheduler registry has {len(registry)} entries, "
                f"skipping default scheduler health check"
            )

        self._initialized = True
        logger.info("InstanceManager initialized")

    def _get_scheduler_client(self, model_id: str) -> SchedulerClient:
        """Get the scheduler client for a specific model.

        Looks up the scheduler URL from the registry. Falls back to the
        default scheduler client if no registry entry exists.

        Args:
            model_id: Model identifier.

        Returns:
            SchedulerClient for the model's scheduler.
        """
        registry = get_scheduler_registry()
        scheduler_url = registry.get_scheduler_url(model_id)

        if scheduler_url is None:
            # No registry entry, use default
            return self.scheduler_client

        # Use cached client or create new one
        if scheduler_url not in self._scheduler_clients:
            logger.info(
                f"Creating scheduler client for {model_id} "
                f"at {scheduler_url}"
            )
            self._scheduler_clients[scheduler_url] = SchedulerClient(
                scheduler_url
            )

        return self._scheduler_clients[scheduler_url]

    def close(self) -> None:
        """Close client connections."""
        self.scheduler_client.close()
        for client in self._scheduler_clients.values():
            client.close()
        self._scheduler_clients.clear()

    @property
    def instances(self) -> dict[str, ManagedInstance]:
        """Get all managed instances."""
        return self._instances.copy()

    def deploy_instance(
        self,
        model_id: str,
        backend: str = "vllm",
        gpu_count: int = 1,
        instance_id: str | None = None,
        target_worker: str | None = None,
        env: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
    ) -> ManagedInstance:
        """Deploy a single model instance.

        Submits to PyLet but does not wait for running state.
        Use wait_instance_ready() to wait for the instance to be ready.

        Args:
            model_id: Model identifier.
            backend: Model backend ("vllm" or "sglang").
            gpu_count: Number of GPUs.
            instance_id: Custom instance ID for scheduler registration.
            target_worker: Target specific worker.
            env: Additional environment variables.
            labels: Additional labels.

        Returns:
            ManagedInstance in DEPLOYING status.

        Raises:
            RuntimeError: If not initialized.
        """
        self._ensure_initialized()

        # Deploy via PyLet (single instance)
        # deploy_model always returns a list now
        infos = self.pylet_client.deploy_model(
            model_id=model_id,
            backend=backend,
            gpu_count=gpu_count,
            env=env,
            labels=labels,
            target_worker=target_worker,
            count=1,
        )
        info = infos[0]  # Get the single deployed instance

        # Use PyLet ID as scheduler instance ID if not provided
        scheduler_instance_id = instance_id or info.pylet_id

        # Create managed instance
        managed = ManagedInstance(
            pylet_id=info.pylet_id,
            instance_id=scheduler_instance_id,
            model_id=model_id,
            backend=backend,
            gpu_count=gpu_count,
            status=ManagedInstanceStatus.DEPLOYING,
        )

        self._instances[info.pylet_id] = managed
        logger.info(
            f"Deployed instance {info.pylet_id} for {model_id} "
            f"(scheduler_id={scheduler_instance_id})"
        )

        return managed

    def deploy_instances(
        self,
        model_id: str,
        count: int,
        backend: str = "vllm",
        gpu_count: int = 1,
        target_worker: str | None = None,
    ) -> DeploymentResult:
        """Deploy multiple model instances.

        Deploys instances one at a time via PyLet's submit API.

        Args:
            model_id: Model identifier.
            count: Number of instances to deploy.
            backend: Model backend.
            gpu_count: GPUs per instance.
            target_worker: Target worker (all instances go to same worker).

        Returns:
            DeploymentResult with deployed and failed instances.
        """
        if count <= 0:
            return DeploymentResult(
                model_id=model_id,
                requested_count=count,
                deployed=[],
                failed=[],
            )

        # Log instance deployment start
        logger.info(
            f"[INSTANCE_DEPLOY] model_id={model_id} count={count} "
            f"backend={backend} gpu_count={gpu_count}"
        )

        deployed = []
        failed = []

        try:
            # Deploy multiple instances (planner manages replicas now)
            infos = self.pylet_client.deploy_model(
                model_id=model_id,
                backend=backend,
                gpu_count=gpu_count,
                target_worker=target_worker,
                count=count,
            )

            # deploy_model always returns list now
            for info in infos:
                managed = ManagedInstance(
                    pylet_id=info.pylet_id,
                    instance_id=info.pylet_id,
                    model_id=model_id,
                    backend=backend,
                    gpu_count=gpu_count,
                    status=ManagedInstanceStatus.DEPLOYING,
                )
                self._instances[info.pylet_id] = managed
                deployed.append(managed)

            # Log PyLet response
            logger.debug(
                f"[INSTANCE_PYLET_RESPONSE] model_id={model_id} "
                f"pylet_ids={[i.pylet_id for i in deployed]} count={len(deployed)}"
            )

        except PartialDeploymentError as e:
            # Handle partial success - some instances deployed, some failed
            for info in e.result.succeeded:
                managed = ManagedInstance(
                    pylet_id=info.pylet_id,
                    instance_id=info.pylet_id,
                    model_id=model_id,
                    backend=backend,
                    gpu_count=gpu_count,
                    status=ManagedInstanceStatus.DEPLOYING,
                )
                self._instances[info.pylet_id] = managed
                deployed.append(managed)

            for idx, error in e.result.failed:
                failed.append((f"{model_id}-{idx}", error))

            logger.warning(
                f"[INSTANCE_DEPLOY] partial success for {model_id}: "
                f"{len(deployed)} deployed, {len(failed)} failed"
            )

        except Exception as e:
            # If all deployments failed, record all as failed
            for i in range(count):
                failed.append((f"{model_id}-{i}", str(e)))
            logger.error(f"All deployments failed for {model_id}: {e}")

        return DeploymentResult(
            model_id=model_id,
            requested_count=count,
            deployed=deployed,
            failed=failed,
        )

    def wait_instance_ready(
        self,
        pylet_id: str,
        timeout: float = 300.0,
        register: bool = True,
    ) -> ManagedInstance:
        """Wait for instance to be ready and optionally register with scheduler.

        Args:
            pylet_id: PyLet instance UUID.
            timeout: Maximum wait time for running state.
            register: Whether to register with scheduler after ready.

        Returns:
            Updated ManagedInstance.

        Raises:
            KeyError: If instance not found.
            TimeoutError: If instance doesn't become ready.
        """
        self._ensure_initialized()

        if pylet_id not in self._instances:
            raise KeyError(f"Instance not found: {pylet_id}")

        managed = self._instances[pylet_id]

        try:
            # Wait for PyLet instance to be running
            old_status = managed.status
            managed.status = ManagedInstanceStatus.WAITING_HEALTH
            logger.debug(
                f"[INSTANCE_STATUS] pylet_id={pylet_id} "
                f"old_status={old_status} new_status={managed.status}"
            )

            info = self.pylet_client.wait_instance_running(pylet_id, timeout=timeout)

            managed.endpoint = info.endpoint
            logger.debug(
                f"[INSTANCE_HEALTH] pylet_id={pylet_id} "
                f"status=running endpoint={info.endpoint}"
            )

            # Register with scheduler if requested
            if register and managed.endpoint:
                managed.status = ManagedInstanceStatus.REGISTERING
                scheduler = self._get_scheduler_client(managed.model_id)
                result = scheduler.register_instance(
                    instance_id=managed.instance_id,
                    model_id=managed.model_id,
                    endpoint=managed.endpoint,
                    backend=managed.backend,
                )

                if result.success:
                    old_status = managed.status
                    managed.status = ManagedInstanceStatus.ACTIVE
                    # Log registration success
                    logger.info(
                        f"[INSTANCE_REGISTER] pylet_id={pylet_id} "
                        f"instance_id={managed.instance_id} model_id={managed.model_id} "
                        f"endpoint={managed.endpoint} registered=true"
                    )
                    logger.debug(
                        f"[INSTANCE_STATUS] pylet_id={pylet_id} "
                        f"old_status={old_status} new_status={managed.status}"
                    )
                else:
                    old_status = managed.status
                    managed.status = ManagedInstanceStatus.FAILED
                    managed.error = result.message
                    # Log registration failure
                    logger.warning(
                        f"[INSTANCE_REGISTER] pylet_id={pylet_id} "
                        f"instance_id={managed.instance_id} registered=false "
                        f"error={result.message}"
                    )
            else:
                managed.status = ManagedInstanceStatus.ACTIVE

        except Exception as e:
            managed.status = ManagedInstanceStatus.FAILED
            managed.error = str(e)
            logger.error(f"Instance {pylet_id} failed: {e}")
            raise

        return managed

    def wait_instances_ready(
        self,
        pylet_ids: list[str],
        timeout: float = 300.0,
        register: bool = True,
    ) -> list[ManagedInstance]:
        """Wait for multiple instances to be ready.

        Args:
            pylet_ids: List of PyLet instance UUIDs.
            timeout: Timeout per instance.
            register: Whether to register with scheduler.

        Returns:
            List of ready ManagedInstances (failed ones have FAILED status).
        """
        results = []
        for pylet_id in pylet_ids:
            try:
                instance = self.wait_instance_ready(
                    pylet_id, timeout=timeout, register=register
                )
                results.append(instance)
            except Exception as e:
                # Instance already marked as failed in wait_instance_ready
                if pylet_id in self._instances:
                    results.append(self._instances[pylet_id])
                logger.error(f"Instance {pylet_id} failed during wait: {e}")
        return results

    def terminate_instance(
        self,
        pylet_id: str,
        wait_for_drain: bool = True,
        drain_timeout: float = 30.0,
    ) -> bool:
        """Terminate a single instance.

        Performs safe termination:
        1. Deregister from scheduler (drain + remove)
        2. Cancel PyLet instance

        Args:
            pylet_id: PyLet instance UUID.
            wait_for_drain: Whether to wait for scheduler drain.
            drain_timeout: Drain wait timeout.

        Returns:
            True if terminated successfully.
        """
        self._ensure_initialized()

        if pylet_id not in self._instances:
            logger.warning(f"Instance {pylet_id} not in managed instances")
            return False

        managed = self._instances[pylet_id]

        # Log termination start
        logger.info(
            f"[INSTANCE_TERMINATE] pylet_id={pylet_id} "
            f"instance_id={managed.instance_id} model_id={managed.model_id} "
            f"action=start"
        )

        try:
            old_status = managed.status
            managed.status = ManagedInstanceStatus.DRAINING
            logger.debug(
                f"[INSTANCE_STATUS] pylet_id={pylet_id} "
                f"old_status={old_status} new_status={managed.status}"
            )

            # Deregister from scheduler (drain + remove)
            scheduler = self._get_scheduler_client(managed.model_id)
            scheduler.deregister_instance(
                instance_id=managed.instance_id,
                wait_for_drain=wait_for_drain,
                drain_timeout=drain_timeout,
            )

            old_status = managed.status
            managed.status = ManagedInstanceStatus.TERMINATING
            logger.debug(
                f"[INSTANCE_STATUS] pylet_id={pylet_id} "
                f"old_status={old_status} new_status={managed.status}"
            )

            # Cancel PyLet instance
            self.pylet_client.cancel_instance(pylet_id, delete=True)

            old_status = managed.status
            managed.status = ManagedInstanceStatus.TERMINATED

            # Remove from managed instances so idle checks are accurate
            del self._instances[pylet_id]

            # Log termination complete
            logger.info(
                f"[INSTANCE_TERMINATE] pylet_id={pylet_id} action=complete success=true"
            )
            return True

        except Exception as e:
            # Log termination failure
            logger.warning(
                f"[INSTANCE_TERMINATE] pylet_id={pylet_id} "
                f"action=complete success=false error={str(e)}"
            )
            managed.error = str(e)
            return False

    def terminate_instances(
        self,
        pylet_ids: list[str],
        wait_for_drain: bool = True,
    ) -> dict[str, bool]:
        """Terminate multiple instances.

        Args:
            pylet_ids: List of PyLet instance UUIDs.
            wait_for_drain: Whether to wait for scheduler drain.

        Returns:
            Dict mapping pylet_id to termination success.
        """
        results = {}
        for pylet_id in pylet_ids:
            results[pylet_id] = self.terminate_instance(
                pylet_id, wait_for_drain=wait_for_drain
            )
        return results

    def terminate_all(self, wait_for_drain: bool = False) -> dict[str, bool]:
        """Terminate all managed instances.

        Args:
            wait_for_drain: Whether to wait for scheduler drain.

        Returns:
            Dict mapping pylet_id to termination success.
        """
        return self.terminate_instances(
            list(self._instances.keys()),
            wait_for_drain=wait_for_drain,
        )

    def get_instance(self, pylet_id: str) -> ManagedInstance | None:
        """Get a managed instance by PyLet ID.

        Args:
            pylet_id: PyLet instance UUID.

        Returns:
            ManagedInstance or None if not found.
        """
        return self._instances.get(pylet_id)

    def get_active_instances(
        self, model_id: str | None = None
    ) -> list[ManagedInstance]:
        """Get all active instances.

        Args:
            model_id: Optional filter by model ID.

        Returns:
            List of active ManagedInstances.
        """
        instances = [
            i
            for i in self._instances.values()
            if i.status == ManagedInstanceStatus.ACTIVE
        ]

        if model_id:
            instances = [i for i in instances if i.model_id == model_id]

        return instances

    def get_instances_by_model(self, model_id: str) -> list[ManagedInstance]:
        """Get all instances for a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of ManagedInstances for the model.
        """
        return [i for i in self._instances.values() if i.model_id == model_id]

    def refresh_instance(self, pylet_id: str) -> ManagedInstance | None:
        """Refresh instance state from PyLet.

        Args:
            pylet_id: PyLet instance UUID.

        Returns:
            Updated ManagedInstance or None if not found.
        """
        if pylet_id not in self._instances:
            return None

        managed = self._instances[pylet_id]
        info = self.pylet_client.get_instance(pylet_id)

        if info:
            managed.endpoint = info.endpoint
            # Update status based on PyLet status
            if info.status in ("COMPLETED", "FAILED", "CANCELLED"):
                managed.status = ManagedInstanceStatus.TERMINATED
            elif info.status == "RUNNING" and managed.is_active:
                pass  # Keep active status
        else:
            managed.status = ManagedInstanceStatus.TERMINATED

        return managed

    def _ensure_initialized(self) -> None:
        """Ensure manager is initialized."""
        if not self._initialized:
            raise RuntimeError("InstanceManager not initialized. Call init() first.")


# Global manager singleton
_instance_manager: InstanceManager | None = None


def create_instance_manager(
    pylet_head_url: str,
    scheduler_url: str,
) -> InstanceManager:
    """Create and initialize the global instance manager.

    Args:
        pylet_head_url: URL of PyLet head node.
        scheduler_url: URL of scheduler.

    Returns:
        Initialized InstanceManager.
    """
    global _instance_manager

    _instance_manager = InstanceManager(pylet_head_url, scheduler_url)
    _instance_manager.init()
    return _instance_manager


def get_instance_manager() -> InstanceManager:
    """Get the global instance manager.

    Returns:
        The global InstanceManager instance.

    Raises:
        RuntimeError: If manager not created.
    """
    if _instance_manager is None:
        raise RuntimeError(
            "InstanceManager not created. " "Call create_instance_manager() first."
        )
    return _instance_manager
