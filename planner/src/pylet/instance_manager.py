"""Instance lifecycle manager (PYLET-008).

This module provides the InstanceManager class that orchestrates the complete
instance lifecycle: deploy via PyLet, wait for health, register with scheduler,
and cleanup on termination.

Example:
    from src.pylet.instance_manager import InstanceManager

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

from src.pylet.client import PyLetClient
from src.pylet.scheduler_client import SchedulerClient


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
            scheduler_url: URL of scheduler.
            custom_command: Optional custom command template for PyLet.
            reuse_cluster: If True, reuse existing PyLet connection.
        """
        self.pylet_client = PyLetClient(
            head_url=pylet_head_url,
            custom_command=custom_command,
            reuse_cluster=reuse_cluster,
        )
        self.scheduler_client = SchedulerClient(scheduler_url)
        self._instances: dict[str, ManagedInstance] = {}
        self._initialized = False

    def init(self) -> None:
        """Initialize connections to PyLet and scheduler.

        Raises:
            RuntimeError: If PyLet or scheduler is unavailable.
        """
        if self._initialized:
            return

        # Initialize PyLet client
        self.pylet_client.init()

        # Verify scheduler is available
        if not self.scheduler_client.health_check():
            raise RuntimeError(
                f"Scheduler not available at " f"{self.scheduler_client.scheduler_url}"
            )

        self._initialized = True
        logger.info("InstanceManager initialized")

    def close(self) -> None:
        """Close client connections."""
        self.scheduler_client.close()

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

        # Deploy via PyLet (single replica)
        info = self.pylet_client.deploy_model(
            model_id=model_id,
            backend=backend,
            gpu_count=gpu_count,
            env=env,
            labels=labels,
            target_worker=target_worker,
            replicas=1,
        )

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
        """Deploy multiple model instances using PyLet's replicas feature.

        Uses PyLet's native batch deployment for efficiency instead of
        deploying one instance at a time.

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

        deployed = []
        failed = []

        try:
            # Use PyLet's replicas feature for batch deployment
            result = self.pylet_client.deploy_model(
                model_id=model_id,
                backend=backend,
                gpu_count=gpu_count,
                target_worker=target_worker,
                replicas=count,
            )

            # Handle both single instance and list return types
            infos = result if isinstance(result, list) else [result]

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

            logger.info(
                f"Deployed {len(deployed)} instances for {model_id} "
                f"using replicas={count}"
            )

        except Exception as e:
            # If batch deployment fails entirely, record all as failed
            for i in range(count):
                failed.append((f"instance-{i}", str(e)))
            logger.error(f"Batch deployment failed for {model_id}: {e}")

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
            managed.status = ManagedInstanceStatus.WAITING_HEALTH
            info = self.pylet_client.wait_instance_running(pylet_id, timeout=timeout)

            managed.endpoint = info.endpoint
            logger.info(f"Instance {pylet_id} running at {info.endpoint}")

            # Register with scheduler if requested
            if register and managed.endpoint:
                managed.status = ManagedInstanceStatus.REGISTERING
                result = self.scheduler_client.register_instance(
                    instance_id=managed.instance_id,
                    model_id=managed.model_id,
                    endpoint=managed.endpoint,
                    backend=managed.backend,
                )

                if result.success:
                    managed.status = ManagedInstanceStatus.ACTIVE
                    logger.info(f"Instance {managed.instance_id} registered and active")
                else:
                    managed.status = ManagedInstanceStatus.FAILED
                    managed.error = result.message
                    logger.error(
                        f"Failed to register instance {managed.instance_id}: "
                        f"{result.message}"
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

        try:
            managed.status = ManagedInstanceStatus.DRAINING

            # Deregister from scheduler (drain + remove)
            self.scheduler_client.deregister_instance(
                instance_id=managed.instance_id,
                wait_for_drain=wait_for_drain,
                drain_timeout=drain_timeout,
            )

            managed.status = ManagedInstanceStatus.TERMINATING

            # Cancel PyLet instance
            self.pylet_client.cancel_instance(pylet_id, delete=True)

            managed.status = ManagedInstanceStatus.TERMINATED
            logger.info(f"Instance {pylet_id} terminated")
            return True

        except Exception as e:
            logger.error(f"Failed to terminate instance {pylet_id}: {e}")
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
