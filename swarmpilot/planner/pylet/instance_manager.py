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

import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from swarmpilot.engines import get_adapter
from swarmpilot.engines.live import probe_platform_info
from swarmpilot.errors import PartialDeploymentError
from swarmpilot.planner.pylet.client import InstanceInfo, PyLetClient
from swarmpilot.planner.pylet.hardware import HardwareResolver, matches
from swarmpilot.planner.pylet.scheduler_client import (
    SchedulerClient,
    default_platform_info,
)
from swarmpilot.planner.scheduler_registry import get_scheduler_registry

# PyLet states in which the engine process is no longer running.
_ENGINE_GONE = frozenset({"COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"})


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
        hardware: Hardware name of that worker, once known.
        error: Error message if failed.
        created_at: Creation timestamp.
        active_since: When it started serving traffic, if it has.
    """

    pylet_id: str
    instance_id: str  # For scheduler registration
    model_id: str
    endpoint: str | None = None
    backend: str = "vllm"
    status: ManagedInstanceStatus = ManagedInstanceStatus.DEPLOYING
    gpu_count: int = 1
    worker_id: str | None = None
    hardware: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    active_since: float | None = None

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
        engine_ready_timeout: float | None = None,
        hardware_name: str | None = None,
        hardware: HardwareResolver | None = None,
        whole_nodes: bool | None = None,
    ):
        """Initialize instance manager.

        Args:
            pylet_head_url: URL of PyLet head node.
            scheduler_url: URL of default scheduler (fallback).
            custom_command: Optional custom command template for PyLet.
            reuse_cluster: If True, reuse existing PyLet connection.
            engine_ready_timeout: Seconds to wait for a new engine's health
                path before registering it (env
                ``PLANNER_ENGINE_READY_TIMEOUT``, default 600).
            hardware_name: Hardware label for workers the hardware map
                does not cover (env ``PLANNER_HARDWARE_NAME``, default
                "gpu").
            hardware: Resolves each worker's hardware name; built from
                ``PLANNER_HARDWARE_MAP`` if None.
            whole_nodes: Give each instance a whole worker (env
                ``PLANNER_WHOLE_NODES``, default false): at most one
                instance per worker, holding all its GPUs, and a model's
                GPU count is the fewest a worker must have.
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
        self._custom_command = custom_command
        self._engine_ready_timeout = (
            engine_ready_timeout
            if engine_ready_timeout is not None
            else float(os.getenv("PLANNER_ENGINE_READY_TIMEOUT", "600"))
        )
        self.hardware = hardware or HardwareResolver(default=hardware_name)
        if whole_nodes is None:
            whole_nodes = os.getenv("PLANNER_WHOLE_NODES", "false").lower() in (
                "1",
                "true",
                "yes",
            )
        self.whole_nodes = whole_nodes
        # The instance watcher removes instances from another thread.
        self._lock = threading.RLock()

    def init(self) -> None:
        """Initialize connection to PyLet.

        Raises:
            RuntimeError: If PyLet is unavailable.
        """
        if self._initialized:
            return

        # Initialize PyLet client
        self.pylet_client.init()

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

        # Use cached client or create new one. Removals drain in parallel
        # threads, so creating one is under the lock.
        with self._lock:
            if scheduler_url not in self._scheduler_clients:
                logger.info(
                    f"Creating scheduler client for {model_id} at "
                    f"{scheduler_url}"
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
        with self._lock:
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
        extra_args: str | None = None,
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
            extra_args: Additional CLI arguments for the launch command.

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
            extra_args=extra_args,
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

        with self._lock:
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
        extra_args: str | None = None,
        hardware: str | None = None,
    ) -> DeploymentResult:
        """Deploy multiple model instances.

        Deploys instances one at a time via PyLet's submit API.

        Args:
            model_id: Model identifier.
            count: Number of instances to deploy.
            backend: Model backend.
            gpu_count: GPUs per instance.
            target_worker: Target worker (all instances go to same worker).
            extra_args: Additional CLI arguments for the launch command.
            hardware: Only place instances on workers whose hardware name
                matches this pattern (any case, e.g. ``"H100*"``).
                Replicas that do not fit are reported as failed. Ignored
                when ``target_worker`` is set.

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

        deployed: list[ManagedInstance] = []
        failed: list[tuple[str, str]] = []

        if self.whole_nodes:
            nodes = self._place_nodes(hardware, gpu_count, count, target_worker)
            for i in range(len(nodes), count):
                where = f" of hardware {hardware!r}" if hardware else ""
                failed.append(
                    (
                        f"{model_id}-{i}",
                        f"no free worker with {gpu_count} or more GPU(s)"
                        + (f" ({target_worker})" if target_worker else where),
                    )
                )
            if failed:
                logger.warning(
                    f"[INSTANCE_DEPLOY] only {len(nodes)} of {count} "
                    f"replicas of {model_id} got a free worker"
                )
            for worker, gpus in nodes:
                self._submit_group(
                    model_id,
                    1,
                    backend,
                    gpus,
                    worker,
                    extra_args,
                    deployed,
                    failed,
                )
            return DeploymentResult(
                model_id=model_id,
                requested_count=count,
                deployed=deployed,
                failed=failed,
            )

        if hardware and target_worker is None:
            groups = self._place(hardware, gpu_count, count)
            placed = sum(groups.values())
            for i in range(placed, count):
                failed.append(
                    (
                        f"{model_id}-{i}",
                        f"no worker with {gpu_count} free GPU(s) of "
                        f"hardware {hardware!r}",
                    )
                )
            if failed:
                logger.warning(
                    f"[INSTANCE_DEPLOY] only {placed} of {count} replicas of "
                    f"{model_id} fit on {hardware!r} workers"
                )
        elif target_worker is None:
            # Best fit on any worker; what does not fit waits in PyLet's
            # queue as before.
            try:
                groups = self._place(None, gpu_count, count)
            except Exception as e:
                logger.debug(f"Could not place {model_id} replicas: {e}")
                groups = {}
            rest = count - sum(groups.values())
            if rest > 0:
                groups[None] = rest
        else:
            groups = {target_worker: count}

        for worker, n in groups.items():
            self._submit_group(
                model_id,
                n,
                backend,
                gpu_count,
                worker,
                extra_args,
                deployed,
                failed,
            )

        return DeploymentResult(
            model_id=model_id,
            requested_count=count,
            deployed=deployed,
            failed=failed,
        )

    def _submit_group(
        self,
        model_id: str,
        count: int,
        backend: str,
        gpu_count: int,
        target_worker: str | None,
        extra_args: str | None,
        deployed: list[ManagedInstance],
        failed: list[tuple[str, str]],
    ) -> None:
        """Submit ``count`` replicas to PyLet, recording the outcome."""
        offset = len(deployed) + len(failed)

        def track(info: InstanceInfo) -> None:
            managed = ManagedInstance(
                pylet_id=info.pylet_id,
                instance_id=info.pylet_id,
                model_id=model_id,
                backend=backend,
                gpu_count=gpu_count,
                worker_id=target_worker,
                status=ManagedInstanceStatus.DEPLOYING,
            )
            with self._lock:
                self._instances[info.pylet_id] = managed
            deployed.append(managed)

        try:
            # Deploy multiple instances (planner manages replicas now)
            infos = self.pylet_client.deploy_model(
                model_id=model_id,
                backend=backend,
                gpu_count=gpu_count,
                target_worker=target_worker,
                count=count,
                extra_args=extra_args,
            )
            for info in infos:
                track(info)

            # Log PyLet response
            logger.debug(
                f"[INSTANCE_PYLET_RESPONSE] model_id={model_id} "
                f"pylet_ids={[i.pylet_id for i in infos]} count={len(infos)}"
            )

        except PartialDeploymentError as e:
            # Handle partial success - some instances deployed, some failed
            for info in e.succeeded:
                track(info)
            for idx, error in e.failed:
                failed.append((f"{model_id}-{offset + idx}", error))

            logger.warning(
                f"[INSTANCE_DEPLOY] partial success for {model_id}: "
                f"{len(e.succeeded)} deployed, {len(e.failed)} failed"
            )

        except Exception as e:
            # If all deployments failed, record all as failed
            for i in range(count):
                failed.append((f"{model_id}-{offset + i}", str(e)))
            logger.opt(exception=True).error(
                f"All deployments failed for {model_id}: {e}"
            )

    def _place(
        self, hardware: str | None, gpu_count: int, count: int
    ) -> dict[str, int]:
        """Pick workers for ``count`` replicas, best fit first.

        Each replica goes to the matching online worker with the fewest
        free GPUs that still hold it, so small instances fill small
        nodes and leave the large ones whole for instances that need
        several GPUs (PyLet itself takes the first worker that fits,
        whatever its size). GPUs held by instances PyLet has not placed
        yet are not visible here, so a later submit can still wait in
        PyLet's queue.

        Args:
            hardware: Hardware name pattern, or None for any worker.
            gpu_count: GPUs per replica.
            count: Replicas wanted.

        Returns:
            Worker id to replica count; fewer than ``count`` in total when
            the matching workers are full.
        """
        workers = [
            w
            for w in self.pylet_client.list_workers()
            if getattr(w, "status", "ONLINE") == "ONLINE"
        ]
        self.hardware.observe_workers(workers)
        # PyLet's ``gpu_available`` is the count at registration; the free
        # GPU indices are computed from current allocations.
        free = {
            w.id: len(w.gpu_indices_available)
            for w in workers
            if hardware is None
            or matches(self.hardware.resolve(w.id, w.host), hardware)
        }
        order = {wid: k for k, wid in enumerate(free)}
        per = max(1, gpu_count)
        groups: Counter[str] = Counter()
        for _ in range(count):
            fits = [wid for wid, n in free.items() if n >= per]
            if not fits:
                break
            best = min(fits, key=lambda k: (free[k], order[k]))
            free[best] -= per
            groups[best] += 1
        return dict(groups)

    def free_nodes(self, hardware: str | None = None) -> list[tuple]:
        """Online workers with every GPU free and no instance of ours.

        An instance PyLet has not placed yet still holds its worker,
        so two deploys never pick the same one.

        Args:
            hardware: Hardware name pattern, or None for any worker.

        Returns:
            (worker id, hardware name, GPUs) per free worker, in PyLet's
            order.
        """
        workers = [
            w
            for w in self.pylet_client.list_workers()
            if getattr(w, "status", "ONLINE") == "ONLINE"
        ]
        self.hardware.observe_workers(workers)
        with self._lock:
            held = {
                m.worker_id
                for m in self._instances.values()
                if m.worker_id is not None and not m.is_terminal
            }
        out = []
        for w in workers:
            total = int(getattr(w, "gpu", 0) or 0)
            if w.id in held or total <= 0:
                continue
            if len(w.gpu_indices_available) < total:
                continue
            hw = self.hardware.resolve(w.id, getattr(w, "host", None))
            if hardware is None or matches(hw, hardware):
                out.append((w.id, hw, total))
        return out

    def _place_nodes(
        self,
        hardware: str | None,
        gpu_count: int,
        count: int,
        target_worker: str | None = None,
    ) -> list[tuple[str, int]]:
        """Whole free workers for ``count`` replicas, smallest first.

        Each replica takes a worker with at least ``gpu_count`` GPUs and
        all of them, so small models take small workers and the large
        ones stay free for the models that need them.

        Args:
            hardware: Hardware name pattern, or None for any worker.
                Ignored when ``target_worker`` is set.
            gpu_count: GPUs a worker must have.
            count: Replicas wanted.
            target_worker: The only worker to use, if set.

        Returns:
            (worker id, its GPU count) per replica placed; fewer than
            ``count`` when not enough workers are free.
        """
        nodes = self.free_nodes(None if target_worker else hardware)
        if target_worker is not None:
            nodes = [n for n in nodes if n[0] == target_worker]
        need = max(1, gpu_count)
        order = {n[0]: k for k, n in enumerate(nodes)}
        fits = sorted(
            (n for n in nodes if n[2] >= need),
            key=lambda n: (n[2], order[n[0]]),
        )
        return [(wid, gpus) for wid, _, gpus in fits[:count]]

    def free_gpus(self) -> int:
        """GPUs no instance holds on the online workers."""
        workers = [
            w
            for w in self.pylet_client.list_workers()
            if getattr(w, "status", "ONLINE") == "ONLINE"
        ]
        return sum(len(w.gpu_indices_available) for w in workers)

    def free_slots(self, gpu_count: int = 1) -> list[tuple[str, str]]:
        """Free places for one instance on the online workers.

        Args:
            gpu_count: GPUs one instance takes.

        Returns:
            One (worker id, hardware name) pair per place.
        """
        workers = [
            w
            for w in self.pylet_client.list_workers()
            if getattr(w, "status", "ONLINE") == "ONLINE"
        ]
        self.hardware.observe_workers(workers)
        if self.whole_nodes:
            # A worker is free only as a whole; one entry per GPU, so the
            # caller sees its size.
            return [
                (wid, hw)
                for wid, hw, gpus in self.free_nodes()
                for _ in range(gpus // max(1, gpu_count))
            ]
        per = max(1, gpu_count)
        return [
            (w.id, self.hardware.resolve(w.id, getattr(w, "host", None)))
            for w in workers
            for _ in range(len(w.gpu_indices_available) // per)
        ]

    def _resolve_hardware(
        self, managed: ManagedInstance, info: InstanceInfo
    ) -> str:
        """Find the hardware of the worker an instance landed on."""
        worker = info.worker_id or managed.worker_id
        host = None
        if info.endpoint and "://" not in info.endpoint:
            host = info.endpoint.rsplit(":", 1)[0]
        if self.hardware.heterogeneous and worker and host is None:
            try:
                self.hardware.observe_workers(self.pylet_client.list_workers())
            except Exception as e:
                logger.debug(f"Could not list PyLet workers: {e}")
        managed.worker_id = worker
        return self.hardware.resolve(worker, host)

    def _platform_info(self, managed: ManagedInstance) -> dict[str, str]:
        """Wait for the engine to be healthy and read its live version.

        Instances started from a custom command are not probed, because
        their adapter's health and version routes may not apply.
        """
        hardware = managed.hardware or self.hardware.default
        if self._custom_command or not managed.endpoint:
            return default_platform_info(managed.backend, hardware)
        try:
            adapter = get_adapter(managed.backend)
        except KeyError:
            return default_platform_info(managed.backend, hardware)
        info = probe_platform_info(
            adapter,
            managed.endpoint,
            hardware_name=hardware,
            ready_timeout_s=self._engine_ready_timeout,
        )
        logger.info(
            f"[INSTANCE_PLATFORM] pylet_id={managed.pylet_id} "
            f"platform_info={info}"
        )
        return info

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

            info = self.pylet_client.wait_instance_running(
                pylet_id, timeout=timeout
            )

            managed.endpoint = info.endpoint
            managed.hardware = self._resolve_hardware(managed, info)
            logger.debug(
                f"[INSTANCE_HEALTH] pylet_id={pylet_id} "
                f"status=running endpoint={info.endpoint} "
                f"worker={managed.worker_id} hardware={managed.hardware}"
            )

            # Register with scheduler if requested
            if register and managed.endpoint:
                platform_info = self._platform_info(managed)
                managed.status = ManagedInstanceStatus.REGISTERING
                scheduler = self._get_scheduler_client(managed.model_id)
                result = scheduler.register_instance(
                    instance_id=managed.instance_id,
                    model_id=managed.model_id,
                    endpoint=managed.endpoint,
                    backend=managed.backend,
                    platform_info=platform_info,
                )

                if result.success:
                    old_status = managed.status
                    managed.status = ManagedInstanceStatus.ACTIVE
                    managed.active_since = time.time()
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
                managed.active_since = time.time()

        except Exception as e:
            managed.status = ManagedInstanceStatus.FAILED
            managed.error = str(e)
            logger.opt(exception=True).error(f"Instance {pylet_id} failed: {e}")
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
                logger.opt(exception=True).error(
                    f"Instance {pylet_id} failed during wait: {e}"
                )
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
            with self._lock:
                self._instances.pop(pylet_id, None)

            # Log termination complete
            logger.info(
                f"[INSTANCE_TERMINATE] pylet_id={pylet_id} action=complete success=true"
            )
            return True

        except Exception as e:
            # Log termination failure
            logger.opt(exception=True).warning(
                f"[INSTANCE_TERMINATE] pylet_id={pylet_id} "
                f"action=complete success=false error={e!s}"
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
        with self._lock:
            pylet_ids = list(self._instances)
        return self.terminate_instances(
            pylet_ids,
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
        with self._lock:
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
        with self._lock:
            return [
                i for i in self._instances.values() if i.model_id == model_id
            ]

    def reap_failed(self) -> list[ManagedInstance]:
        """Drop active instances whose engine is gone.

        PyLet marks an instance FAILED (or COMPLETED) when its process
        exits, for example when the engine crashes, and forgets it if its
        worker is deleted. Such an instance is removed from its Scheduler
        at once, without draining (nothing can finish on a dead engine;
        the Scheduler re-routes what was queued on it), its PyLet record
        is deleted to free the GPUs, and it is dropped from the managed
        set. Instances that are still starting, draining or terminating
        are left to the code handling them.

        Returns:
            The removed instances, with status FAILED and an error.
        """
        self._ensure_initialized()
        with self._lock:
            active = [
                m
                for m in self._instances.values()
                if m.status == ManagedInstanceStatus.ACTIVE
            ]
        if not active:
            return []
        seen = {i.pylet_id: i for i in self.pylet_client.list_model_instances()}
        reaped = []
        for managed in active:
            info = seen.get(managed.pylet_id)
            if info is not None and info.status not in _ENGINE_GONE:
                continue
            state = info.status if info is not None else "missing"
            with self._lock:
                if managed.status != ManagedInstanceStatus.ACTIVE:
                    continue  # a terminate got there first
                managed.status = ManagedInstanceStatus.FAILED
                managed.error = f"engine {state.lower()} in PyLet"
            logger.warning(
                f"[INSTANCE_FAILED] pylet_id={managed.pylet_id} "
                f"model_id={managed.model_id} endpoint={managed.endpoint} "
                f"pylet_status={state}; removing it from the scheduler"
            )
            try:
                # Nothing can drain from a dead engine; the Scheduler
                # re-routes whatever was queued on it.
                scheduler = self._get_scheduler_client(managed.model_id)
                scheduler.remove_instance(managed.instance_id)
            except Exception as e:
                logger.warning(
                    f"Could not deregister failed instance "
                    f"{managed.instance_id}: {e}"
                )
            if info is not None:
                try:
                    self.pylet_client.cancel_instance(
                        managed.pylet_id, delete=True
                    )
                except Exception as e:
                    # Still dropped below, so it is replaced; PyLet keeps
                    # the dead record until someone deletes it.
                    logger.warning(
                        f"Could not delete failed instance "
                        f"{managed.pylet_id} in PyLet: {e}"
                    )
            with self._lock:
                self._instances.pop(managed.pylet_id, None)
            reaped.append(managed)
        return reaped

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
            raise RuntimeError(
                "InstanceManager not initialized. Call init() first."
            )


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
            "InstanceManager not created. Call create_instance_manager() first."
        )
    return _instance_manager
