"""PyLet client wrapper for SwarmPilot planner (PYLET-007).

This module provides a high-level wrapper around the PyLet API for deploying
and managing model instances. It handles PyLet-specific operations while
providing a clean interface for the planner.

Example:
    from src.pylet.client import PyLetClient, create_pylet_client

    client = create_pylet_client("http://localhost:8000")
    instance = client.deploy_model(
        model_id="Qwen/Qwen3-0.6B",
        backend="vllm",
        gpu_count=1,
    )
    print(f"Deployed: {instance.pylet_id} at {instance.endpoint}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
from pylet.errors import NotFoundError, PyletError

import pylet
from pylet import Instance


@dataclass
class InstanceInfo:
    """Information about a deployed model instance.

    Attributes:
        pylet_id: PyLet instance UUID.
        model_id: Model identifier (e.g., "Qwen/Qwen3-0.6B").
        endpoint: HTTP endpoint in "host:port" format, or None if not running.
        status: PyLet instance status (PENDING, RUNNING, COMPLETED, etc.).
        backend: Model backend (vllm, sglang).
        labels: Additional instance labels.
    """

    pylet_id: str
    model_id: str
    endpoint: str | None
    status: str
    backend: str = "vllm"
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_pylet_instance(cls, instance: Instance) -> InstanceInfo:
        """Create InstanceInfo from a PyLet Instance object.

        Args:
            instance: PyLet Instance object.

        Returns:
            InstanceInfo with data from the PyLet instance.
        """
        labels = instance.labels
        return cls(
            pylet_id=instance.id,
            model_id=labels.get("model_id", "unknown"),
            endpoint=instance.endpoint,
            status=instance.status,
            backend=labels.get("backend", "vllm"),
            labels=labels,
        )


# Model launch command templates
# $PORT is replaced by PyLet with the auto-allocated port
MODEL_COMMANDS: dict[str, str] = {
    "vllm": "vllm serve {model_id} --port $PORT --host 0.0.0.0",
    "sglang": (
        "python -m sglang.launch_server "
        "--model {model_id} --port $PORT --host 0.0.0.0"
    ),
}


class PyLetClient:
    """Client for interacting with PyLet to manage model instances.

    This class wraps PyLet operations and provides SwarmPilot-specific
    functionality for model deployment and lifecycle management.

    Attributes:
        head_url: URL of the PyLet head node.
        initialized: Whether PyLet has been initialized.
    """

    def __init__(self, head_url: str):
        """Initialize PyLet client.

        Args:
            head_url: URL of the PyLet head node (e.g., "http://localhost:8000").
        """
        self.head_url = head_url
        self._initialized = False

    def init(self) -> None:
        """Initialize connection to PyLet cluster.

        Raises:
            PyletError: If connection to PyLet fails.
        """
        if self._initialized:
            logger.debug("PyLet already initialized")
            return

        logger.info(f"Initializing PyLet client with head: {self.head_url}")
        pylet.init(self.head_url)
        self._initialized = True
        logger.info("PyLet client initialized successfully")

    @property
    def initialized(self) -> bool:
        """Check if PyLet client is initialized."""
        return self._initialized

    def deploy_model(
        self,
        model_id: str,
        backend: str = "vllm",
        gpu_count: int = 1,
        name: str | None = None,
        env: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
        target_worker: str | None = None,
    ) -> InstanceInfo:
        """Deploy a model instance via PyLet.

        Args:
            model_id: Model identifier (e.g., "Qwen/Qwen3-0.6B").
            backend: Model backend, one of "vllm" or "sglang".
            gpu_count: Number of GPUs to allocate.
            name: Optional instance name.
            env: Additional environment variables.
            labels: Additional instance labels.
            target_worker: Target specific worker for placement.

        Returns:
            InstanceInfo with deployment details.

        Raises:
            ValueError: If backend is not supported.
            PyletError: If deployment fails.
        """
        self._ensure_initialized()

        if backend not in MODEL_COMMANDS:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Supported: {list(MODEL_COMMANDS.keys())}"
            )

        # Build command
        command_template = MODEL_COMMANDS[backend]
        command = command_template.format(model_id=model_id)

        # Generate instance name if not provided
        if name is None:
            safe_model_id = model_id.replace("/", "-")
            name = f"{safe_model_id}-{backend}"

        # Build labels with SwarmPilot metadata
        instance_labels = {
            "model_id": model_id,
            "backend": backend,
            "managed_by": "swarmpilot",
        }
        if labels:
            instance_labels.update(labels)

        # Build environment
        instance_env = {}
        if env:
            instance_env.update(env)

        logger.info(
            f"Deploying model: {model_id} (backend={backend}, gpu={gpu_count})"
        )

        # Build submit kwargs
        submit_kwargs: dict = {
            "gpu": gpu_count,
            "name": name,
            "labels": instance_labels,
            "env": instance_env,
        }
        if target_worker is not None:
            submit_kwargs["target_worker"] = target_worker

        instance = pylet.submit(command, **submit_kwargs)
        logger.info(f"Instance submitted: pylet_id={instance.id}, name={name}")

        return InstanceInfo.from_pylet_instance(instance)

    def get_instance(self, pylet_id: str) -> InstanceInfo | None:
        """Get instance info by PyLet ID.

        Args:
            pylet_id: PyLet instance UUID.

        Returns:
            InstanceInfo if found, None otherwise.
        """
        self._ensure_initialized()

        try:
            instance = pylet.get(pylet_id)
            return InstanceInfo.from_pylet_instance(instance)
        except NotFoundError:
            logger.debug(f"Instance not found: {pylet_id}")
            return None

    def wait_instance_running(
        self,
        pylet_id: str,
        timeout: float = 300.0,
    ) -> InstanceInfo:
        """Wait for instance to reach RUNNING state.

        Args:
            pylet_id: PyLet instance UUID.
            timeout: Maximum wait time in seconds.

        Returns:
            Updated InstanceInfo with endpoint.

        Raises:
            NotFoundError: If instance not found.
            TimeoutError: If instance doesn't reach RUNNING within timeout.
            InstanceFailedError: If instance enters FAILED state.
        """
        self._ensure_initialized()

        instance = pylet.get(pylet_id)
        logger.info(f"Waiting for instance {pylet_id} to be running...")

        instance.wait_running(timeout=timeout)

        logger.info(
            f"Instance {pylet_id} running at endpoint: {instance.endpoint}"
        )
        return InstanceInfo.from_pylet_instance(instance)

    def cancel_instance(
        self,
        pylet_id: str,
        delete: bool = True,
    ) -> bool:
        """Cancel and optionally delete an instance.

        Args:
            pylet_id: PyLet instance UUID.
            delete: Whether to delete the instance after cancellation.

        Returns:
            True if cancelled successfully, False if not found.
        """
        self._ensure_initialized()

        try:
            instance = pylet.get(pylet_id)
            instance.cancel(delete=delete)
            logger.info(f"Instance {pylet_id} cancelled (delete={delete})")
            return True
        except NotFoundError:
            logger.warning(f"Instance not found for cancellation: {pylet_id}")
            return False
        except PyletError as e:
            logger.error(f"Failed to cancel instance {pylet_id}: {e}")
            return False

    def cancel_instances(
        self,
        pylet_ids: list[str],
        delete: bool = True,
    ) -> dict[str, bool]:
        """Cancel multiple instances.

        Args:
            pylet_ids: List of PyLet instance UUIDs.
            delete: Whether to delete instances after cancellation.

        Returns:
            Dict mapping pylet_id to success status.
        """
        results = {}
        for pylet_id in pylet_ids:
            results[pylet_id] = self.cancel_instance(pylet_id, delete=delete)
        return results

    def list_model_instances(
        self,
        model_id: str | None = None,
    ) -> list[InstanceInfo]:
        """List all SwarmPilot-managed model instances.

        Args:
            model_id: Optional filter by model ID.

        Returns:
            List of InstanceInfo for matching instances.
        """
        self._ensure_initialized()

        all_instances = pylet.instances()
        result = []

        for instance in all_instances:
            labels = instance.labels
            # Filter to SwarmPilot-managed instances
            if labels.get("managed_by") != "swarmpilot":
                continue

            # Filter by model_id if specified
            if model_id is not None and labels.get("model_id") != model_id:
                continue

            result.append(InstanceInfo.from_pylet_instance(instance))

        return result

    def list_running_instances(
        self,
        model_id: str | None = None,
    ) -> list[InstanceInfo]:
        """List all running SwarmPilot-managed instances.

        Args:
            model_id: Optional filter by model ID.

        Returns:
            List of running InstanceInfo.
        """
        all_instances = self.list_model_instances(model_id)
        return [i for i in all_instances if i.status == "RUNNING"]

    def _ensure_initialized(self) -> None:
        """Ensure PyLet client is initialized.

        Raises:
            RuntimeError: If client not initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "PyLet client not initialized. Call init() first."
            )


# Global client singleton
_pylet_client: PyLetClient | None = None


def create_pylet_client(head_url: str) -> PyLetClient:
    """Create and initialize the global PyLet client.

    Args:
        head_url: URL of the PyLet head node.

    Returns:
        Initialized PyLetClient instance.
    """
    global _pylet_client

    _pylet_client = PyLetClient(head_url)
    _pylet_client.init()
    return _pylet_client


def get_pylet_client() -> PyLetClient:
    """Get the global PyLet client.

    Returns:
        The global PyLetClient instance.

    Raises:
        RuntimeError: If client not created.
    """
    if _pylet_client is None:
        raise RuntimeError(
            "PyLet client not created. Call create_pylet_client() first."
        )
    return _pylet_client
