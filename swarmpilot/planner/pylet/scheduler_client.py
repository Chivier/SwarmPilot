"""Scheduler client for planner-managed registration (PYLET-012).

This module provides an HTTP client for the scheduler's instance management API.
It allows the planner to register and deregister instances on their behalf,
eliminating the need for instances to self-register.

Example:
    from swarmpilot.planner.pylet.scheduler_client import SchedulerClient

    client = SchedulerClient("http://localhost:8000")
    result = client.register_instance(
        instance_id="my-instance",
        model_id="Qwen/Qwen3-0.6B",
        endpoint="192.168.1.100:8001",
        backend="vllm",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger


@dataclass
class RegistrationInfo:
    """Information about an instance registration.

    Attributes:
        instance_id: Instance identifier.
        model_id: Model identifier.
        endpoint: HTTP endpoint in "host:port" format.
        success: Whether registration succeeded.
        message: Status message from scheduler.
    """

    instance_id: str
    model_id: str
    endpoint: str
    success: bool
    message: str


# Default platform info templates for backends
PLATFORM_INFO_TEMPLATES: dict[str, dict[str, str]] = {
    "vllm": {
        "software_name": "vllm",
        "software_version": "0.6.0",
        "hardware_name": "gpu",
    },
    "sglang": {
        "software_name": "sglang",
        "software_version": "0.4.0",
        "hardware_name": "gpu",
    },
}


class SchedulerClient:
    """HTTP client for scheduler instance management API.

    This client wraps the scheduler's instance registration and removal
    endpoints, providing a clean interface for the planner to manage
    instance lifecycle.

    Attributes:
        scheduler_url: Base URL of the scheduler.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        scheduler_url: str,
        timeout: float = 10.0,
    ):
        """Initialize scheduler client.

        Args:
            scheduler_url: Base URL of the scheduler (e.g., "http://localhost:8000").
            timeout: Request timeout in seconds.
        """
        self.scheduler_url = scheduler_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> SchedulerClient:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def register_instance(
        self,
        instance_id: str,
        model_id: str,
        endpoint: str,
        backend: str = "vllm",
        platform_info: dict[str, str] | None = None,
    ) -> RegistrationInfo:
        """Register an instance with the scheduler.

        Args:
            instance_id: Unique instance identifier.
            model_id: Model identifier (e.g., "Qwen/Qwen3-0.6B").
            endpoint: HTTP endpoint in "host:port" format.
            backend: Model backend ("vllm" or "sglang").
            platform_info: Custom platform info (overrides defaults).

        Returns:
            RegistrationInfo with registration result.
        """
        # Build platform info
        if platform_info is None:
            platform_info = PLATFORM_INFO_TEMPLATES.get(
                backend,
                PLATFORM_INFO_TEMPLATES["vllm"],
            ).copy()

        # Ensure endpoint has http:// prefix for scheduler
        if not endpoint.startswith("http://") and not endpoint.startswith(
            "https://"
        ):
            full_endpoint = f"http://{endpoint}"
        else:
            full_endpoint = endpoint

        payload = {
            "instance_id": instance_id,
            "model_id": model_id,
            "endpoint": full_endpoint,
            "platform_info": platform_info,
        }

        logger.info(
            f"Registering instance with scheduler: {instance_id} "
            f"(model={model_id}, endpoint={endpoint})"
        )

        try:
            response = self._client.post(
                f"{self.scheduler_url}/v1/instance/register",
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(
                    f"Instance registered: {instance_id} - {data.get('message', 'OK')}"
                )
                return RegistrationInfo(
                    instance_id=instance_id,
                    model_id=model_id,
                    endpoint=endpoint,
                    success=True,
                    message=data.get("message", "Registered successfully"),
                )
            else:
                error_detail = response.json().get("detail", {})
                error_msg = error_detail.get("error", response.text)
                logger.error(
                    f"Registration failed for {instance_id}: "
                    f"{response.status_code} - {error_msg}"
                )
                return RegistrationInfo(
                    instance_id=instance_id,
                    model_id=model_id,
                    endpoint=endpoint,
                    success=False,
                    message=f"Registration failed: {error_msg}",
                )

        except httpx.RequestError as e:
            logger.error(f"Registration request failed for {instance_id}: {e}")
            return RegistrationInfo(
                instance_id=instance_id,
                model_id=model_id,
                endpoint=endpoint,
                success=False,
                message=f"Request error: {e}",
            )

    def drain_instance(self, instance_id: str) -> bool:
        """Start draining an instance (stop new task assignments).

        Args:
            instance_id: Instance identifier.

        Returns:
            True if drain started successfully, False otherwise.
        """
        logger.info(f"Draining instance: {instance_id}")

        try:
            response = self._client.post(
                f"{self.scheduler_url}/v1/instance/drain",
                json={"instance_id": instance_id},
            )

            if response.status_code == 200:
                logger.debug(f"Instance {instance_id} draining started")
                return True
            elif response.status_code == 404:
                logger.warning(f"Instance {instance_id} not found for draining")
                return True  # Already gone is OK
            else:
                logger.error(
                    f"Drain failed for {instance_id}: {response.status_code}"
                )
                return False

        except httpx.RequestError as e:
            logger.error(f"Drain request failed for {instance_id}: {e}")
            return False

    def check_drain_status(
        self, instance_id: str
    ) -> tuple[bool, int]:
        """Check if instance can be safely removed.

        Args:
            instance_id: Instance identifier.

        Returns:
            Tuple of (can_remove, pending_tasks_count).
        """
        try:
            response = self._client.get(
                f"{self.scheduler_url}/v1/instance/drain/status",
                params={"instance_id": instance_id},
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("can_remove", False), data.get("pending_tasks", 0)
            else:
                return False, -1

        except httpx.RequestError as e:
            logger.error(f"Drain status check failed for {instance_id}: {e}")
            return False, -1

    def remove_instance(self, instance_id: str) -> bool:
        """Remove an instance from the scheduler.

        Note: Safe removal requires draining first. Use deregister_instance()
        for the full drain-then-remove flow.

        Args:
            instance_id: Instance identifier.

        Returns:
            True if removal succeeded, False otherwise.
        """
        logger.info(f"Removing instance: {instance_id}")

        try:
            response = self._client.post(
                f"{self.scheduler_url}/v1/instance/remove",
                json={"instance_id": instance_id},
            )

            if response.status_code == 200:
                logger.info(f"Instance {instance_id} removed")
                return True
            elif response.status_code == 404:
                logger.warning(
                    f"Instance {instance_id} not found (already removed?)"
                )
                return True  # Already gone is OK
            else:
                error_detail = response.json().get("detail", {})
                logger.error(
                    f"Remove failed for {instance_id}: {error_detail}"
                )
                return False

        except httpx.RequestError as e:
            logger.error(f"Remove request failed for {instance_id}: {e}")
            return False

    def deregister_instance(
        self,
        instance_id: str,
        wait_for_drain: bool = True,
        drain_timeout: float = 30.0,
    ) -> bool:
        """Safely deregister an instance (drain then remove).

        This performs the safe removal flow:
        1. Start draining (stop new task assignments)
        2. Optionally wait for pending tasks to complete
        3. Remove the instance

        Args:
            instance_id: Instance identifier.
            wait_for_drain: Whether to wait for drain to complete.
            drain_timeout: Maximum time to wait for drain in seconds.

        Returns:
            True if deregistration succeeded, False otherwise.
        """
        # Step 1: Drain
        if not self.drain_instance(instance_id):
            # If drain fails but instance is not found, that's OK
            return self.remove_instance(instance_id)

        # Step 2: Wait for drain (optional)
        if wait_for_drain:
            import time

            start_time = time.time()
            while time.time() - start_time < drain_timeout:
                can_remove, pending = self.check_drain_status(instance_id)
                if can_remove or pending == 0:
                    break
                logger.debug(
                    f"Waiting for {instance_id} to drain "
                    f"({pending} tasks pending)..."
                )
                time.sleep(1.0)

        # Step 3: Remove
        return self.remove_instance(instance_id)

    def list_instances(
        self, model_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List registered instances.

        Args:
            model_id: Optional filter by model ID.

        Returns:
            List of instance dictionaries.
        """
        try:
            params = {}
            if model_id:
                params["model_id"] = model_id

            response = self._client.get(
                f"{self.scheduler_url}/v1/instance/list",
                params=params,
            )

            if response.status_code == 200:
                return response.json().get("instances", [])
            else:
                logger.error(f"List instances failed: {response.status_code}")
                return []

        except httpx.RequestError as e:
            logger.error(f"List instances request failed: {e}")
            return []

    def get_instance_info(self, instance_id: str) -> dict[str, Any] | None:
        """Get detailed instance information.

        Args:
            instance_id: Instance identifier.

        Returns:
            Instance info dictionary, or None if not found.
        """
        try:
            response = self._client.get(
                f"{self.scheduler_url}/v1/instance/info",
                params={"instance_id": instance_id},
            )

            if response.status_code == 200:
                return response.json()
            else:
                return None

        except httpx.RequestError as e:
            logger.error(f"Get instance info failed for {instance_id}: {e}")
            return None

    def health_check(self) -> bool:
        """Check scheduler health.

        Returns:
            True if scheduler is healthy, False otherwise.
        """
        try:
            response = self._client.get(f"{self.scheduler_url}/v1/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def reassign_model(self, new_model_id: str) -> bool:
        """Reassign the scheduler to serve a different model.

        Only succeeds if the scheduler has zero registered instances.

        Args:
            new_model_id: New model identifier.

        Returns:
            True if reassignment succeeded.
        """
        try:
            response = self._client.post(
                f"{self.scheduler_url}/v1/model/reassign",
                json={"model_id": new_model_id},
            )
            if response.status_code == 200:
                logger.info(
                    f"Scheduler {self.scheduler_url} reassigned "
                    f"to {new_model_id}"
                )
                return True
            logger.warning(
                f"Scheduler reassign failed "
                f"({response.status_code}): {response.text}"
            )
            return False
        except httpx.RequestError as e:
            logger.error(f"Scheduler reassign request error: {e}")
            return False


# Global client singleton
_scheduler_client: SchedulerClient | None = None


def create_scheduler_client(scheduler_url: str) -> SchedulerClient:
    """Create the global scheduler client.

    Args:
        scheduler_url: Base URL of the scheduler.

    Returns:
        SchedulerClient instance.
    """
    global _scheduler_client
    _scheduler_client = SchedulerClient(scheduler_url)
    return _scheduler_client


def get_scheduler_client() -> SchedulerClient:
    """Get the global scheduler client.

    Returns:
        The global SchedulerClient instance.

    Raises:
        RuntimeError: If client not created.
    """
    if _scheduler_client is None:
        raise RuntimeError(
            "Scheduler client not created. "
            "Call create_scheduler_client() first."
        )
    return _scheduler_client
