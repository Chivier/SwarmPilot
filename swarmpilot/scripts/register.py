"""Model registration with scheduler (PYLET-002).

This module provides functions for registering model instances with the
SwarmPilot scheduler. When a model is deployed via PyLet, this module
handles the registration flow to make the model available for routing.

Registration Flow:
1. Model is deployed via PyLet (PYLET-001)
2. Wait for model health check to pass
3. Register instance with scheduler via /instance/register
4. On shutdown, deregister via /instance/drain + /instance/remove

Example:
    from swarmpilot.scripts.register import register_with_scheduler, deregister_from_scheduler

    # After model is ready
    success = register_with_scheduler(
        scheduler_url="http://localhost:8000",
        instance_id="my-instance",
        model_id="Qwen/Qwen3-0.6B",
        endpoint="192.168.1.100:8001",
    )

    # On shutdown
    deregister_from_scheduler(
        scheduler_url="http://localhost:8000",
        instance_id="my-instance",
    )
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import httpx
from loguru import logger


def get_instance_info(
    instance_id: Optional[str] = None,
    model_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    backend: Optional[str] = None,
    gpu_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Get instance information from arguments or environment.

    This function collects instance metadata for registration. Values can be
    provided directly or read from environment variables (for use in PyLet
    startup scripts).

    Args:
        instance_id: Instance ID. Env: PYLET_INSTANCE_ID, PYLET_INSTANCE_NAME.
        model_id: Model identifier. Env: MODEL_ID.
        endpoint: Instance endpoint. Env: constructed from HOSTNAME:PORT.
        backend: Model backend. Env: MODEL_BACKEND.
        gpu_count: GPU count. Env: GPU_COUNT.

    Returns:
        Dictionary with instance registration info.

    Example:
        # From arguments
        info = get_instance_info(
            instance_id="my-instance",
            model_id="Qwen/Qwen3-0.6B",
            endpoint="192.168.1.100:8001",
        )

        # From environment (in PyLet startup script)
        info = get_instance_info()
    """
    # Get instance ID from env if not provided
    if instance_id is None:
        instance_id = os.getenv(
            "PYLET_INSTANCE_ID",
            os.getenv("PYLET_INSTANCE_NAME", f"inst-{os.getpid()}"),
        )

    # Get model ID from env if not provided
    if model_id is None:
        model_id = os.getenv("MODEL_ID", "unknown")

    # Get endpoint from env if not provided
    if endpoint is None:
        hostname = os.getenv("HOSTNAME", "localhost")
        port = os.getenv("PORT")
        if port is None:
            raise ValueError("PORT environment variable must be set")
        endpoint = f"{hostname}:{port}"

    # Get backend from env if not provided
    if backend is None:
        backend = os.getenv("MODEL_BACKEND", "vllm")

    # Get GPU count from env if not provided
    if gpu_count is None:
        gpu_count = int(os.getenv("GPU_COUNT", "1"))

    return {
        "instance_id": instance_id,
        "model_id": model_id,
        "endpoint": endpoint,
        "backend": backend,
        "gpu_count": gpu_count,
    }


def register_with_scheduler(
    scheduler_url: str,
    instance_id: str,
    model_id: str,
    endpoint: str,
    platform_info: Optional[Dict[str, str]] = None,
    retries: int = 5,
    initial_backoff: float = 1.0,
) -> bool:
    """Register instance with the SwarmPilot scheduler.

    Sends a registration request to the scheduler's /instance/register endpoint.
    Uses exponential backoff on failure.

    Args:
        scheduler_url: Scheduler base URL (e.g., "http://localhost:8000").
        instance_id: Unique instance identifier.
        model_id: Model identifier for routing.
        endpoint: Instance endpoint in "host:port" or "http://host:port" format.
        platform_info: Platform information for runtime prediction. If not
            provided, uses defaults for docker/unknown.
        retries: Number of retry attempts on failure.
        initial_backoff: Initial backoff time in seconds (doubles each retry).

    Returns:
        True if registration succeeded, False otherwise.

    Example:
        success = register_with_scheduler(
            scheduler_url="http://localhost:8000",
            instance_id="my-instance",
            model_id="Qwen/Qwen3-0.6B",
            endpoint="192.168.1.100:8001",
        )
    """
    register_url = f"{scheduler_url.rstrip('/')}/instance/register"

    # Normalize endpoint to include http:// if not present
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = f"http://{endpoint}"

    # Default platform info if not provided
    if platform_info is None:
        platform_info = {
            "software_name": "pylet",
            "software_version": "1.0",
            "hardware_name": "unknown",
        }

    payload = {
        "instance_id": instance_id,
        "model_id": model_id,
        "endpoint": endpoint,
        "platform_info": platform_info,
    }

    logger.info(f"Registering instance {instance_id} with scheduler at {scheduler_url}")

    backoff = initial_backoff
    for attempt in range(retries):
        try:
            response = httpx.post(register_url, json=payload, timeout=10.0)

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(
                        f"Successfully registered instance {instance_id} "
                        f"with scheduler"
                    )
                    return True
                else:
                    logger.warning(f"Registration failed: {data.get('error')}")
            else:
                logger.warning(
                    f"Registration failed with status {response.status_code}: "
                    f"{response.text}"
                )

        except httpx.RequestError as e:
            logger.warning(f"Registration request failed: {e}")

        if attempt < retries - 1:
            logger.info(f"Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff

    logger.error(f"Failed to register instance {instance_id} after {retries} attempts")
    return False


def deregister_from_scheduler(
    scheduler_url: str,
    instance_id: str,
    wait_for_drain: bool = True,
    drain_timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> bool:
    """Deregister instance from the SwarmPilot scheduler.

    Performs a graceful shutdown by:
    1. Draining the instance (stops new task assignments)
    2. Waiting for pending tasks to complete
    3. Removing the instance from the registry

    Args:
        scheduler_url: Scheduler base URL.
        instance_id: Instance ID to deregister.
        wait_for_drain: If True, wait for drain to complete before removal.
        drain_timeout: Maximum time to wait for drain completion.
        poll_interval: Time between drain status checks.

    Returns:
        True if deregistration succeeded, False otherwise.

    Example:
        deregister_from_scheduler(
            scheduler_url="http://localhost:8000",
            instance_id="my-instance",
        )
    """
    base_url = scheduler_url.rstrip("/")

    # Step 1: Start draining
    logger.info(f"Draining instance {instance_id}...")
    try:
        response = httpx.post(
            f"{base_url}/instance/drain",
            json={"instance_id": instance_id},
            timeout=10.0,
        )

        if response.status_code == 404:
            logger.warning(f"Instance {instance_id} not found in scheduler")
            return True  # Already gone

        if response.status_code != 200:
            data = response.json()
            # If already draining or not active, proceed anyway
            if "must be in ACTIVE state" in data.get("error", ""):
                logger.info("Instance already draining or not active")
            else:
                logger.warning(f"Drain failed: {data.get('error')}")
                return False

    except httpx.RequestError as e:
        logger.warning(f"Drain request failed: {e}")
        return False

    # Step 2: Wait for drain to complete
    if wait_for_drain:
        logger.info("Waiting for drain to complete...")
        start_time = time.time()

        while time.time() - start_time < drain_timeout:
            try:
                response = httpx.get(
                    f"{base_url}/instance/drain/status",
                    params={"instance_id": instance_id},
                    timeout=10.0,
                )

                if response.status_code == 404:
                    logger.info("Instance already removed")
                    return True

                if response.status_code == 200:
                    data = response.json()
                    if data.get("can_remove"):
                        logger.info("Drain complete, ready for removal")
                        break
                    pending = data.get("pending_tasks", 0)
                    logger.debug(f"Drain in progress, {pending} tasks remaining")

            except httpx.RequestError as e:
                logger.warning(f"Drain status check failed: {e}")

            time.sleep(poll_interval)
        else:
            logger.warning(f"Drain timed out after {drain_timeout}s")
            # Proceed with removal anyway

    # Step 3: Remove the instance
    logger.info(f"Removing instance {instance_id}...")
    try:
        response = httpx.post(
            f"{base_url}/instance/remove",
            json={"instance_id": instance_id},
            timeout=10.0,
        )

        if response.status_code == 200:
            logger.info(f"Instance {instance_id} removed successfully")
            return True
        elif response.status_code == 404:
            logger.info(f"Instance {instance_id} already removed")
            return True
        else:
            data = response.json()
            logger.warning(f"Remove failed: {data.get('error')}")
            return False

    except httpx.RequestError as e:
        logger.warning(f"Remove request failed: {e}")
        return False


def force_remove_from_scheduler(
    scheduler_url: str,
    instance_id: str,
) -> bool:
    """Force remove an instance without draining.

    This is a last-resort option when graceful shutdown is not possible.
    Pending tasks on this instance may be lost.

    Args:
        scheduler_url: Scheduler base URL.
        instance_id: Instance ID to remove.

    Returns:
        True if removal succeeded, False otherwise.
    """
    base_url = scheduler_url.rstrip("/")

    logger.warning(f"Force removing instance {instance_id} (skipping drain)")

    try:
        # First try to drain (to change status)
        httpx.post(
            f"{base_url}/instance/drain",
            json={"instance_id": instance_id},
            timeout=5.0,
        )
    except httpx.RequestError:
        pass  # Ignore errors

    # Then remove immediately
    try:
        response = httpx.post(
            f"{base_url}/instance/remove",
            json={"instance_id": instance_id},
            timeout=10.0,
        )

        if response.status_code in (200, 404):
            logger.info(f"Instance {instance_id} force removed")
            return True

        data = response.json()
        logger.warning(f"Force remove failed: {data.get('error')}")
        return False

    except httpx.RequestError as e:
        logger.warning(f"Force remove request failed: {e}")
        return False
