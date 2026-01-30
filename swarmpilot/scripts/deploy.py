"""Model deployment via PyLet (PYLET-001).

This module provides functions to deploy model services (vLLM, sglang)
directly via PyLet. Models expose their native HTTP API on PyLet-assigned
ports without a wrapper layer.

Example:
    import pylet
    from swarmpilot.scripts.deploy import deploy_model, wait_model_ready

    pylet.init("http://localhost:8000")

    instance = deploy_model("Qwen/Qwen3-0.6B", backend="vllm", gpu_count=1)
    endpoint = await wait_model_ready(instance)
    print(f"Model ready at: {endpoint}")
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx
from loguru import logger

if TYPE_CHECKING:
    # Avoid import error when pylet is not installed
    pass


# Model launch command templates
# $PORT is replaced by PyLet with the auto-allocated port
MODEL_COMMANDS: Dict[str, str] = {
    "vllm": "vllm serve {model_id} --port $PORT --host 0.0.0.0",
    "sglang": (
        "python -m sglang.launch_server "
        "--model {model_id} --port $PORT --host 0.0.0.0"
    ),
}


def deploy_model(
    model_id: str,
    backend: str = "vllm",
    gpu_count: int = 1,
    env: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
    target_worker: Optional[str] = None,
) -> Any:
    """Deploy a model directly via PyLet.

    This function submits a model service to PyLet for execution. The model
    exposes its native HTTP API on the PyLet-assigned port ($PORT).

    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen3-0.6B").
        backend: Model backend, one of "vllm" or "sglang".
        gpu_count: Number of GPUs to allocate for the model.
        env: Additional environment variables to set.
        labels: Custom labels for the instance.
        name: Optional name for the instance. If not provided, a name is
            generated from model_id and backend.
        target_worker: Optional worker ID to target for placement.

    Returns:
        PyLet Instance handle. Use wait_model_ready() to wait for the model
        to be ready.

    Raises:
        ValueError: If backend is not supported.
        pylet.NotInitializedError: If pylet.init() was not called.

    Example:
        instance = deploy_model("Qwen/Qwen3-0.6B", backend="vllm", gpu_count=1)
    """
    import pylet

    if backend not in MODEL_COMMANDS:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: {list(MODEL_COMMANDS.keys())}"
        )

    command_template = MODEL_COMMANDS[backend]
    command = command_template.format(model_id=model_id)

    # Generate instance name if not provided
    if name is None:
        # Replace / with - for valid instance names
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

    # Build environment variables
    instance_env = {}
    if env:
        instance_env.update(env)

    logger.info(f"Deploying model: {model_id} (backend={backend}, gpu={gpu_count})")

    # Build submit kwargs
    submit_kwargs: Dict[str, Any] = {
        "gpu": gpu_count,
        "name": name,
        "labels": instance_labels,
        "env": instance_env,
    }
    if target_worker is not None:
        submit_kwargs["target_worker"] = target_worker

    instance = pylet.submit(command, **submit_kwargs)

    logger.info(f"Instance submitted: id={instance.id}, name={instance.name}")
    return instance


async def wait_model_ready(
    instance: Any,
    timeout: float = 300.0,
    health_path: str = "/health",
    poll_interval: float = 2.0,
) -> str:
    """Wait for model to be ready and return endpoint.

    This function waits for the PyLet instance to reach RUNNING state,
    then polls the model's health endpoint until it responds successfully.

    Args:
        instance: PyLet Instance handle from deploy_model().
        timeout: Maximum wait time in seconds.
        health_path: Health check endpoint path.
        poll_interval: Time between health check polls.

    Returns:
        Model endpoint in "host:port" format.

    Raises:
        TimeoutError: If model does not become healthy within timeout.
        pylet.InstanceFailedError: If instance enters FAILED or CANCELLED state.

    Example:
        instance = deploy_model("Qwen/Qwen3-0.6B")
        endpoint = await wait_model_ready(instance)
        # endpoint = "192.168.1.100:8001"
    """
    # Wait for instance to be running (blocking call)
    logger.info(f"Waiting for instance {instance.id} to reach RUNNING state...")
    instance.wait_running(timeout=timeout)

    endpoint = instance.endpoint
    if endpoint is None:
        raise RuntimeError(f"Instance {instance.id} is RUNNING but has no endpoint")

    logger.info(f"Instance running at endpoint: {endpoint}")

    # Wait for model health check
    health_url = f"http://{endpoint}{health_path}"
    logger.info(f"Waiting for health check at {health_url}...")

    start_time = asyncio.get_event_loop().time()
    remaining_timeout = timeout - (start_time - asyncio.get_event_loop().time())

    async with httpx.AsyncClient(timeout=10.0) as client:
        while asyncio.get_event_loop().time() - start_time < remaining_timeout:
            try:
                response = await client.get(health_url)
                if response.status_code == 200:
                    logger.info(f"Model healthy at {endpoint}")
                    return endpoint
                logger.debug(
                    f"Health check returned {response.status_code}, retrying..."
                )
            except httpx.RequestError as e:
                logger.debug(f"Health check failed: {e}, retrying...")

            await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Model at {endpoint} did not become healthy within {timeout}s")


def wait_model_ready_sync(
    instance: Any,
    timeout: float = 300.0,
    health_path: str = "/health",
    poll_interval: float = 2.0,
) -> str:
    """Synchronous version of wait_model_ready().

    For use in non-async contexts.

    Args:
        instance: PyLet Instance handle from deploy_model().
        timeout: Maximum wait time in seconds.
        health_path: Health check endpoint path.
        poll_interval: Time between health check polls.

    Returns:
        Model endpoint in "host:port" format.

    Raises:
        TimeoutError: If model does not become healthy within timeout.
        pylet.InstanceFailedError: If instance enters FAILED or CANCELLED state.
    """
    import time

    # Wait for instance to be running
    logger.info(f"Waiting for instance {instance.id} to reach RUNNING state...")
    instance.wait_running(timeout=timeout)

    endpoint = instance.endpoint
    if endpoint is None:
        raise RuntimeError(f"Instance {instance.id} is RUNNING but has no endpoint")

    logger.info(f"Instance running at endpoint: {endpoint}")

    # Wait for model health check
    health_url = f"http://{endpoint}{health_path}"
    logger.info(f"Waiting for health check at {health_url}...")

    start_time = time.time()
    with httpx.Client(timeout=10.0) as client:
        while time.time() - start_time < timeout:
            try:
                response = client.get(health_url)
                if response.status_code == 200:
                    logger.info(f"Model healthy at {endpoint}")
                    return endpoint
                logger.debug(
                    f"Health check returned {response.status_code}, retrying..."
                )
            except httpx.RequestError as e:
                logger.debug(f"Health check failed: {e}, retrying...")

            time.sleep(poll_interval)

    raise TimeoutError(f"Model at {endpoint} did not become healthy within {timeout}s")
