# PYLET-002: Model Registration

## Objective

Implement model self-registration with the scheduler when deployed via PyLet. The model startup script registers the instance with the scheduler after the model service is ready.

## Prerequisites

- PYLET-001 completed (Direct model deployment)
- Scheduler registration API available

## Background

When a model is deployed via PyLet:
1. PyLet starts the model process with `$PORT` assigned
2. Model service starts and becomes healthy
3. A registration script notifies the scheduler of the new instance
4. Scheduler adds the instance to its routing table

## Implementation Steps

### Step 1: Create Registration Script

Create `scripts/register_with_scheduler.py`:

```python
#!/usr/bin/env python3
"""Register model instance with scheduler.

This script is called after model startup to register
the instance with the SwarmPilot scheduler.
"""

import os
import sys
import time
import httpx
from loguru import logger


def get_instance_info() -> dict:
    """Get instance information from environment."""
    return {
        "instance_id": os.getenv("PYLET_INSTANCE_ID", f"inst-{os.getpid()}"),
        "model_id": os.getenv("MODEL_ID", "unknown"),
        "endpoint": f"{os.getenv('HOSTNAME', 'localhost')}:{os.getenv('PORT')}",
        "gpu_count": int(os.getenv("GPU_COUNT", "1")),
        "backend": os.getenv("MODEL_BACKEND", "vllm"),
    }


def wait_for_health(endpoint: str, timeout: float = 120.0) -> bool:
    """Wait for model health check to pass."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"http://{endpoint}/health", timeout=5.0)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(2.0)
    return False


def register_with_scheduler(
    scheduler_url: str,
    instance_info: dict,
    retries: int = 5,
) -> bool:
    """Register instance with scheduler."""
    for attempt in range(retries):
        try:
            response = httpx.post(
                f"{scheduler_url}/model/register",
                json=instance_info,
                timeout=10.0,
            )
            if response.status_code == 200:
                logger.info(f"Registered with scheduler: {instance_info['instance_id']}")
                return True
            logger.warning(f"Registration failed: {response.status_code}")
        except httpx.RequestError as e:
            logger.warning(f"Registration error: {e}")

        time.sleep(2 ** attempt)  # Exponential backoff

    return False


def main():
    scheduler_url = os.getenv("SCHEDULER_URL", "http://localhost:8000")
    instance_info = get_instance_info()

    logger.info(f"Instance info: {instance_info}")

    # Wait for model to be healthy
    if not wait_for_health(instance_info["endpoint"]):
        logger.error("Model did not become healthy")
        sys.exit(1)

    # Register with scheduler
    if not register_with_scheduler(scheduler_url, instance_info):
        logger.error("Failed to register with scheduler")
        sys.exit(1)

    logger.info("Registration complete")


if __name__ == "__main__":
    main()
```

### Step 2: Create Startup Wrapper Script

Create `scripts/start_model.sh`:

```bash
#!/bin/bash
# Start model and register with scheduler

set -e

MODEL_ID="${MODEL_ID:?MODEL_ID required}"
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
SCHEDULER_URL="${SCHEDULER_URL:-http://localhost:8000}"

# Start model in background
if [ "$MODEL_BACKEND" = "vllm" ]; then
    vllm serve "$MODEL_ID" --port "$PORT" --host 0.0.0.0 &
elif [ "$MODEL_BACKEND" = "sglang" ]; then
    python -m sglang.launch_server --model "$MODEL_ID" --port "$PORT" --host 0.0.0.0 &
else
    echo "Unknown backend: $MODEL_BACKEND"
    exit 1
fi

MODEL_PID=$!

# Register with scheduler
python scripts/register_with_scheduler.py

# Wait for model process
wait $MODEL_PID
```

### Step 3: Update PyLet Deploy Command

```python
def deploy_model_with_registration(
    model_id: str,
    scheduler_url: str,
    backend: str = "vllm",
    gpu_count: int = 1,
) -> pylet.Instance:
    """Deploy model with automatic scheduler registration.

    Args:
        model_id: Model identifier
        scheduler_url: Scheduler URL for registration
        backend: Model backend
        gpu_count: GPU allocation

    Returns:
        PyLet instance handle
    """
    command = f"bash scripts/start_model.sh"

    instance = pylet.submit(
        command,
        gpu=gpu_count,
        name=f"{model_id.replace('/', '-')}-{backend}",
        env={
            "MODEL_ID": model_id,
            "MODEL_BACKEND": backend,
            "SCHEDULER_URL": scheduler_url,
        },
        labels={
            "model_id": model_id,
            "backend": backend,
            "managed_by": "swarmpilot",
        },
    )

    return instance
```

### Step 4: Implement Deregistration on Shutdown

Add signal handling to deregister on shutdown:

```python
# In register_with_scheduler.py
import signal


def deregister_from_scheduler(scheduler_url: str, instance_id: str) -> None:
    """Deregister instance from scheduler."""
    try:
        response = httpx.post(
            f"{scheduler_url}/model/deregister",
            json={"instance_id": instance_id},
            timeout=5.0,
        )
        logger.info(f"Deregistered: {response.status_code}")
    except httpx.RequestError as e:
        logger.warning(f"Deregistration error: {e}")


def setup_signal_handlers(scheduler_url: str, instance_id: str) -> None:
    """Setup signal handlers for graceful shutdown."""
    def handler(signum, frame):
        logger.info(f"Received signal {signum}, deregistering...")
        deregister_from_scheduler(scheduler_url, instance_id)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
```

## Test Strategy

### Unit Tests

```python
def test_get_instance_info():
    """Test instance info extraction."""
    os.environ["PORT"] = "8000"
    os.environ["MODEL_ID"] = "test-model"

    info = get_instance_info()
    assert info["model_id"] == "test-model"
    assert "8000" in info["endpoint"]


def test_wait_for_health_timeout():
    """Test health check timeout."""
    result = wait_for_health("localhost:99999", timeout=1.0)
    assert result is False
```

### Integration Tests

```bash
# Start mock scheduler
python -m http.server 8000 &

# Deploy and verify registration
python -c "
import pylet
pylet.init('http://localhost:8000')
instance = deploy_model_with_registration('test-model', 'http://localhost:8000')
instance.wait_running()
# Verify registration happened
"
```

## Acceptance Criteria

- [ ] Registration script created
- [ ] Model registers after health check passes
- [ ] Exponential backoff on registration failure
- [ ] Deregistration on shutdown signal
- [ ] Environment variables passed correctly

## Next Steps

Proceed to [PYLET-003](PYLET-003-signal-handling.md) for signal handling.

## Code References

- Scheduler registration API: [scheduler/src/api.py](../../scheduler/src/api.py)
