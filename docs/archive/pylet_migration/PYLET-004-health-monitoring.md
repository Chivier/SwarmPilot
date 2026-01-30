# PYLET-004: Health Monitoring

## Objective

Implement health monitoring for PyLet-managed model instances. The model's native health endpoint is used for readiness checks, and periodic heartbeats are sent to the scheduler.

## Prerequisites

- PYLET-001 completed (Direct model deployment)
- PYLET-002 completed (Model registration)
- PYLET-003 completed (Signal handling)

## Background

Health monitoring for direct model deployment:
1. **Model Health**: Use model's native `/health` endpoint
2. **Scheduler Heartbeat**: Periodic keepalive to scheduler
3. **PyLet Monitoring**: PyLet monitors process status

Since models expose their native API directly, we use their built-in health endpoints.

## Implementation Steps

### Step 1: Health Check in Startup Script

Update `scripts/wait_for_health.py`:

```python
#!/usr/bin/env python3
"""Wait for model health check to pass."""

import os
import sys
import time
import httpx
from loguru import logger


def wait_for_health(endpoint: str, timeout: float = 120.0) -> bool:
    """Wait for model health check.

    Args:
        endpoint: Model endpoint (host:port)
        timeout: Maximum wait time

    Returns:
        True if health check passes
    """
    start_time = time.time()
    check_count = 0

    while time.time() - start_time < timeout:
        check_count += 1
        try:
            # Try vLLM/OpenAI style health endpoint
            response = httpx.get(f"http://{endpoint}/health", timeout=5.0)
            if response.status_code == 200:
                logger.info(f"Model healthy at {endpoint} (check #{check_count})")
                return True
        except httpx.RequestError:
            pass

        try:
            # Try sglang style endpoint
            response = httpx.get(f"http://{endpoint}/v1/models", timeout=5.0)
            if response.status_code == 200:
                logger.info(f"Model ready at {endpoint} (check #{check_count})")
                return True
        except httpx.RequestError:
            pass

        if check_count % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Waiting for model... {elapsed:.0f}s elapsed")

        time.sleep(2.0)

    return False


def main():
    port = os.getenv("PORT", "8000")
    endpoint = f"localhost:{port}"
    timeout = float(os.getenv("HEALTH_TIMEOUT", "300"))

    logger.info(f"Waiting for model at {endpoint} (timeout: {timeout}s)")

    if not wait_for_health(endpoint, timeout):
        logger.error("Model failed to become healthy")
        sys.exit(1)

    logger.info("Model is healthy and ready")


if __name__ == "__main__":
    main()
```

### Step 2: Heartbeat to Scheduler

Create `scripts/heartbeat.py`:

```python
#!/usr/bin/env python3
"""Send periodic heartbeats to scheduler."""

import os
import sys
import time
import signal
import httpx
from loguru import logger


class HeartbeatSender:
    """Sends periodic heartbeats to scheduler."""

    def __init__(
        self,
        scheduler_url: str,
        instance_id: str,
        interval: float = 30.0,
    ):
        self.scheduler_url = scheduler_url
        self.instance_id = instance_id
        self.interval = interval
        self._running = True

    def stop(self):
        """Stop heartbeat loop."""
        self._running = False

    def run(self):
        """Run heartbeat loop."""
        logger.info(f"Starting heartbeat to {self.scheduler_url}")

        while self._running:
            try:
                response = httpx.post(
                    f"{self.scheduler_url}/instance/heartbeat",
                    json={"instance_id": self.instance_id},
                    timeout=5.0,
                )
                if response.status_code != 200:
                    logger.warning(f"Heartbeat returned {response.status_code}")
            except httpx.RequestError as e:
                logger.warning(f"Heartbeat failed: {e}")

            time.sleep(self.interval)


def main():
    scheduler_url = os.getenv("SCHEDULER_URL", "http://localhost:8000")
    instance_id = os.getenv("PYLET_INSTANCE_ID", f"inst-{os.getpid()}")
    interval = float(os.getenv("HEARTBEAT_INTERVAL", "30"))

    sender = HeartbeatSender(scheduler_url, instance_id, interval)

    # Handle signals
    def signal_handler(signum, frame):
        logger.info("Stopping heartbeat...")
        sender.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    sender.run()


if __name__ == "__main__":
    main()
```

### Step 3: Update Startup Script with Heartbeat

Update `scripts/start_model.sh`:

```bash
#!/bin/bash
# Start model with health monitoring and heartbeat

set -e

MODEL_ID="${MODEL_ID:?MODEL_ID required}"
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
SCHEDULER_URL="${SCHEDULER_URL:-http://localhost:8000}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-30}"

# Track child PIDs
PIDS=()

cleanup() {
    echo "Received shutdown signal"

    # Stop heartbeat
    if [ -n "$HEARTBEAT_PID" ]; then
        kill -TERM "$HEARTBEAT_PID" 2>/dev/null || true
    fi

    # Deregister from scheduler
    python scripts/register_with_scheduler.py --deregister || true

    # Stop model
    if [ -n "$MODEL_PID" ]; then
        kill -TERM "$MODEL_PID" 2>/dev/null || true
        wait "$MODEL_PID" 2>/dev/null || true
    fi

    exit 0
}

trap cleanup SIGTERM SIGINT

# Start model in background
echo "Starting $MODEL_BACKEND for $MODEL_ID on port $PORT"
if [ "$MODEL_BACKEND" = "vllm" ]; then
    vllm serve "$MODEL_ID" --port "$PORT" --host 0.0.0.0 &
elif [ "$MODEL_BACKEND" = "sglang" ]; then
    python -m sglang.launch_server --model "$MODEL_ID" --port "$PORT" --host 0.0.0.0 &
else
    echo "Unknown backend: $MODEL_BACKEND"
    exit 1
fi
MODEL_PID=$!

# Wait for model to be healthy
python scripts/wait_for_health.py

# Register with scheduler
python scripts/register_with_scheduler.py

# Start heartbeat in background
python scripts/heartbeat.py &
HEARTBEAT_PID=$!

echo "Model running (PID: $MODEL_PID, Heartbeat: $HEARTBEAT_PID)"

# Wait for model process
wait $MODEL_PID
```

### Step 4: Test Health Monitoring

```python
import subprocess
import time
import os
import httpx


def test_health_check_endpoint():
    """Test that model exposes health endpoint."""
    # This requires an actual model running
    port = os.getenv("TEST_MODEL_PORT", "8000")
    endpoint = f"http://localhost:{port}"

    response = httpx.get(f"{endpoint}/health", timeout=5.0)
    assert response.status_code == 200


def test_heartbeat_sends_to_scheduler():
    """Test heartbeat sends to scheduler."""
    # Start mock scheduler that tracks heartbeats
    heartbeats_received = []

    # ... (mock scheduler implementation)

    # Start heartbeat process
    proc = subprocess.Popen(
        ["python", "scripts/heartbeat.py"],
        env={
            **os.environ,
            "SCHEDULER_URL": "http://localhost:8080",
            "HEARTBEAT_INTERVAL": "1",
        },
    )

    time.sleep(3)
    proc.terminate()

    assert len(heartbeats_received) >= 2
```

## Test Strategy

### Unit Tests

```python
def test_wait_for_health_timeout():
    """Test health check timeout."""
    from scripts.wait_for_health import wait_for_health

    result = wait_for_health("localhost:99999", timeout=2.0)
    assert result is False


def test_heartbeat_sender_interval():
    """Test heartbeat interval."""
    # Mock and verify interval timing
    pass
```

### Integration Tests

```bash
# Start a simple HTTP server as mock model
python -m http.server 8000 &
MODEL_PID=$!

# Test health wait
PORT=8000 python scripts/wait_for_health.py

# Cleanup
kill $MODEL_PID
```

## Acceptance Criteria

- [ ] Health check waits for model readiness
- [ ] Multiple health endpoint formats supported (vLLM, sglang)
- [ ] Heartbeat sends periodically to scheduler
- [ ] Heartbeat stops on shutdown signal
- [ ] Timeout and retry logic works correctly

## Next Steps

Proceed to [PYLET-005](PYLET-005-phase1-integration-tests.md) for Phase 1 integration tests.

## Code References

- vLLM health endpoint: `/health`
- sglang health endpoint: `/v1/models`
