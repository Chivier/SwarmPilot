# PYLET-003: Signal Handling

## Objective

Implement graceful shutdown signal handling for PyLet-managed model instances. When PyLet sends SIGTERM/SIGINT, the instance must deregister from the scheduler and terminate cleanly.

## Prerequisites

- PYLET-001 completed (Direct model deployment)
- PYLET-002 completed (Model registration)

## Background

PyLet terminates instances by:
1. Sending SIGTERM (default grace period: 30s)
2. Waiting for process to exit
3. Sending SIGKILL if still running after grace period

The model instance must:
1. Catch SIGTERM/SIGINT
2. Deregister from scheduler
3. Allow in-flight requests to complete (if possible)
4. Terminate model process
5. Exit cleanly

## Implementation Steps

### Step 1: Create Signal Handler Module

Create `scripts/signal_handler.py`:

```python
"""Signal handling for graceful shutdown."""

import os
import signal
import sys
import time
from typing import Callable, Optional

import httpx
from loguru import logger


class GracefulShutdown:
    """Manages graceful shutdown on signal."""

    def __init__(
        self,
        scheduler_url: str,
        instance_id: str,
        grace_period: float = 25.0,
    ):
        """Initialize graceful shutdown handler.

        Args:
            scheduler_url: Scheduler URL for deregistration.
            instance_id: Instance identifier.
            grace_period: Max time for graceful shutdown.
        """
        self.scheduler_url = scheduler_url
        self.instance_id = instance_id
        self.grace_period = grace_period
        self._shutdown_requested = False
        self._cleanup_callbacks: list[Callable] = []

    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add a cleanup callback to run on shutdown."""
        self._cleanup_callbacks.append(callback)

    def setup(self) -> None:
        """Install signal handlers."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        logger.info("Signal handlers installed")

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle received signal."""
        if self._shutdown_requested:
            logger.warning("Forced exit on second signal")
            sys.exit(1)

        self._shutdown_requested = True
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received {signal_name}, initiating graceful shutdown")

        self._graceful_shutdown()

    def _graceful_shutdown(self) -> None:
        """Execute graceful shutdown sequence."""
        start_time = time.time()

        # Step 1: Deregister from scheduler
        self._deregister()

        # Step 2: Run cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                remaining = self.grace_period - (time.time() - start_time)
                if remaining <= 0:
                    break
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback error: {e}")

        # Step 3: Exit
        elapsed = time.time() - start_time
        logger.info(f"Graceful shutdown complete in {elapsed:.1f}s")
        sys.exit(0)

    def _deregister(self) -> None:
        """Deregister from scheduler."""
        try:
            response = httpx.post(
                f"{self.scheduler_url}/model/deregister",
                json={"instance_id": self.instance_id},
                timeout=5.0,
            )
            if response.status_code == 200:
                logger.info("Deregistered from scheduler")
            else:
                logger.warning(f"Deregistration returned {response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Deregistration failed: {e}")

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested
```

### Step 2: Update Startup Script

Update `scripts/start_model.sh`:

```bash
#!/bin/bash
# Start model with graceful shutdown handling

set -e

MODEL_ID="${MODEL_ID:?MODEL_ID required}"
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
SCHEDULER_URL="${SCHEDULER_URL:-http://localhost:8000}"
GRACE_PERIOD="${GRACE_PERIOD:-25}"

# Trap signals and forward to model
cleanup() {
    echo "Received shutdown signal"

    # Deregister from scheduler
    python -c "
import httpx
import os
try:
    httpx.post(
        '$SCHEDULER_URL/model/deregister',
        json={'instance_id': os.getenv('PYLET_INSTANCE_ID', 'unknown')},
        timeout=5.0
    )
    print('Deregistered from scheduler')
except Exception as e:
    print(f'Deregistration error: {e}')
"

    # Forward signal to model
    if [ -n "$MODEL_PID" ]; then
        kill -TERM "$MODEL_PID" 2>/dev/null || true
        wait "$MODEL_PID" 2>/dev/null || true
    fi

    exit 0
}

trap cleanup SIGTERM SIGINT

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

# Wait for model to be healthy
python scripts/wait_for_health.py

# Register with scheduler
python scripts/register_with_scheduler.py

# Wait for model process (or signal)
wait $MODEL_PID
```

### Step 3: Create Health Wait Script

Create `scripts/wait_for_health.py`:

```python
#!/usr/bin/env python3
"""Wait for model health check to pass."""

import os
import sys
import time
import httpx
from loguru import logger


def wait_for_health(endpoint: str, timeout: float = 120.0) -> bool:
    """Wait for model health check."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"http://{endpoint}/health", timeout=5.0)
            if response.status_code == 200:
                logger.info(f"Model healthy at {endpoint}")
                return True
        except httpx.RequestError:
            pass
        time.sleep(2.0)
    return False


def main():
    port = os.getenv("PORT", "8000")
    endpoint = f"localhost:{port}"
    timeout = float(os.getenv("HEALTH_TIMEOUT", "120"))

    if not wait_for_health(endpoint, timeout):
        logger.error("Model failed to become healthy")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 4: Test Signal Handling

```python
import subprocess
import time
import signal
import os


def test_graceful_shutdown():
    """Test graceful shutdown on SIGTERM."""
    # Start model via script
    proc = subprocess.Popen(
        ["bash", "scripts/start_model.sh"],
        env={
            **os.environ,
            "MODEL_ID": "test-model",
            "PORT": "8000",
            "SCHEDULER_URL": "http://localhost:8080",
        },
    )

    # Wait for startup
    time.sleep(30)

    # Send SIGTERM
    proc.send_signal(signal.SIGTERM)

    # Wait for exit
    exit_code = proc.wait(timeout=30)
    assert exit_code == 0, f"Expected clean exit, got {exit_code}"
```

## Test Strategy

### Unit Tests

```python
def test_graceful_shutdown_deregisters():
    """Test that shutdown deregisters from scheduler."""
    deregistered = False

    def mock_deregister():
        nonlocal deregistered
        deregistered = True

    shutdown = GracefulShutdown(
        scheduler_url="http://mock",
        instance_id="test",
    )
    shutdown._deregister = mock_deregister
    shutdown._graceful_shutdown()

    assert deregistered
```

### Integration Tests

```bash
# Start model
MODEL_ID=test PORT=8000 bash scripts/start_model.sh &
PID=$!

# Wait for ready
sleep 30

# Send SIGTERM
kill -TERM $PID

# Check clean exit
wait $PID
echo "Exit code: $?"
```

## Acceptance Criteria

- [ ] SIGTERM triggers graceful shutdown
- [ ] SIGINT triggers graceful shutdown
- [ ] Deregistration happens before exit
- [ ] Model process terminated cleanly
- [ ] Second signal forces immediate exit
- [ ] Grace period respected

## Next Steps

Proceed to [PYLET-004](PYLET-004-health-monitoring.md) for health monitoring.

## Code References

- PyLet signal handling: [pylet/worker.py](/home/yanweiye/Projects/pylet/pylet/worker.py)
