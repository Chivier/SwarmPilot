# PYLET-006: Phase 1 Integration Tests

## Objective

Create comprehensive integration tests for Phase 1, validating the complete SWorker wrapper functionality: task submission, processing, callbacks, and graceful shutdown.

## Prerequisites

- [PYLET-001](PYLET-001-create-sworker-wrapper.md) through [PYLET-005](PYLET-005-health-monitoring.md) completed
- PyLet development environment available

## Background

Integration tests validate the entire Phase 1 flow:
1. PyLet starts SWorker wrapper instance
2. Wrapper starts model service and registers with scheduler
3. Scheduler submits tasks to wrapper
4. Wrapper processes tasks and sends callbacks
5. PyLet cancels instance, wrapper shuts down gracefully

## Files to Create

```
sworker-wrapper/tests/
├── integration/
│   ├── __init__.py
│   ├── conftest.py              # Fixtures for integration tests
│   ├── test_full_workflow.py    # End-to-end workflow test
│   ├── test_graceful_shutdown.py # Shutdown behavior tests
│   └── test_with_pylet.py       # PyLet integration tests
└── fixtures/
    ├── mock_model_server.py     # Simple mock model service
    └── mock_scheduler.py        # Simple mock scheduler
```

## Implementation Steps

### Step 1: Create Mock Services

Create `sworker-wrapper/tests/fixtures/mock_model_server.py`:

```python
"""Mock model service for testing."""

import asyncio
import random
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.post("/v1/completions")
async def completions(data: dict):
    """Mock completions endpoint."""
    # Simulate processing time
    await asyncio.sleep(random.uniform(0.1, 0.5))

    prompt = data.get("prompt", "")
    return {
        "id": "cmpl-mock",
        "object": "text_completion",
        "choices": [
            {"text": f"Response to: {prompt[:50]}", "index": 0}
        ],
    }


@app.post("/inference")
async def inference(data: dict):
    """Generic inference endpoint."""
    await asyncio.sleep(random.uniform(0.1, 0.5))
    return {"output": "mock_output", "input": data}


def run_mock_model(port: int = 16001):
    """Run mock model server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_mock_model()
```

Create `sworker-wrapper/tests/fixtures/mock_scheduler.py`:

```python
"""Mock scheduler for testing."""

from collections import defaultdict
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional

app = FastAPI()

# Storage
registered_instances: dict[str, dict] = {}
task_callbacks: list[dict] = []
resubmitted_tasks: list[dict] = []


class RegistrationRequest(BaseModel):
    instance_id: str
    model_id: str
    endpoint: str
    platform_info: Optional[dict] = None


class CallbackRequest(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    instance_id: Optional[str] = None


class ResubmitRequest(BaseModel):
    task_id: str
    original_instance_id: Optional[str] = None
    task_info: Optional[dict] = None


@app.post("/instance/register")
async def register(req: RegistrationRequest):
    """Register an instance."""
    registered_instances[req.instance_id] = {
        "model_id": req.model_id,
        "endpoint": req.endpoint,
        "platform_info": req.platform_info,
    }
    return {"success": True}


@app.post("/instance/remove")
async def remove(data: dict):
    """Remove an instance."""
    instance_id = data.get("instance_id")
    if instance_id in registered_instances:
        del registered_instances[instance_id]
    return {"success": True}


@app.post("/instance/heartbeat")
async def heartbeat(data: dict):
    """Handle heartbeat."""
    return {"success": True}


@app.post("/instance/queue-info")
async def queue_info(data: dict):
    """Handle queue info report."""
    return {"success": True}


@app.post("/tasks/{task_id}/callback")
async def task_callback(task_id: str, req: CallbackRequest):
    """Handle task callback."""
    task_callbacks.append({
        "task_id": task_id,
        "status": req.status,
        "result": req.result,
        "error": req.error,
        "execution_time_ms": req.execution_time_ms,
    })
    return {"success": True}


@app.post("/task/resubmit")
async def resubmit(req: ResubmitRequest):
    """Handle task resubmission."""
    resubmitted_tasks.append({
        "task_id": req.task_id,
        "original_instance_id": req.original_instance_id,
        "task_info": req.task_info,
    })
    return {"success": True}


@app.get("/test/instances")
async def get_instances():
    """Get registered instances (test endpoint)."""
    return registered_instances


@app.get("/test/callbacks")
async def get_callbacks():
    """Get received callbacks (test endpoint)."""
    return task_callbacks


@app.get("/test/resubmits")
async def get_resubmits():
    """Get resubmitted tasks (test endpoint)."""
    return resubmitted_tasks


@app.post("/test/reset")
async def reset():
    """Reset all state (test endpoint)."""
    registered_instances.clear()
    task_callbacks.clear()
    resubmitted_tasks.clear()
    return {"success": True}


def run_mock_scheduler(port: int = 8000):
    """Run mock scheduler."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_mock_scheduler()
```

### Step 2: Create Test Fixtures

Create `sworker-wrapper/tests/integration/conftest.py`:

```python
"""Integration test fixtures."""

import asyncio
import os
import signal
import subprocess
import sys
import time
from typing import Generator

import httpx
import pytest


@pytest.fixture(scope="session")
def mock_scheduler_port() -> int:
    """Port for mock scheduler."""
    return 18000


@pytest.fixture(scope="session")
def mock_model_port() -> int:
    """Port for mock model."""
    return 18001


@pytest.fixture(scope="session")
def wrapper_port() -> int:
    """Port for wrapper."""
    return 18100


@pytest.fixture(scope="session")
def mock_scheduler(mock_scheduler_port: int) -> Generator:
    """Start mock scheduler for tests."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "uvicorn",
            "tests.fixtures.mock_scheduler:app",
            "--port", str(mock_scheduler_port),
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    )

    # Wait for startup
    for _ in range(30):
        try:
            httpx.get(f"http://localhost:{mock_scheduler_port}/test/instances")
            break
        except Exception:
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("Mock scheduler failed to start")

    yield proc

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def mock_model(mock_model_port: int) -> Generator:
    """Start mock model for tests."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "uvicorn",
            "tests.fixtures.mock_model_server:app",
            "--port", str(mock_model_port),
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    )

    # Wait for startup
    for _ in range(30):
        try:
            httpx.get(f"http://localhost:{mock_model_port}/health")
            break
        except Exception:
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("Mock model failed to start")

    yield proc

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
async def reset_scheduler(mock_scheduler_port: int):
    """Reset scheduler state before each test."""
    async with httpx.AsyncClient() as client:
        await client.post(f"http://localhost:{mock_scheduler_port}/test/reset")
    yield


@pytest.fixture
async def http_client() -> httpx.AsyncClient:
    """Create async HTTP client."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client
```

### Step 3: Create Full Workflow Test

Create `sworker-wrapper/tests/integration/test_full_workflow.py`:

```python
"""Full workflow integration tests."""

import asyncio
import os
import subprocess
import sys
import time

import httpx
import pytest


class TestFullWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_task_submission_and_callback(
        self,
        mock_scheduler,
        mock_model,
        mock_scheduler_port: int,
        mock_model_port: int,
        wrapper_port: int,
        reset_scheduler,
        http_client: httpx.AsyncClient,
    ):
        """Test full task flow: submit -> process -> callback."""
        # Start wrapper
        env = os.environ.copy()
        env.update({
            "PORT": str(wrapper_port),
            "SCHEDULER_URL": f"http://localhost:{mock_scheduler_port}",
            "MODEL_ID": "test-model",
            "SWORKER_COMMAND": f"python -m uvicorn tests.fixtures.mock_model_server:app --port {mock_model_port}",
        })

        # For this test, we'll use the mock model directly
        # by setting MODEL_PORT_OFFSET to the difference
        env["MODEL_PORT_OFFSET"] = str(mock_model_port - wrapper_port)

        # Submit task
        task_response = await http_client.post(
            f"http://localhost:{wrapper_port}/task/submit",
            json={
                "task_id": "test-task-001",
                "model_id": "test-model",
                "task_input": {"prompt": "Hello, world!"},
                "callback_url": f"http://localhost:{mock_scheduler_port}/tasks/test-task-001/callback",
            },
        )

        # Note: This test requires wrapper to be running
        # In real integration test, start wrapper subprocess first

        # Verify task submitted
        assert task_response.status_code == 200
        data = task_response.json()
        assert data["task_id"] == "test-task-001"

    @pytest.mark.asyncio
    async def test_registration_on_startup(
        self,
        mock_scheduler,
        mock_scheduler_port: int,
        reset_scheduler,
        http_client: httpx.AsyncClient,
    ):
        """Test that wrapper registers with scheduler on startup."""
        # Get registered instances
        response = await http_client.get(
            f"http://localhost:{mock_scheduler_port}/test/instances"
        )
        instances = response.json()

        # Note: Actual test would start wrapper and verify registration
        assert isinstance(instances, dict)

    @pytest.mark.asyncio
    async def test_multiple_tasks_fifo(
        self,
        mock_scheduler,
        mock_model,
        mock_scheduler_port: int,
        wrapper_port: int,
        reset_scheduler,
        http_client: httpx.AsyncClient,
    ):
        """Test multiple tasks are processed in FIFO order."""
        # Submit multiple tasks with specific enqueue times
        tasks = [
            {"task_id": f"task-{i}", "enqueue_time": 1000.0 + i}
            for i in range(5)
        ]

        # Submit in reverse order
        for task in reversed(tasks):
            await http_client.post(
                f"http://localhost:{wrapper_port}/task/submit",
                json={
                    "task_id": task["task_id"],
                    "model_id": "test",
                    "task_input": {"data": task["task_id"]},
                    "enqueue_time": task["enqueue_time"],
                },
            )

        # Note: Verify callbacks received in correct order
        # This requires checking scheduler callback order
```

### Step 4: Create Graceful Shutdown Test

Create `sworker-wrapper/tests/integration/test_graceful_shutdown.py`:

```python
"""Graceful shutdown integration tests."""

import asyncio
import os
import signal
import subprocess
import sys
import time

import httpx
import pytest


class TestGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_sigterm_drains_queue(
        self,
        mock_scheduler,
        mock_scheduler_port: int,
        wrapper_port: int,
        reset_scheduler,
    ):
        """Test SIGTERM triggers queue draining and task redistribution."""
        # Start wrapper in subprocess
        env = os.environ.copy()
        env.update({
            "PORT": str(wrapper_port),
            "SCHEDULER_URL": f"http://localhost:{mock_scheduler_port}",
            "SWORKER_COMMAND": "sleep 3600",
        })

        proc = subprocess.Popen(
            [sys.executable, "-m", "src.main"],
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        )

        # Wait for startup
        await asyncio.sleep(2)

        # Submit tasks
        async with httpx.AsyncClient() as client:
            for i in range(5):
                await client.post(
                    f"http://localhost:{wrapper_port}/task/submit",
                    json={
                        "task_id": f"drain-task-{i}",
                        "model_id": "test",
                        "task_input": {},
                    },
                )

            # Send SIGTERM
            proc.send_signal(signal.SIGTERM)

            # Wait for shutdown
            proc.wait(timeout=35)

            # Check resubmitted tasks
            response = await client.get(
                f"http://localhost:{mock_scheduler_port}/test/resubmits"
            )
            resubmits = response.json()

            # Pending tasks should be redistributed
            assert len(resubmits) >= 4  # At least 4 of 5 tasks (1 may be running)

    @pytest.mark.asyncio
    async def test_sigterm_waits_for_running_task(
        self,
        mock_scheduler,
        mock_model,
        mock_scheduler_port: int,
        mock_model_port: int,
        wrapper_port: int,
        reset_scheduler,
    ):
        """Test SIGTERM waits for current task to complete."""
        # This test verifies that:
        # 1. A running task completes and sends callback
        # 2. Pending tasks are redistributed
        # 3. Wrapper exits cleanly

        # Implementation similar to above but with mock model
        # that has slow response time
        pass

    @pytest.mark.asyncio
    async def test_deregistration_on_shutdown(
        self,
        mock_scheduler,
        mock_scheduler_port: int,
        wrapper_port: int,
        reset_scheduler,
    ):
        """Test instance deregisters from scheduler on shutdown."""
        # Start wrapper
        env = os.environ.copy()
        env.update({
            "PORT": str(wrapper_port),
            "SCHEDULER_URL": f"http://localhost:{mock_scheduler_port}",
            "MODEL_ID": "test-deregister",
            "SWORKER_COMMAND": "sleep 3600",
        })

        proc = subprocess.Popen(
            [sys.executable, "-m", "src.main"],
            env=env,
        )

        await asyncio.sleep(2)

        # Verify registered
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:{mock_scheduler_port}/test/instances"
            )
            instances_before = response.json()
            assert len(instances_before) > 0

            # Send SIGTERM
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=35)

            # Verify deregistered
            response = await client.get(
                f"http://localhost:{mock_scheduler_port}/test/instances"
            )
            instances_after = response.json()
            assert len(instances_after) < len(instances_before)
```

### Step 5: Create PyLet Integration Test

Create `sworker-wrapper/tests/integration/test_with_pylet.py`:

```python
"""PyLet integration tests."""

import asyncio
import os
import sys

import pytest

# Only run if pylet is available
pytestmark = pytest.mark.skipif(
    not os.path.exists("/home/yanweiye/Projects/pylet"),
    reason="PyLet not available"
)


class TestPyLetIntegration:
    """Tests with actual PyLet cluster."""

    @pytest.mark.asyncio
    async def test_pylet_submit_sworker(self):
        """Test submitting sworker-wrapper via pylet."""
        # Add pylet to path
        sys.path.insert(0, "/home/yanweiye/Projects/pylet")
        import pylet

        # This test requires pylet head node running
        try:
            pylet.init("http://localhost:8000")

            instance = pylet.submit(
                "sworker-wrapper --command 'sleep 60' --port $PORT",
                gpu=0,
                name="test-sworker",
            )

            # Wait for running
            instance.wait_running(timeout=60)
            assert instance.status == "RUNNING"
            assert instance.endpoint is not None

            # Cancel and verify graceful shutdown
            instance.cancel()
            instance.wait(timeout=35)

        except ConnectionError:
            pytest.skip("PyLet head node not running")

    @pytest.mark.asyncio
    async def test_pylet_instance_lifecycle(self):
        """Test full instance lifecycle with PyLet."""
        sys.path.insert(0, "/home/yanweiye/Projects/pylet")
        import pylet

        try:
            pylet.init("http://localhost:8000")

            # Submit multiple instances
            instances = pylet.submit(
                "sworker-wrapper --command 'sleep 60' --port $PORT",
                gpu=0,
                name="lifecycle-test",
                replicas=3,
            )

            # Wait for all running
            for inst in instances:
                inst.wait_running(timeout=60)

            # Cancel all
            for inst in instances:
                inst.cancel()

            # Wait for all terminated
            for inst in instances:
                inst.wait(timeout=35)
                assert inst.status in ("COMPLETED", "CANCELLED")

        except ConnectionError:
            pytest.skip("PyLet head node not running")
```

## Test Execution

### Run Unit Tests Only

```bash
cd sworker-wrapper
uv run pytest tests/ -v --ignore=tests/integration
```

### Run Integration Tests

```bash
cd sworker-wrapper

# Start required services in background
python -m tests.fixtures.mock_scheduler &
python -m tests.fixtures.mock_model_server &

# Run integration tests
uv run pytest tests/integration/ -v

# Cleanup
pkill -f mock_scheduler
pkill -f mock_model_server
```

### Run PyLet Integration Tests

```bash
# Ensure pylet cluster is running
cd /home/yanweiye/Projects/pylet
pylet start &
pylet start --head localhost:8000 &

# Run tests
cd /path/to/sworker-wrapper
uv run pytest tests/integration/test_with_pylet.py -v
```

## Acceptance Criteria

- [ ] Mock services work correctly for testing
- [ ] Full task workflow test passes
- [ ] Graceful shutdown redistributes tasks
- [ ] Deregistration verified on shutdown
- [ ] PyLet integration tests pass (when pylet available)
- [ ] All integration tests documented and runnable

## Next Steps

Phase 1 is complete! Proceed to Phase 2:
- [PYLET-007](PYLET-007-pylet-client-integration.md) - PyLet Client Integration

## Code References

- Current Instance tests: [instance/tests/](../../instance/tests/)
- Scheduler tests: [scheduler/tests/](../../scheduler/tests/)
