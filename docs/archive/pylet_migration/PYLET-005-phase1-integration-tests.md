# PYLET-005: Phase 1 Integration Tests

## Objective

Create comprehensive integration tests for Phase 1 (direct model deployment via PyLet). These tests validate the complete workflow from PyLet deployment through scheduler registration to request handling.

## Prerequisites

- PYLET-001 through PYLET-004 completed
- PyLet cluster available for testing
- Model binaries (vLLM/sglang) installed

## Test Scenarios

### Scenario 1: Basic Deployment Flow

```
1. Submit model via PyLet
2. Wait for instance to be running
3. Verify model health endpoint responds
4. Verify scheduler registration
5. Send test inference request
6. Cancel instance
7. Verify deregistration
```

### Scenario 2: Multiple Instances

```
1. Deploy 3 instances of same model
2. Verify all register with scheduler
3. Cancel 2 instances
4. Verify remaining instance still works
```

### Scenario 3: Graceful Shutdown

```
1. Deploy model
2. Send long-running request
3. Send SIGTERM via PyLet cancel
4. Verify request completes
5. Verify deregistration
```

## Test Implementation

### Test Fixtures

Create `tests/integration/conftest.py`:

```python
"""Integration test fixtures."""

import os
import pytest
import asyncio
from typing import Optional

# Skip if no PyLet cluster
pytestmark = pytest.mark.skipif(
    os.getenv("PYLET_HEAD_ADDRESS") is None,
    reason="PYLET_HEAD_ADDRESS not set",
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def pylet_address() -> str:
    """Get PyLet head address."""
    return os.getenv("PYLET_HEAD_ADDRESS", "http://localhost:8000")


@pytest.fixture(scope="session")
def scheduler_address() -> str:
    """Get scheduler address."""
    return os.getenv("SCHEDULER_ADDRESS", "http://localhost:8080")


@pytest.fixture(scope="session")
def test_model_id() -> str:
    """Get test model ID."""
    return os.getenv("TEST_MODEL_ID", "facebook/opt-125m")


@pytest.fixture
async def pylet_client(pylet_address):
    """Create PyLet client."""
    import pylet
    pylet.init(pylet_address)
    yield pylet
    pylet.shutdown()


@pytest.fixture
async def mock_scheduler(scheduler_address):
    """Start mock scheduler for testing."""
    import uvicorn
    from fastapi import FastAPI
    import threading

    app = FastAPI()
    registered_instances = {}
    heartbeats = {}

    @app.post("/model/register")
    async def register(data: dict):
        instance_id = data.get("instance_id")
        registered_instances[instance_id] = data
        return {"success": True}

    @app.post("/model/deregister")
    async def deregister(data: dict):
        instance_id = data.get("instance_id")
        registered_instances.pop(instance_id, None)
        return {"success": True}

    @app.post("/instance/heartbeat")
    async def heartbeat(data: dict):
        instance_id = data.get("instance_id")
        heartbeats[instance_id] = heartbeats.get(instance_id, 0) + 1
        return {"success": True}

    @app.get("/instances")
    async def list_instances():
        return {"instances": registered_instances}

    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    yield {"instances": registered_instances, "heartbeats": heartbeats}

    server.should_exit = True
```

### Basic Deployment Test

Create `tests/integration/test_deployment.py`:

```python
"""Integration tests for model deployment."""

import pytest
import asyncio
import httpx


class TestBasicDeployment:
    """Test basic model deployment flow."""

    @pytest.mark.asyncio
    async def test_deploy_and_register(
        self,
        pylet_client,
        mock_scheduler,
        test_model_id,
    ):
        """Test deploying a model and registering with scheduler."""
        import pylet

        # Deploy model
        instance = pylet.submit(
            f"bash scripts/start_model.sh",
            gpu=1,
            name="test-model-1",
            env={
                "MODEL_ID": test_model_id,
                "MODEL_BACKEND": "vllm",
                "SCHEDULER_URL": "http://localhost:8080",
            },
            labels={"test": "true"},
        )

        try:
            # Wait for running
            instance.wait_running(timeout=300)
            endpoint = instance.endpoint

            # Verify health
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"http://{endpoint}/health")
                assert response.status_code == 200

            # Verify registration (with retry)
            for _ in range(10):
                if mock_scheduler["instances"]:
                    break
                await asyncio.sleep(1)

            assert len(mock_scheduler["instances"]) > 0

        finally:
            instance.cancel()
            instance.wait()

    @pytest.mark.asyncio
    async def test_inference_request(
        self,
        pylet_client,
        test_model_id,
    ):
        """Test sending inference request to deployed model."""
        import pylet

        instance = pylet.submit(
            f"bash scripts/start_model.sh",
            gpu=1,
            env={
                "MODEL_ID": test_model_id,
                "MODEL_BACKEND": "vllm",
                "SCHEDULER_URL": "http://localhost:8080",
            },
        )

        try:
            instance.wait_running(timeout=300)
            endpoint = instance.endpoint

            # Send inference request
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"http://{endpoint}/v1/completions",
                    json={
                        "model": test_model_id,
                        "prompt": "Hello, world!",
                        "max_tokens": 10,
                    },
                )
                assert response.status_code == 200
                result = response.json()
                assert "choices" in result

        finally:
            instance.cancel()
            instance.wait()


class TestMultipleInstances:
    """Test multiple instance deployment."""

    @pytest.mark.asyncio
    async def test_deploy_multiple(
        self,
        pylet_client,
        mock_scheduler,
        test_model_id,
    ):
        """Test deploying multiple instances."""
        import pylet

        instances = []
        try:
            # Deploy 3 instances
            for i in range(3):
                instance = pylet.submit(
                    f"bash scripts/start_model.sh",
                    gpu=1,
                    name=f"test-model-{i}",
                    env={
                        "MODEL_ID": test_model_id,
                        "MODEL_BACKEND": "vllm",
                        "SCHEDULER_URL": "http://localhost:8080",
                    },
                )
                instances.append(instance)

            # Wait for all to be running
            for instance in instances:
                instance.wait_running(timeout=300)

            # Verify all registered
            await asyncio.sleep(5)
            assert len(mock_scheduler["instances"]) == 3

            # Cancel 2
            instances[0].cancel()
            instances[1].cancel()
            instances[0].wait()
            instances[1].wait()

            # Verify 1 remaining
            await asyncio.sleep(3)
            assert len(mock_scheduler["instances"]) == 1

        finally:
            for instance in instances:
                try:
                    instance.cancel()
                    instance.wait(timeout=30)
                except Exception:
                    pass


class TestGracefulShutdown:
    """Test graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_deregisters(
        self,
        pylet_client,
        mock_scheduler,
        test_model_id,
    ):
        """Test that shutdown triggers deregistration."""
        import pylet

        instance = pylet.submit(
            f"bash scripts/start_model.sh",
            gpu=1,
            env={
                "MODEL_ID": test_model_id,
                "MODEL_BACKEND": "vllm",
                "SCHEDULER_URL": "http://localhost:8080",
            },
        )

        try:
            instance.wait_running(timeout=300)

            # Verify registered
            await asyncio.sleep(5)
            assert len(mock_scheduler["instances"]) == 1

            # Cancel (triggers SIGTERM)
            instance.cancel()
            instance.wait(timeout=60)

            # Verify deregistered
            await asyncio.sleep(3)
            assert len(mock_scheduler["instances"]) == 0

        except Exception:
            instance.cancel()
            raise

    @pytest.mark.asyncio
    async def test_heartbeat_sent(
        self,
        pylet_client,
        mock_scheduler,
        test_model_id,
    ):
        """Test that heartbeats are sent periodically."""
        import pylet

        instance = pylet.submit(
            f"bash scripts/start_model.sh",
            gpu=1,
            env={
                "MODEL_ID": test_model_id,
                "MODEL_BACKEND": "vllm",
                "SCHEDULER_URL": "http://localhost:8080",
                "HEARTBEAT_INTERVAL": "5",
            },
        )

        try:
            instance.wait_running(timeout=300)

            # Wait for heartbeats
            await asyncio.sleep(20)

            # Should have at least 2-3 heartbeats
            total_heartbeats = sum(mock_scheduler["heartbeats"].values())
            assert total_heartbeats >= 2

        finally:
            instance.cancel()
            instance.wait()
```

### Test Runner Script

Create `tests/integration/run_tests.sh`:

```bash
#!/bin/bash
# Run Phase 1 integration tests

set -e

export PYLET_HEAD_ADDRESS="${PYLET_HEAD_ADDRESS:-http://localhost:8000}"
export SCHEDULER_ADDRESS="${SCHEDULER_ADDRESS:-http://localhost:8080}"
export TEST_MODEL_ID="${TEST_MODEL_ID:-facebook/opt-125m}"

echo "Phase 1 Integration Tests"
echo "========================="
echo "PyLet: $PYLET_HEAD_ADDRESS"
echo "Model: $TEST_MODEL_ID"
echo ""

# Check PyLet is available
if ! curl -s "$PYLET_HEAD_ADDRESS/health" > /dev/null 2>&1; then
    echo "ERROR: PyLet not available at $PYLET_HEAD_ADDRESS"
    exit 1
fi

# Run tests
cd "$(dirname "$0")/../.."
uv run pytest tests/integration/ -v --tb=short "$@"
```

## Acceptance Criteria

- [ ] Basic deployment test passes
- [ ] Model health check works
- [ ] Scheduler registration verified
- [ ] Inference request succeeds
- [ ] Multiple instance deployment works
- [ ] Graceful shutdown deregisters
- [ ] Heartbeats sent periodically

## Running Tests

```bash
# Prerequisites
export PYLET_HEAD_ADDRESS=http://localhost:8000
export TEST_MODEL_ID=facebook/opt-125m

# Start PyLet cluster
pylet start &
pylet start --head localhost:8000 &

# Run tests
./tests/integration/run_tests.sh
```

## Next Steps

Phase 1 complete. Proceed to Phase 2:
- [PYLET-006](PYLET-006-pylet-client-integration.md) - PyLet Client Integration

## Code References

- PyLet submit API: [pylet/_sync_api.py](/home/yanweiye/Projects/pylet/pylet/_sync_api.py)
- Scheduler API: [scheduler/src/api.py](../../scheduler/src/api.py)
