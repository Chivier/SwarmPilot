"""Fixtures for PyLet integration tests.

This module provides pytest fixtures for testing PyLet + Planner optimization
integration. Fixtures include:
- pylet_local_cluster: Session-scoped local PyLet cluster with 5 workers
- dummy_script_path: Path to the dummy model server script
- cleanup_instances: Function-scoped cleanup of PyLet instances
- planner_client: Async client for planner API endpoints

PYLET-013: PyLet Optimizer Integration E2E Test
PYLET-015: E2E tests using Planner deploy API
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

import httpx
import pytest
import pytest_asyncio

# Try to import pylet - skip tests if not available
try:
    import pylet
    from pylet import Instance

    PYLET_AVAILABLE = True
except ImportError:
    PYLET_AVAILABLE = False
    pylet = None  # type: ignore


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires PyLet cluster)",
    )


@pytest.fixture(scope="session")
def check_pylet_available():
    """Check if PyLet is available, skip if not."""
    if not PYLET_AVAILABLE:
        pytest.skip("pylet package not available")


@pytest.fixture(scope="session")
def pylet_local_cluster(check_pylet_available, request):
    """Connect to PyLet cluster for integration tests.

    This fixture connects to an external PyLet cluster. Due to signal handler
    limitations in pytest's threading context, pylet.local_cluster() cannot
    reliably spawn workers.

    Start the cluster externally before running tests:
        # Option 1: Use the helper script
        python scripts/run_pylet_integration_tests.py

        # Option 2: Manual startup
        # Terminal 1: Start PyLet head
        pylet start --port 5100

        # Terminal 2-6: Start workers (5 workers)
        pylet start --head localhost:5100 --cpu-cores 2

    Yields:
        None (PyLet is initialized to external cluster)
    """
    pylet_head_url = request.config.getoption("--pylet-head")

    print(f"\n[FIXTURE] Connecting to external PyLet cluster at {pylet_head_url}...")

    try:
        pylet.init(pylet_head_url)
        print(f"[FIXTURE] Connected to PyLet at {pylet_head_url}")

        # Wait a moment for connection to stabilize
        time.sleep(1)

        workers = pylet.workers()
        print(f"[FIXTURE] Found {len(workers)} workers")

        if len(workers) == 0:
            pytest.skip(
                f"No workers found at {pylet_head_url}. "
                "Start PyLet cluster before running integration tests: "
                "python scripts/run_pylet_integration_tests.py"
            )

        # Log worker details
        for w in workers:
            print(f"[FIXTURE]   Worker: {w.id}")

        yield None  # No cluster object, but PyLet is initialized

    except Exception as e:
        pytest.skip(
            f"Could not connect to PyLet at {pylet_head_url}: {e}. "
            "Start PyLet cluster before running integration tests: "
            "python scripts/run_pylet_integration_tests.py"
        )


@pytest.fixture(scope="session")
def pylet_head_address(pylet_local_cluster) -> str:
    """Get the PyLet head address.

    Returns:
        str: PyLet head URL (e.g., "http://localhost:5100")
    """
    if pylet_local_cluster is not None:
        return pylet_local_cluster.address
    return os.getenv("PYLET_HEAD", "http://localhost:5100")


@pytest.fixture(scope="session")
def dummy_script_path() -> str:
    """Get absolute path to dummy model server script.

    Returns:
        str: Absolute path to dummy_model_server.py
    """
    path = Path(__file__).parent / "dummy_model_server.py"
    if not path.exists():
        pytest.fail(f"Dummy model server not found at {path}")
    return str(path.absolute())


@pytest.fixture(scope="session")
def mock_scheduler_server() -> Generator[dict, None, None]:
    """Start mock scheduler server for registration testing.

    Starts a simple FastAPI server on port 8080 that mocks
    scheduler registration endpoints.

    Yields:
        dict: State dict with registered instances.
    """
    try:
        import uvicorn
        from fastapi import FastAPI
    except ImportError:
        pytest.skip("fastapi/uvicorn not available for mock scheduler")

    app = FastAPI()
    state = {
        "instances": {},
        "drain_requests": [],
    }

    @app.post("/v1/instance/register")
    async def register(data: dict):
        instance_id = data.get("instance_id", "unknown")
        state["instances"][instance_id] = data
        return {"success": True, "instance_id": instance_id}

    @app.post("/v1/instance/drain")
    async def drain(data: dict):
        state["drain_requests"].append(data)
        return {"success": True}

    @app.get("/v1/instance/drain/status")
    async def drain_status(pylet_id: str = ""):
        return {"drained": True, "remaining_tasks": 0}

    @app.post("/v1/instance/remove")
    async def remove(data: dict):
        instance_id = data.get("instance_id", "unknown")
        state["instances"].pop(instance_id, None)
        return {"success": True}

    @app.get("/v1/health")
    async def health():
        return {"status": "healthy", "mock": True}

    # Start server in background thread
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    time.sleep(0.5)

    yield state

    # Cleanup
    server.should_exit = True


@pytest.fixture
def cleanup_instances(pylet_local_cluster):
    """Cleanup all PyLet instances after each test.

    This fixture runs AFTER the test and cancels/deletes
    all instances managed by swarmpilot.

    Yields:
        None
    """
    yield

    # Cleanup after test
    if not PYLET_AVAILABLE:
        return

    try:
        for instance in pylet.instances():
            labels = instance.labels or {}
            # Only cleanup swarmpilot-managed instances
            if labels.get("managed_by") == "swarmpilot":
                try:
                    instance.cancel(delete=True)
                except Exception:
                    pass  # Instance may already be cancelled
    except Exception:
        pass  # PyLet may not be initialized


@pytest.fixture
def deployed_instances(pylet_local_cluster):
    """Track and cleanup deployed instances within a test.

    Provides a list that tests can append instances to.
    All instances in the list are cleaned up after the test.

    Yields:
        list: Empty list to track deployed instances.
    """
    instances = []
    yield instances

    # Cleanup all tracked instances
    cancelled_count = 0
    for instance in instances:
        try:
            instance.cancel(delete=True)
            cancelled_count += 1
        except Exception:
            pass

    # Give workers time to fully release resources before next test
    if cancelled_count > 0:
        print(f"\n[CLEANUP] Cancelled {cancelled_count} instances, waiting for workers...")
        time.sleep(2)  # Allow workers to clean up


class DeploymentHelper:
    """Helper class for deploying dummy models in tests."""

    def __init__(self, script_path: str, python_executable: str | None = None):
        """Initialize deployment helper.

        Args:
            script_path: Path to dummy_model_server.py
            python_executable: Path to Python executable (defaults to sys.executable)
        """
        import sys

        self.script_path = script_path
        self.python_executable = python_executable or sys.executable

    def deploy_dummy_model(
        self,
        model_id: str,
        throughput: float = 1.0,
        cpu: int = 1,
        labels: dict | None = None,
    ) -> Any:
        """Deploy a dummy model server.

        Args:
            model_id: Model identifier to report.
            throughput: Simulated requests per second.
            cpu: CPU cores to request.
            labels: Additional labels to set.

        Returns:
            PyLet Instance handle.
        """
        instance_labels = {
            "model_id": model_id,
            "managed_by": "swarmpilot",
            "backend": "dummy",
        }
        if labels:
            instance_labels.update(labels)

        return pylet.submit(
            f"{self.python_executable} {self.script_path}",
            cpu=cpu,
            gpu=0,
            env={
                "MODEL_ID": model_id,
                "THROUGHPUT": str(throughput),
            },
            labels=instance_labels,
        )


@pytest.fixture
def deployment_helper(dummy_script_path: str) -> DeploymentHelper:
    """Create deployment helper for dummy models.

    Args:
        dummy_script_path: Path to dummy model server.

    Returns:
        DeploymentHelper instance.
    """
    return DeploymentHelper(dummy_script_path)


# ==============================================================================
# Planner API Test Fixtures (PYLET-015)
# ==============================================================================


@pytest.fixture(scope="session")
def planner_server(
    pylet_local_cluster,
    mock_scheduler_server,
    dummy_script_path: str,
    request,
) -> Generator[str, None, None]:
    """Start planner server configured for integration tests.

    The planner is configured with:
    - PYLET_ENABLED=true
    - PYLET_REUSE_CLUSTER=true (reuses existing PyLet connection)
    - PYLET_CUSTOM_COMMAND=python dummy_model_server.py
    - PYLET_GPU_COUNT=0 (CPU-only for tests)

    Yields:
        str: Base URL of planner server (e.g., "http://localhost:8081")
    """
    import uvicorn

    pylet_head_url = request.config.getoption("--pylet-head")
    planner_port = 8081  # Use different port from mock scheduler (8080)

    # Build custom command - note $PORT is handled by PyLet
    python_exec = sys.executable
    custom_command = f"{python_exec} {dummy_script_path}"

    # Set environment for planner
    os.environ["PYLET_ENABLED"] = "true"
    os.environ["PYLET_HEAD_URL"] = pylet_head_url
    os.environ["PYLET_REUSE_CLUSTER"] = "true"
    os.environ["PYLET_CUSTOM_COMMAND"] = custom_command
    os.environ["PYLET_GPU_COUNT"] = "0"
    os.environ["PYLET_CPU_COUNT"] = "1"
    os.environ["PYLET_BACKEND"] = "dummy"  # Won't be used with custom_command
    os.environ["SCHEDULER_URL"] = "http://localhost:8080"  # Mock scheduler

    # Import planner app (must be after setting env vars)
    # Force reimport of config with new env vars
    import importlib

    from swarmpilot.planner import config as config_module

    importlib.reload(config_module)

    from swarmpilot.planner.api import app

    # Start planner in background thread
    uvicorn_config = uvicorn.Config(
        app, host="0.0.0.0", port=planner_port, log_level="warning"
    )
    server = uvicorn.Server(uvicorn_config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    planner_url = f"http://localhost:{planner_port}"

    # Wait for server to be ready by polling health endpoint
    # The lifespan context needs time to initialize PyLet connection
    max_wait = 30  # Maximum wait time in seconds
    poll_interval = 0.5
    waited = 0

    print(f"\n[FIXTURE] Waiting for planner server at {planner_url}...")
    while waited < max_wait:
        try:
            resp = httpx.get(f"{planner_url}/v1/health", timeout=2.0)
            if resp.status_code == 200:
                print(f"[FIXTURE] Planner server ready after {waited:.1f}s")
                break
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(poll_interval)
        waited += poll_interval
    else:
        pytest.fail(f"Planner server did not become ready within {max_wait}s")

    print(f"[FIXTURE] Planner server started at {planner_url}")

    yield planner_url

    # Cleanup
    server.should_exit = True
    print("[FIXTURE] Planner server stopped")


@pytest_asyncio.fixture
async def planner_client(
    planner_server: str,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async HTTP client for planner API.

    Args:
        planner_server: Base URL of planner server.

    Yields:
        httpx.AsyncClient configured for planner API.
    """
    async with httpx.AsyncClient(
        base_url=planner_server,
        timeout=60.0,
    ) as client:
        # Verify planner is healthy
        resp = await client.get("/v1/health")
        assert resp.status_code == 200, f"Planner not healthy: {resp.text}"

        yield client


@pytest_asyncio.fixture
async def cleanup_via_planner(planner_client: httpx.AsyncClient):
    """Cleanup all PyLet instances via planner API after test.

    This fixture cleans up instances using the planner's terminate-all
    endpoint instead of direct PyLet calls.

    Yields:
        None
    """
    yield

    # Cleanup after test
    try:
        resp = await planner_client.post("/v1/terminate-all")
        if resp.status_code == 200:
            data = resp.json()
            print(f"[CLEANUP] Terminated {data.get('succeeded', 0)} instances")
    except Exception as e:
        print(f"[CLEANUP] Failed to terminate instances: {e}")
