# PYLET-007: PyLet Client Integration

## Objective

Add PyLet as a dependency to the planner service and create a wrapper client that abstracts PyLet operations for SwarmPilot use cases.

## Prerequisites

- Phase 1 completed (PYLET-001 through PYLET-006)
- PyLet library available at `/home/yanweiye/Projects/pylet`

## Background

The planner needs to:
1. Initialize connection to PyLet head node
2. Submit sworker-wrapper instances
3. Track instance status
4. Cancel instances for redeployment

## Files to Create/Modify

```
planner/
├── pyproject.toml           # MODIFY: Add pylet dependency
└── src/
    └── pylet_client.py      # NEW: PyLet wrapper client
```

## Implementation Steps

### Step 1: Add PyLet Dependency

Update `planner/pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing deps ...
    "pylet @ file:///home/yanweiye/Projects/pylet",  # Local development
]

# Or for production:
# "pylet>=0.1.0",
```

### Step 2: Create PyLet Client Wrapper

Create `planner/src/pylet_client.py`:

```python
"""PyLet client wrapper for SwarmPilot Planner.

This module provides a high-level interface to PyLet for managing
sworker-wrapper instances.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger


@dataclass
class InstanceInfo:
    """Information about a deployed instance."""

    pylet_id: str
    name: str
    model_id: str
    endpoint: Optional[str] = None
    status: str = "PENDING"
    gpu_count: int = 1
    labels: dict = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


class PyLetClient:
    """High-level PyLet client for SwarmPilot.

    Wraps PyLet sync API for instance management operations.
    """

    def __init__(
        self,
        head_address: str = "http://localhost:8000",
        sworker_command_template: str = "sworker-wrapper --command '{command}' --port $PORT",
        scheduler_url: str = "http://localhost:8000",
    ):
        """Initialize PyLet client.

        Args:
            head_address: PyLet head node address.
            sworker_command_template: Template for sworker-wrapper command.
            scheduler_url: SwarmPilot scheduler URL (passed to wrapper).
        """
        self.head_address = head_address
        self.sworker_command_template = sworker_command_template
        self.scheduler_url = scheduler_url
        self._initialized = False

    def init(self) -> None:
        """Initialize connection to PyLet head node."""
        import pylet

        try:
            pylet.init(self.head_address)
            self._initialized = True
            logger.info(f"Connected to PyLet at {self.head_address}")
        except ConnectionError as e:
            logger.error(f"Failed to connect to PyLet: {e}")
            raise

    def shutdown(self) -> None:
        """Shutdown PyLet connection."""
        import pylet
        pylet.shutdown()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure PyLet is initialized."""
        if not self._initialized:
            raise RuntimeError("PyLet not initialized. Call init() first.")

    def deploy_model(
        self,
        model_id: str,
        model_command: str,
        count: int = 1,
        gpu_count: int = 1,
        name_prefix: Optional[str] = None,
        extra_env: Optional[dict] = None,
    ) -> list[InstanceInfo]:
        """Deploy model instances via PyLet.

        Args:
            model_id: Model identifier (for scheduler registration).
            model_command: Command to run the model service.
            count: Number of instances to deploy.
            gpu_count: GPUs per instance.
            name_prefix: Prefix for instance names.
            extra_env: Additional environment variables.

        Returns:
            List of InstanceInfo for deployed instances.
        """
        import pylet

        self._ensure_initialized()

        name_prefix = name_prefix or model_id.replace("/", "-").replace(":", "-")

        # Build sworker-wrapper command
        wrapper_command = self.sworker_command_template.format(command=model_command)

        # Build environment variables
        env = {
            "SCHEDULER_URL": self.scheduler_url,
            "MODEL_ID": model_id,
        }
        if extra_env:
            env.update(extra_env)

        # Build labels
        labels = {
            "model_id": model_id,
            "managed_by": "swarmpilot-planner",
        }

        logger.info(f"Deploying {count} instances of {model_id}")

        instances = []
        for i in range(count):
            instance_name = f"{name_prefix}-{i}"

            try:
                pylet_instance = pylet.submit(
                    wrapper_command,
                    name=instance_name,
                    gpu=gpu_count,
                    labels=labels,
                    env=env,
                )

                info = InstanceInfo(
                    pylet_id=pylet_instance.id,
                    name=instance_name,
                    model_id=model_id,
                    status=pylet_instance.status,
                    gpu_count=gpu_count,
                    labels=labels,
                )
                instances.append(info)
                logger.info(f"Submitted instance {instance_name} ({pylet_instance.id})")

            except Exception as e:
                logger.error(f"Failed to submit instance {instance_name}: {e}")
                raise

        return instances

    def wait_instances_ready(
        self,
        instances: list[InstanceInfo],
        timeout: float = 300.0,
    ) -> list[InstanceInfo]:
        """Wait for instances to become running.

        Args:
            instances: List of instances to wait for.
            timeout: Maximum wait time per instance.

        Returns:
            Updated list of InstanceInfo with endpoints.
        """
        import pylet

        self._ensure_initialized()

        for info in instances:
            try:
                instance = pylet.get(id=info.pylet_id)
                instance.wait_running(timeout=timeout)

                info.status = instance.status
                info.endpoint = instance.endpoint
                logger.info(f"Instance {info.name} ready at {info.endpoint}")

            except Exception as e:
                logger.error(f"Instance {info.name} failed to start: {e}")
                info.status = "FAILED"

        return instances

    def cancel_instances(
        self,
        instances: list[InstanceInfo],
        wait: bool = True,
        timeout: float = 60.0,
    ) -> None:
        """Cancel instances.

        Args:
            instances: List of instances to cancel.
            wait: Wait for instances to terminate.
            timeout: Maximum wait time per instance.
        """
        import pylet

        self._ensure_initialized()

        for info in instances:
            try:
                instance = pylet.get(id=info.pylet_id)
                instance.cancel()
                logger.info(f"Cancelled instance {info.name}")

                if wait:
                    instance.wait(timeout=timeout)
                    info.status = instance.status
                    logger.info(f"Instance {info.name} terminated: {info.status}")

            except Exception as e:
                logger.error(f"Error cancelling instance {info.name}: {e}")

    def list_model_instances(self, model_id: str) -> list[InstanceInfo]:
        """List all instances for a model.

        Args:
            model_id: Model identifier to filter by.

        Returns:
            List of InstanceInfo for matching instances.
        """
        import pylet

        self._ensure_initialized()

        pylet_instances = pylet.instances(labels={"model_id": model_id})

        return [
            InstanceInfo(
                pylet_id=inst.id,
                name=inst.name or inst.id,
                model_id=model_id,
                endpoint=inst.endpoint,
                status=inst.status,
                labels=inst.labels,
            )
            for inst in pylet_instances
        ]

    def get_workers(self) -> list[dict]:
        """Get PyLet workers.

        Returns:
            List of worker information dictionaries.
        """
        import pylet

        self._ensure_initialized()

        workers = pylet.workers()
        return [
            {
                "worker_id": w.worker_id,
                "host": w.host,
                "status": w.status,
                "total_gpus": w.total_resources.gpu_units,
                "available_gpus": w.available_resources.gpu_units,
            }
            for w in workers
        ]


# Global client instance
_pylet_client: Optional[PyLetClient] = None


def get_pylet_client() -> Optional[PyLetClient]:
    """Get the global PyLet client."""
    return _pylet_client


def create_pylet_client(
    head_address: str = "http://localhost:8000",
    scheduler_url: str = "http://localhost:8000",
) -> PyLetClient:
    """Create and initialize global PyLet client."""
    global _pylet_client
    _pylet_client = PyLetClient(
        head_address=head_address,
        scheduler_url=scheduler_url,
    )
    _pylet_client.init()
    return _pylet_client


def shutdown_pylet_client() -> None:
    """Shutdown the global PyLet client."""
    global _pylet_client
    if _pylet_client:
        _pylet_client.shutdown()
        _pylet_client = None
```

### Step 3: Create Async Variant

Create `planner/src/pylet_client_async.py`:

```python
"""Async PyLet client wrapper for SwarmPilot Planner."""

import asyncio
from typing import Optional

from loguru import logger

from src.pylet_client import InstanceInfo


class AsyncPyLetClient:
    """Async wrapper for PyLet client.

    Uses pylet.aio module for async operations.
    """

    def __init__(
        self,
        head_address: str = "http://localhost:8000",
        sworker_command_template: str = "sworker-wrapper --command '{command}' --port $PORT",
        scheduler_url: str = "http://localhost:8000",
    ):
        self.head_address = head_address
        self.sworker_command_template = sworker_command_template
        self.scheduler_url = scheduler_url
        self._initialized = False

    async def init(self) -> None:
        """Initialize connection to PyLet head node."""
        import pylet.aio as pylet

        try:
            await pylet.init(self.head_address)
            self._initialized = True
            logger.info(f"Async connected to PyLet at {self.head_address}")
        except ConnectionError as e:
            logger.error(f"Failed to connect to PyLet: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown PyLet connection."""
        import pylet.aio as pylet
        await pylet.shutdown()
        self._initialized = False

    async def deploy_model(
        self,
        model_id: str,
        model_command: str,
        count: int = 1,
        gpu_count: int = 1,
        name_prefix: Optional[str] = None,
        extra_env: Optional[dict] = None,
    ) -> list[InstanceInfo]:
        """Deploy model instances via PyLet (async)."""
        import pylet.aio as pylet

        if not self._initialized:
            raise RuntimeError("PyLet not initialized")

        name_prefix = name_prefix or model_id.replace("/", "-").replace(":", "-")
        wrapper_command = self.sworker_command_template.format(command=model_command)

        env = {"SCHEDULER_URL": self.scheduler_url, "MODEL_ID": model_id}
        if extra_env:
            env.update(extra_env)

        labels = {"model_id": model_id, "managed_by": "swarmpilot-planner"}

        logger.info(f"Deploying {count} instances of {model_id}")

        # Submit instances in parallel
        async def submit_one(i: int) -> InstanceInfo:
            instance_name = f"{name_prefix}-{i}"
            pylet_instance = await pylet.submit(
                wrapper_command,
                name=instance_name,
                gpu=gpu_count,
                labels=labels,
                env=env,
            )
            return InstanceInfo(
                pylet_id=pylet_instance.id,
                name=instance_name,
                model_id=model_id,
                status=pylet_instance.status,
                gpu_count=gpu_count,
                labels=labels,
            )

        instances = await asyncio.gather(*[submit_one(i) for i in range(count)])
        return list(instances)

    async def cancel_instances(
        self,
        instances: list[InstanceInfo],
        wait: bool = True,
        timeout: float = 60.0,
    ) -> None:
        """Cancel instances in parallel (async)."""
        import pylet.aio as pylet

        if not self._initialized:
            raise RuntimeError("PyLet not initialized")

        async def cancel_one(info: InstanceInfo) -> None:
            instance = await pylet.get(id=info.pylet_id)
            await instance.cancel()
            if wait:
                await instance.wait(timeout=timeout)
            info.status = instance.status

        await asyncio.gather(*[cancel_one(info) for info in instances])
```

### Step 4: Create Tests

Create `planner/tests/test_pylet_client.py`:

```python
"""Tests for PyLet client wrapper."""

import pytest
from unittest.mock import MagicMock, patch

from src.pylet_client import PyLetClient, InstanceInfo


class TestPyLetClient:
    """Tests for PyLetClient."""

    @pytest.fixture
    def client(self):
        """Create a PyLet client without initializing."""
        return PyLetClient(
            head_address="http://localhost:8000",
            scheduler_url="http://scheduler:8000",
        )

    def test_deploy_model_not_initialized(self, client):
        """Test error when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            client.deploy_model("model", "command")

    @patch("pylet.init")
    @patch("pylet.submit")
    def test_deploy_model(self, mock_submit, mock_init, client):
        """Test model deployment."""
        client._initialized = True

        mock_instance = MagicMock()
        mock_instance.id = "inst-123"
        mock_instance.status = "PENDING"
        mock_submit.return_value = mock_instance

        instances = client.deploy_model(
            model_id="test-model",
            model_command="python serve.py",
            count=2,
        )

        assert len(instances) == 2
        assert mock_submit.call_count == 2
        assert all(inst.model_id == "test-model" for inst in instances)

    @patch("pylet.init")
    @patch("pylet.get")
    def test_cancel_instances(self, mock_get, mock_init, client):
        """Test instance cancellation."""
        client._initialized = True

        mock_instance = MagicMock()
        mock_instance.status = "CANCELLED"
        mock_get.return_value = mock_instance

        instances = [
            InstanceInfo(pylet_id="1", name="inst-1", model_id="test"),
            InstanceInfo(pylet_id="2", name="inst-2", model_id="test"),
        ]

        client.cancel_instances(instances)

        assert mock_get.call_count == 2
        assert mock_instance.cancel.call_count == 2
```

## Test Strategy

### Unit Tests

```bash
cd planner
uv run pytest tests/test_pylet_client.py -v
```

### Integration with PyLet

```bash
# Start PyLet cluster
cd /home/yanweiye/Projects/pylet
pylet start &
pylet start --head localhost:8000 &

# Run integration test
cd planner
uv run python -c "
from src.pylet_client import create_pylet_client
client = create_pylet_client()
workers = client.get_workers()
print(f'Workers: {workers}')
"
```

## Acceptance Criteria

- [ ] PyLet added as dependency
- [ ] Sync client wrapper implemented
- [ ] Async client wrapper implemented
- [ ] Deploy model creates sworker-wrapper instances
- [ ] Cancel instances works correctly
- [ ] Labels used for tracking
- [ ] All tests pass

## Next Steps

Proceed to [PYLET-008](PYLET-008-instance-lifecycle.md) for instance lifecycle management.

## Code References

- PyLet sync API: [pylet/_sync_api.py](/home/yanweiye/Projects/pylet/pylet/_sync_api.py)
- PyLet async API: [pylet/aio/__init__.py](/home/yanweiye/Projects/pylet/pylet/aio/__init__.py)
