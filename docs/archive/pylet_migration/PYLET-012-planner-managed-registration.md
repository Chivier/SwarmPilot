# PYLET-012: Planner-Managed Registration

## Objective

Move instance registration/deregistration from the model startup script to the planner. The planner will register instances with the scheduler after `pylet.submit()` succeeds and deregister them before calling `pylet.cancel()`.

## Prerequisites

- PYLET-006 through PYLET-010 completed
- PyLet client integration working
- Instance lifecycle management implemented

## Background

### Current Flow (Phase 1)

In Phase 1, models self-register with the scheduler:

```
Planner                      PyLet                    Model Instance
   │                           │                            │
   ├──submit()────────────────▶│                            │
   │                           ├───start model─────────────▶│
   │                           │                            │
   │                           │                 ┌──────────┤
   │                           │                 │ wait for │
   │                           │                 │ healthy  │
   │                           │                 └──────────┤
   │                           │                            │
   │                           │          ┌─────────────────┤
   │                           │          │ register with   │
   │                           │          │ scheduler       │
   │                           │          └─────────────────┤
   │                           │                            │
   ├──cancel()────────────────▶│                            │
   │                           ├───SIGTERM─────────────────▶│
   │                           │                            │
   │                           │          ┌─────────────────┤
   │                           │          │ deregister from │
   │                           │          │ scheduler       │
   │                           │          └─────────────────┤
   │                           │                            ▼
```

### Target Flow (This Task)

Planner manages registration:

```
Planner                      PyLet                    Model Instance        Scheduler
   │                           │                            │                   │
   ├──submit()────────────────▶│                            │                   │
   │                           ├───start model─────────────▶│                   │
   │                           │                            │                   │
   │◀──instance.endpoint───────┤                            │                   │
   │                           │                            │                   │
   ├──register(endpoint)──────────────────────────────────────────────────────▶│
   │                           │                            │                   │
   │                           │                   [model serves requests]      │
   │                           │                            │                   │
   ├──deregister(instance_id)─────────────────────────────────────────────────▶│
   │                           │                            │                   │
   ├──cancel()────────────────▶│                            │                   │
   │                           ├───SIGTERM─────────────────▶│                   │
   │                           │                            ▼                   │
```

**Benefits:**
- Single source of truth for instance state in planner
- Simpler model startup scripts (no registration logic)
- Coordinated lifecycle: registration only after instance is ready
- Graceful deregistration before shutdown (no race conditions)

## Files to Create/Modify

```
planner/
└── src/
    ├── scheduler_client.py       # NEW: HTTP client for scheduler registration
    ├── instance_manager.py       # MODIFY: Add register/deregister calls
    └── pylet_client.py           # MODIFY: Update deploy flow

scripts/
└── start_model.sh                # MODIFY: Remove registration logic
```

## Implementation Steps

### Step 1: Create Scheduler Client

Create `planner/src/scheduler_client.py`:

```python
"""Scheduler client for instance registration.

This module handles registering and deregistering model instances
with the SwarmPilot scheduler.
"""

import httpx
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class RegistrationInfo:
    """Information for scheduler registration."""

    instance_id: str
    model_id: str
    endpoint: str
    gpu_count: int = 1
    backend: str = "vllm"
    capacity: int = 1


class SchedulerClient:
    """HTTP client for scheduler registration operations."""

    def __init__(
        self,
        scheduler_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        retries: int = 3,
    ):
        """Initialize scheduler client.

        Args:
            scheduler_url: Base URL of the scheduler service.
            timeout: Request timeout in seconds.
            retries: Number of retry attempts.
        """
        self.scheduler_url = scheduler_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def register_instance(self, info: RegistrationInfo) -> bool:
        """Register an instance with the scheduler.

        Args:
            info: Registration information.

        Returns:
            True if registration succeeded.
        """
        client = await self._get_client()

        payload = {
            "instance_id": info.instance_id,
            "model_id": info.model_id,
            "endpoint": info.endpoint,
            "gpu_count": info.gpu_count,
            "backend": info.backend,
            "capacity": info.capacity,
        }

        for attempt in range(self.retries):
            try:
                response = await client.post(
                    f"{self.scheduler_url}/model/register",
                    json=payload,
                )

                if response.status_code == 200:
                    logger.info(
                        f"Registered instance {info.instance_id} "
                        f"({info.model_id}) at {info.endpoint}"
                    )
                    return True

                logger.warning(
                    f"Registration failed: {response.status_code} - "
                    f"{response.text}"
                )

            except httpx.RequestError as e:
                logger.warning(f"Registration error (attempt {attempt + 1}): {e}")

            if attempt < self.retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Failed to register instance {info.instance_id}")
        return False

    async def deregister_instance(self, instance_id: str) -> bool:
        """Deregister an instance from the scheduler.

        Args:
            instance_id: Instance identifier to deregister.

        Returns:
            True if deregistration succeeded.
        """
        client = await self._get_client()

        payload = {"instance_id": instance_id}

        for attempt in range(self.retries):
            try:
                response = await client.post(
                    f"{self.scheduler_url}/model/deregister",
                    json=payload,
                )

                if response.status_code == 200:
                    logger.info(f"Deregistered instance {instance_id}")
                    return True

                logger.warning(
                    f"Deregistration failed: {response.status_code} - "
                    f"{response.text}"
                )

            except httpx.RequestError as e:
                logger.warning(f"Deregistration error (attempt {attempt + 1}): {e}")

            if attempt < self.retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)

        logger.error(f"Failed to deregister instance {instance_id}")
        return False

    async def health_check(self) -> bool:
        """Check if scheduler is reachable.

        Returns:
            True if scheduler health check passes.
        """
        client = await self._get_client()

        try:
            response = await client.get(f"{self.scheduler_url}/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
```

### Step 2: Update Instance Manager

Modify `planner/src/instance_manager.py` to include registration:

```python
"""Instance lifecycle management with scheduler registration."""

from typing import Optional

from loguru import logger

from src.pylet_client import AsyncPyLetClient, InstanceInfo
from src.scheduler_client import SchedulerClient, RegistrationInfo


class InstanceManager:
    """Manages model instances via PyLet with scheduler registration."""

    def __init__(
        self,
        pylet_client: Optional[AsyncPyLetClient] = None,
        scheduler_client: Optional[SchedulerClient] = None,
        model_commands: Optional[dict[str, str]] = None,
    ):
        """Initialize instance manager.

        Args:
            pylet_client: PyLet client instance.
            scheduler_client: Scheduler client for registration.
            model_commands: Mapping of model_id to startup command.
        """
        self._pylet_client = pylet_client
        self._scheduler_client = scheduler_client
        self._model_commands = model_commands or {}
        self._instances: dict[str, ManagedInstance] = {}
        self._lock = asyncio.Lock()

    async def deploy_instances(
        self,
        model_id: str,
        count: int,
        gpu_count: int = 1,
        wait_ready: bool = True,
        register: bool = True,
    ) -> list[ManagedInstance]:
        """Deploy model instances with scheduler registration.

        Args:
            model_id: Model to deploy.
            count: Number of instances.
            gpu_count: GPUs per instance.
            wait_ready: Wait for instances to be ready before returning.
            register: Register instances with scheduler after ready.

        Returns:
            List of deployed managed instances.
        """
        if not self._pylet_client:
            raise RuntimeError("PyLet client not configured")

        command = self._model_commands.get(model_id)
        if not command:
            raise ValueError(f"No command registered for {model_id}")

        logger.info(f"Deploying {count} instances of {model_id}")

        # Deploy via PyLet
        infos = await self._pylet_client.deploy_model(
            model_id=model_id,
            model_command=command,
            count=count,
            gpu_count=gpu_count,
        )

        # Track instances
        managed = []
        async with self._lock:
            for info in infos:
                instance = ManagedInstance(
                    pylet_id=info.pylet_id,
                    instance_id=info.name,
                    model_id=model_id,
                    status=info.status,
                    gpu_count=gpu_count,
                )
                self._instances[info.pylet_id] = instance
                managed.append(instance)

        # Wait for instances to be ready
        if wait_ready:
            managed = await self.wait_instances_ready(managed)

            # Register with scheduler
            if register and self._scheduler_client:
                await self._register_instances(managed)

        return managed

    async def _register_instances(
        self,
        instances: list[ManagedInstance],
    ) -> None:
        """Register instances with scheduler.

        Args:
            instances: Instances to register.
        """
        if not self._scheduler_client:
            logger.warning("No scheduler client configured, skipping registration")
            return

        for inst in instances:
            if inst.status != "RUNNING" or not inst.endpoint:
                logger.warning(
                    f"Skipping registration for {inst.instance_id}: "
                    f"status={inst.status}, endpoint={inst.endpoint}"
                )
                continue

            reg_info = RegistrationInfo(
                instance_id=inst.instance_id,
                model_id=inst.model_id,
                endpoint=inst.endpoint,
                gpu_count=inst.gpu_count,
            )

            success = await self._scheduler_client.register_instance(reg_info)
            inst.registered = success

            if success:
                logger.info(f"Registered {inst.instance_id} with scheduler")
            else:
                logger.error(f"Failed to register {inst.instance_id}")

    async def terminate_instances(
        self,
        model_id: str,
        count: int,
        deregister: bool = True,
    ) -> list[ManagedInstance]:
        """Terminate instances with scheduler deregistration.

        Args:
            model_id: Model identifier.
            count: Number of instances to terminate.
            deregister: Deregister from scheduler before cancelling.

        Returns:
            List of terminated instances.
        """
        if not self._pylet_client:
            raise RuntimeError("PyLet client not configured")

        # Find instances to terminate
        async with self._lock:
            candidates = [
                inst for inst in self._instances.values()
                if inst.model_id == model_id and inst.status == "RUNNING"
            ]

        if not candidates:
            logger.warning(f"No running instances of {model_id} to terminate")
            return []

        to_terminate = candidates[:count]

        # Deregister from scheduler FIRST
        if deregister and self._scheduler_client:
            await self._deregister_instances(to_terminate)

        # Cancel via PyLet (triggers graceful shutdown)
        infos = [
            InstanceInfo(
                pylet_id=inst.pylet_id,
                name=inst.instance_id,
                model_id=inst.model_id,
            )
            for inst in to_terminate
        ]
        await self._pylet_client.cancel_instances(infos, wait=True)

        # Remove from tracking
        async with self._lock:
            for inst in to_terminate:
                if inst.pylet_id in self._instances:
                    del self._instances[inst.pylet_id]

        logger.info(f"Terminated {len(to_terminate)} instances of {model_id}")
        return to_terminate

    async def _deregister_instances(
        self,
        instances: list[ManagedInstance],
    ) -> None:
        """Deregister instances from scheduler.

        Args:
            instances: Instances to deregister.
        """
        if not self._scheduler_client:
            return

        for inst in instances:
            if not inst.registered:
                continue

            success = await self._scheduler_client.deregister_instance(
                inst.instance_id
            )

            if success:
                inst.registered = False
                logger.info(f"Deregistered {inst.instance_id} from scheduler")
            else:
                logger.warning(
                    f"Failed to deregister {inst.instance_id}, "
                    "proceeding with cancellation anyway"
                )
```

### Step 3: Update Model Startup Script

Simplify `scripts/start_model.sh` to remove registration:

```bash
#!/bin/bash
# Start model service (no registration - handled by planner)

set -e

MODEL_ID="${MODEL_ID:?MODEL_ID required}"
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"

echo "Starting model: $MODEL_ID"
echo "Backend: $MODEL_BACKEND"
echo "Port: $PORT"

# Trap signals for graceful shutdown
cleanup() {
    echo "Received shutdown signal, stopping model..."
    # No deregistration needed - planner handles it
    kill -TERM "$MODEL_PID" 2>/dev/null || true
    wait "$MODEL_PID" 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start model service
if [ "$MODEL_BACKEND" = "vllm" ]; then
    vllm serve "$MODEL_ID" --port "$PORT" --host 0.0.0.0 &
elif [ "$MODEL_BACKEND" = "sglang" ]; then
    python -m sglang.launch_server --model "$MODEL_ID" --port "$PORT" --host 0.0.0.0 &
else
    echo "Unknown backend: $MODEL_BACKEND"
    exit 1
fi

MODEL_PID=$!

# Wait for model process
wait $MODEL_PID
```

### Step 4: Update PYLET-002 (Model Registration)

Since registration is now handled by the planner, PYLET-002 in Phase 1 should be updated to be optional/fallback. The model CAN still self-register if run standalone, but when managed by the planner, the planner handles registration.

### Step 5: Create Tests

Create `planner/tests/test_scheduler_client.py`:

```python
"""Tests for scheduler client."""

import pytest
from unittest.mock import AsyncMock, patch

from src.scheduler_client import SchedulerClient, RegistrationInfo


class TestSchedulerClient:
    """Tests for SchedulerClient."""

    @pytest.fixture
    def client(self):
        """Create scheduler client."""
        return SchedulerClient(scheduler_url="http://localhost:8000")

    @pytest.mark.asyncio
    async def test_register_instance_success(self, client):
        """Test successful registration."""
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_http.post.return_value = mock_response
            mock_get.return_value = mock_http

            info = RegistrationInfo(
                instance_id="test-1",
                model_id="test-model",
                endpoint="localhost:8001",
            )

            result = await client.register_instance(info)

            assert result is True
            mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_deregister_instance_success(self, client):
        """Test successful deregistration."""
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_http.post.return_value = mock_response
            mock_get.return_value = mock_http

            result = await client.deregister_instance("test-1")

            assert result is True
            mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_with_retry(self, client):
        """Test registration with retries on failure."""
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()

            # Fail twice, succeed on third
            mock_fail = AsyncMock()
            mock_fail.status_code = 500
            mock_success = AsyncMock()
            mock_success.status_code = 200

            mock_http.post.side_effect = [mock_fail, mock_fail, mock_success]
            mock_get.return_value = mock_http

            info = RegistrationInfo(
                instance_id="test-1",
                model_id="test-model",
                endpoint="localhost:8001",
            )

            result = await client.register_instance(info)

            assert result is True
            assert mock_http.post.call_count == 3


class TestInstanceManagerWithRegistration:
    """Tests for InstanceManager with scheduler registration."""

    @pytest.mark.asyncio
    async def test_deploy_registers_with_scheduler(self):
        """Test that deploy calls scheduler registration."""
        from src.instance_manager import InstanceManager, ManagedInstance
        from src.pylet_client import InstanceInfo

        mock_pylet = AsyncMock()
        mock_scheduler = AsyncMock()

        mock_pylet.deploy_model.return_value = [
            InstanceInfo(
                pylet_id="pylet-1",
                name="test-0",
                model_id="test",
                status="RUNNING",
                endpoint="localhost:8001",
            )
        ]

        mock_scheduler.register_instance.return_value = True

        manager = InstanceManager(
            pylet_client=mock_pylet,
            scheduler_client=mock_scheduler,
        )
        manager.register_model_command("test", "sleep 3600")

        # Mock wait_instances_ready to return instances with endpoints
        async def mock_wait(instances, timeout=300.0):
            for inst in instances:
                inst.status = "RUNNING"
                inst.endpoint = "localhost:8001"
            return instances

        manager.wait_instances_ready = mock_wait

        instances = await manager.deploy_instances("test", count=1)

        assert len(instances) == 1
        mock_scheduler.register_instance.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_deregisters_first(self):
        """Test that terminate deregisters before cancelling."""
        from src.instance_manager import InstanceManager

        mock_pylet = AsyncMock()
        mock_scheduler = AsyncMock()
        mock_scheduler.deregister_instance.return_value = True

        manager = InstanceManager(
            pylet_client=mock_pylet,
            scheduler_client=mock_scheduler,
        )

        # Add a running instance
        inst = ManagedInstance(
            pylet_id="pylet-1",
            instance_id="test-0",
            model_id="test",
            status="RUNNING",
            endpoint="localhost:8001",
            registered=True,
        )
        manager._instances["pylet-1"] = inst

        await manager.terminate_instances("test", count=1)

        # Verify order: deregister called before cancel
        mock_scheduler.deregister_instance.assert_called_once_with("test-0")
        mock_pylet.cancel_instances.assert_called_once()
```

## Test Strategy

### Unit Tests

```bash
cd planner
uv run pytest tests/test_scheduler_client.py -v
```

### Integration Test

```bash
# Start scheduler mock
python -c "
from fastapi import FastAPI
import uvicorn

app = FastAPI()
instances = {}

@app.post('/model/register')
async def register(data: dict):
    instances[data['instance_id']] = data
    return {'success': True}

@app.post('/model/deregister')
async def deregister(data: dict):
    instances.pop(data['instance_id'], None)
    return {'success': True}

@app.get('/instances')
async def list_instances():
    return {'instances': instances}

uvicorn.run(app, port=8000)
" &

# Start PyLet and test
cd planner
uv run python -c "
import asyncio
from src.instance_manager import InstanceManager
from src.scheduler_client import SchedulerClient
from src.pylet_client_async import AsyncPyLetClient

async def test():
    pylet = AsyncPyLetClient()
    await pylet.init()

    scheduler = SchedulerClient()

    mgr = InstanceManager(pylet, scheduler)
    mgr.register_model_command('test', 'sleep 3600')

    # Deploy - should register
    instances = await mgr.deploy_instances('test', count=1)
    print(f'Deployed and registered: {instances}')

    # Terminate - should deregister first
    await mgr.terminate_instances('test', count=1)
    print('Terminated and deregistered')

    await scheduler.close()
    await pylet.shutdown()

asyncio.run(test())
"
```

## Acceptance Criteria

- [ ] SchedulerClient created with register/deregister methods
- [ ] InstanceManager integrates scheduler registration
- [ ] Registration happens after instance is RUNNING
- [ ] Deregistration happens BEFORE cancel is called
- [ ] Model startup script simplified (no registration code)
- [ ] Retry logic with exponential backoff
- [ ] All tests pass

## Architecture Impact

### Phase 1 Changes

PYLET-002 (Model Registration) becomes optional:
- Models CAN still self-register for standalone testing
- When managed by planner, planner handles registration
- Startup script checks `PLANNER_MANAGED` env var to skip self-registration

### Scheduler API Requirements

The scheduler must support:
- `POST /model/register` - Register instance
- `POST /model/deregister` - Deregister instance
- Both endpoints should be idempotent

## Next Steps

Proceed to [PYLET-011](PYLET-011-phase2-integration-tests.md) for Phase 2 integration tests.

## Code References

- Scheduler API: [scheduler/src/api.py](../../scheduler/src/api.py)
- Current registration: [PYLET-002](PYLET-002-model-registration.md)
- Instance manager: [PYLET-007](PYLET-007-instance-lifecycle.md)
