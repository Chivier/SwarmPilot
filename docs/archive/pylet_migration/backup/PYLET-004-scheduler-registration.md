# PYLET-004: Scheduler Registration

## Objective

Implement scheduler registration and deregistration for the SWorker wrapper. The wrapper must register with the SwarmPilot scheduler on startup and deregister during graceful shutdown.

## Prerequisites

- [PYLET-001](PYLET-001-create-sworker-wrapper.md) completed
- [PYLET-002](PYLET-002-implement-task-queue.md) completed
- [PYLET-003](PYLET-003-signal-handling.md) completed

## Background

The SWorker wrapper maintains the same scheduler interaction pattern as the current Instance service:

1. **Startup**: Register with scheduler to receive tasks
2. **Runtime**: Accept tasks, process them, send callbacks
3. **Shutdown**: Deregister to stop receiving new tasks

## Files to Create/Modify

```
sworker-wrapper/src/
├── scheduler_client.py   # NEW: Scheduler communication
├── main.py               # MODIFY: Call registration on startup
└── api.py                # MODIFY: Add registration endpoints
```

## Implementation Steps

### Step 1: Create Scheduler Client

Create `sworker-wrapper/src/scheduler_client.py`:

```python
"""Scheduler client for registration and callbacks."""

import asyncio
import platform
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from loguru import logger

from src.config import get_config


@dataclass
class PlatformInfo:
    """Platform information for registration."""
    software_name: str
    software_version: str
    hardware_name: str


def get_platform_info() -> PlatformInfo:
    """Detect platform information."""
    # Try to get GPU info
    hardware_name = "CPU"
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            hardware_name = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return PlatformInfo(
        software_name=platform.system(),
        software_version=platform.release(),
        hardware_name=hardware_name,
    )


class SchedulerClient:
    """HTTP client for scheduler communication."""

    def __init__(
        self,
        scheduler_url: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize scheduler client."""
        config = get_config()
        self.scheduler_url = scheduler_url or config.scheduler_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._registered = False

    @property
    def is_registered(self) -> bool:
        """Check if registered with scheduler."""
        return self._registered

    async def register(
        self,
        model_id: str,
        platform_info: Optional[PlatformInfo] = None,
    ) -> bool:
        """Register instance with scheduler.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.

        Returns:
            True if registration successful.
        """
        config = get_config()

        if platform_info is None:
            platform_info = get_platform_info()

        registration_data = {
            "instance_id": config.instance_id,
            "model_id": model_id,
            "endpoint": f"http://localhost:{config.port}",
            "platform_info": {
                "software_name": platform_info.software_name,
                "software_version": platform_info.software_version,
                "hardware_name": platform_info.hardware_name,
            },
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.scheduler_url}/instance/register",
                        json=registration_data,
                    )
                    response.raise_for_status()
                    result = response.json()

                if result.get("success", False):
                    self._registered = True
                    logger.info(
                        f"Registered with scheduler: {config.instance_id}"
                    )
                    return True
                else:
                    logger.error(f"Registration failed: {result.get('error')}")
                    return False

            except Exception as e:
                logger.warning(f"Registration attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        logger.error("Registration failed after all retries")
        return False

    async def deregister(self) -> bool:
        """Deregister instance from scheduler.

        Returns:
            True if deregistration successful.
        """
        if not self._registered:
            logger.info("Not registered, skipping deregistration")
            return True

        config = get_config()

        deregistration_data = {
            "instance_id": config.instance_id,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.scheduler_url}/instance/remove",
                    json=deregistration_data,
                )
                response.raise_for_status()
                result = response.json()

            if result.get("success", False):
                self._registered = False
                logger.info(f"Deregistered from scheduler: {config.instance_id}")
                return True
            else:
                logger.warning(f"Deregistration failed: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Deregistration error: {e}")
            return False

    async def send_heartbeat(self) -> bool:
        """Send heartbeat to scheduler.

        Returns:
            True if heartbeat acknowledged.
        """
        config = get_config()

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.scheduler_url}/instance/heartbeat",
                    json={"instance_id": config.instance_id},
                )
                return response.status_code == 200
        except Exception:
            return False

    async def report_queue_info(
        self,
        queued_count: int,
        running_task_id: Optional[str] = None,
        estimated_time_ms: Optional[float] = None,
    ) -> bool:
        """Report queue information to scheduler.

        Args:
            queued_count: Number of queued tasks.
            running_task_id: Currently running task ID.
            estimated_time_ms: Estimated completion time.

        Returns:
            True if report sent successfully.
        """
        config = get_config()

        queue_info = {
            "instance_id": config.instance_id,
            "queued_count": queued_count,
        }
        if running_task_id:
            queue_info["running_task_id"] = running_task_id
        if estimated_time_ms:
            queue_info["estimated_time_ms"] = estimated_time_ms

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.scheduler_url}/instance/queue-info",
                    json=queue_info,
                )
                return response.status_code == 200
        except Exception:
            return False


# Global scheduler client
_scheduler_client: Optional[SchedulerClient] = None


def get_scheduler_client() -> Optional[SchedulerClient]:
    """Get the global scheduler client."""
    return _scheduler_client


def create_scheduler_client() -> SchedulerClient:
    """Create and set the global scheduler client."""
    global _scheduler_client
    _scheduler_client = SchedulerClient()
    return _scheduler_client
```

### Step 2: Update Main to Call Registration

Update `sworker-wrapper/src/main.py` to register on startup:

```python
async def run_wrapper() -> None:
    """Run the wrapper with full lifecycle management."""
    config = get_config()
    lifecycle = get_lifecycle()

    # Start model process
    await lifecycle.transition_to(LifecycleState.STARTING_MODEL)
    process_manager = create_process_manager(
        command=config.command,
        port=config.model_port,
    )

    if not await process_manager.start():
        logger.error("Failed to start model process")
        await lifecycle.transition_to(LifecycleState.STOPPED)
        return

    # Wait for model to be ready
    from src.model_client import get_model_client
    model_client = get_model_client()

    if not await model_client.wait_ready(timeout=300.0):
        logger.error("Model service did not become ready")
        await process_manager.stop()
        await lifecycle.transition_to(LifecycleState.STOPPED)
        return

    # Register with scheduler
    await lifecycle.transition_to(LifecycleState.REGISTERING)

    from src.scheduler_client import create_scheduler_client
    scheduler_client = create_scheduler_client()

    if not await scheduler_client.register(model_id=config.model_id or "unknown"):
        logger.warning("Scheduler registration failed, continuing anyway")

    # Set up signal handler with deregistration
    async def deregister():
        if scheduler_client:
            await scheduler_client.deregister()

    signal_handler = create_signal_handler(
        grace_period=config.grace_period_seconds,
        on_deregister=deregister,
    )
    signal_handler.setup()

    # Transition to running
    await lifecycle.transition_to(LifecycleState.RUNNING)
    logger.info("SWorker wrapper is running")

    # Wait for shutdown
    await signal_handler.wait_for_shutdown()
```

### Step 3: Add Registration API Endpoints

Update `sworker-wrapper/src/api.py`:

```python
@app.post("/model/register")
async def register_model(model_id: str):
    """Manually trigger registration."""
    from src.scheduler_client import get_scheduler_client

    client = get_scheduler_client()
    if client is None:
        raise HTTPException(status_code=500, detail="Scheduler client not initialized")

    success = await client.register(model_id=model_id)
    if success:
        return {"success": True, "message": "Registered"}
    else:
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/model/deregister")
async def deregister_model():
    """Manually trigger deregistration."""
    from src.scheduler_client import get_scheduler_client

    client = get_scheduler_client()
    if client is None:
        raise HTTPException(status_code=500, detail="Scheduler client not initialized")

    success = await client.deregister()
    if success:
        return {"success": True, "message": "Deregistered"}
    else:
        raise HTTPException(status_code=500, detail="Deregistration failed")
```

### Step 4: Create Tests

Create `sworker-wrapper/tests/test_scheduler_client.py`:

```python
"""Tests for scheduler client."""

import pytest
from unittest.mock import AsyncMock, patch

from src.scheduler_client import SchedulerClient, PlatformInfo


class TestSchedulerClient:
    """Tests for SchedulerClient."""

    @pytest.fixture
    def client(self):
        """Create a scheduler client."""
        return SchedulerClient(scheduler_url="http://localhost:8000")

    @pytest.mark.asyncio
    async def test_register_success(self, client):
        """Test successful registration."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = AsyncMock()
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status = AsyncMock()

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            success = await client.register(model_id="test-model")
            assert success is True
            assert client.is_registered is True

    @pytest.mark.asyncio
    async def test_register_failure(self, client):
        """Test failed registration."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = AsyncMock()
            mock_response.json.return_value = {"success": False, "error": "Test error"}
            mock_response.raise_for_status = AsyncMock()

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            success = await client.register(model_id="test-model")
            assert success is False

    @pytest.mark.asyncio
    async def test_deregister_when_not_registered(self, client):
        """Test deregistration when not registered."""
        success = await client.deregister()
        assert success is True  # Should succeed silently
```

## Test Strategy

### Unit Tests

```bash
cd sworker-wrapper
uv run pytest tests/test_scheduler_client.py -v
```

### Integration Testing

```bash
# Start scheduler (in another terminal)
cd scheduler && uv run uvicorn src.main:app --port 8000

# Start wrapper
PORT=16000 MODEL_ID=test-model SWORKER_COMMAND="sleep 3600" uv run sworker-wrapper

# Verify registration in scheduler logs
```

## Acceptance Criteria

- [ ] Register with scheduler on startup
- [ ] Deregister during graceful shutdown
- [ ] Retry logic for transient failures
- [ ] Platform info detection works
- [ ] Manual registration/deregistration endpoints
- [ ] All tests pass

## Next Steps

Proceed to [PYLET-005](PYLET-005-health-monitoring.md) for health monitoring.

## Code References

- Current SchedulerClient: [instance/src/scheduler_client.py](../../instance/src/scheduler_client.py)
- Registration API: [scheduler/src/api.py](../../scheduler/src/api.py)
