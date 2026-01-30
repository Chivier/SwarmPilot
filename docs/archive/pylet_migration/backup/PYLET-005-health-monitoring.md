# PYLET-005: Health Monitoring

## Objective

Implement comprehensive health monitoring for the SWorker wrapper, including model service health checks and periodic reporting to the scheduler.

## Prerequisites

- [PYLET-001](PYLET-001-create-sworker-wrapper.md) completed
- [PYLET-002](PYLET-002-implement-task-queue.md) completed
- [PYLET-003](PYLET-003-signal-handling.md) completed
- [PYLET-004](PYLET-004-scheduler-registration.md) completed

## Background

Health monitoring serves multiple purposes:
1. **Readiness**: Ensure model service is ready before accepting tasks
2. **Liveness**: Detect and report model service failures
3. **Queue Status**: Report queue depth to scheduler for load balancing

## Files to Create/Modify

```
sworker-wrapper/src/
├── health.py    # NEW: Health monitoring service
├── api.py       # MODIFY: Enhanced health endpoint
└── main.py      # MODIFY: Start health monitor
```

## Implementation Steps

### Step 1: Create Health Monitor

Create `sworker-wrapper/src/health.py`:

```python
"""Health monitoring for SWorker wrapper."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from loguru import logger

from src.config import get_config
from src.lifecycle import get_lifecycle
from src.model_client import get_model_client
from src.scheduler_client import get_scheduler_client
from src.task_queue import get_task_queue


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Model unhealthy but wrapper running
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class HealthReport:
    """Health status report."""

    status: HealthStatus
    model_healthy: bool
    queue_size: int
    running_tasks: int
    uptime_seconds: float
    last_check: str
    details: Optional[dict] = None


class HealthMonitor:
    """Monitors health of wrapper and model service."""

    def __init__(
        self,
        check_interval: float = 10.0,
        report_interval: float = 30.0,
    ):
        """Initialize health monitor.

        Args:
            check_interval: Interval between health checks (seconds).
            report_interval: Interval between scheduler reports (seconds).
        """
        self.check_interval = check_interval
        self.report_interval = report_interval

        self._start_time = datetime.now(timezone.utc)
        self._last_model_check: Optional[datetime] = None
        self._model_healthy = False
        self._consecutive_failures = 0

        self._monitor_task: Optional[asyncio.Task] = None
        self._reporter_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._health_check_loop())
        self._reporter_task = asyncio.create_task(self._report_loop())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False

        for task in [self._monitor_task, self._reporter_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Health monitor stopped")

    async def check_health(self) -> HealthReport:
        """Perform health check.

        Returns:
            HealthReport with current status.
        """
        lifecycle = get_lifecycle()
        queue = get_task_queue()
        model_client = get_model_client()

        # Check lifecycle state
        if lifecycle.state.value.startswith("start"):
            status = HealthStatus.STARTING
        elif lifecycle.state.value in ("draining", "deregistering", "stopping"):
            status = HealthStatus.STOPPING
        else:
            status = HealthStatus.HEALTHY

        # Check model health
        model_healthy = await model_client.health_check()
        self._last_model_check = datetime.now(timezone.utc)

        if model_healthy:
            self._model_healthy = True
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self._model_healthy = False

        # Adjust status based on model health
        if status == HealthStatus.HEALTHY and not self._model_healthy:
            status = HealthStatus.DEGRADED

        # Get queue stats
        stats = await queue.get_queue_stats()

        return HealthReport(
            status=status,
            model_healthy=self._model_healthy,
            queue_size=stats.queued,
            running_tasks=stats.running,
            uptime_seconds=self.uptime_seconds,
            last_check=self._last_model_check.isoformat(),
            details={
                "lifecycle_state": lifecycle.state.value,
                "consecutive_failures": self._consecutive_failures,
                "completed_tasks": stats.completed,
                "failed_tasks": stats.failed,
            },
        )

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await self.check_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _report_loop(self) -> None:
        """Periodic scheduler reporting loop."""
        while self._running:
            try:
                await self._report_to_scheduler()
                await asyncio.sleep(self.report_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Report error: {e}")
                await asyncio.sleep(self.report_interval)

    async def _report_to_scheduler(self) -> None:
        """Report status to scheduler."""
        scheduler_client = get_scheduler_client()
        if not scheduler_client or not scheduler_client.is_registered:
            return

        queue = get_task_queue()
        stats = await queue.get_queue_stats()

        await scheduler_client.report_queue_info(
            queued_count=stats.queued,
            running_task_id=queue.current_task_id,
        )


# Global health monitor
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> Optional[HealthMonitor]:
    """Get the global health monitor."""
    return _health_monitor


def create_health_monitor() -> HealthMonitor:
    """Create and set the global health monitor."""
    global _health_monitor
    config = get_config()
    _health_monitor = HealthMonitor(
        check_interval=config.health_check_interval,
    )
    return _health_monitor
```

### Step 2: Update API with Enhanced Health Endpoint

Update `sworker-wrapper/src/api.py`:

```python
from src.health import HealthReport, HealthStatus, get_health_monitor


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status information.
    """
    monitor = get_health_monitor()

    if monitor:
        report = await monitor.check_health()
        status_code = 200 if report.status == HealthStatus.HEALTHY else 503

        return {
            "status": report.status.value,
            "model_healthy": report.model_healthy,
            "queue_size": report.queue_size,
            "running_tasks": report.running_tasks,
            "uptime_seconds": report.uptime_seconds,
            "last_check": report.last_check,
            "details": report.details,
        }
    else:
        # Fallback if monitor not started
        queue = get_task_queue()
        stats = await queue.get_queue_stats()
        return {
            "status": "healthy",
            "queue_stats": stats.model_dump(),
        }


@app.get("/ready")
async def readiness_check() -> dict:
    """Readiness check for Kubernetes/load balancer.

    Returns 200 only if the wrapper is ready to accept tasks.
    """
    from src.lifecycle import get_lifecycle

    lifecycle = get_lifecycle()

    if lifecycle.is_accepting_tasks:
        return {"ready": True}
    else:
        from fastapi import Response
        return Response(
            content='{"ready": false}',
            status_code=503,
            media_type="application/json",
        )


@app.get("/live")
async def liveness_check() -> dict:
    """Liveness check for Kubernetes.

    Returns 200 if the wrapper process is alive.
    """
    return {"alive": True}
```

### Step 3: Update Main to Start Health Monitor

Update `sworker-wrapper/src/main.py`:

```python
async def run_wrapper() -> None:
    """Run the wrapper with full lifecycle management."""
    # ... (previous code) ...

    # Start health monitor
    from src.health import create_health_monitor
    health_monitor = create_health_monitor()
    await health_monitor.start()

    # ... (rest of startup) ...

    # In shutdown, stop health monitor
    signal_handler = create_signal_handler(
        grace_period=config.grace_period_seconds,
        on_deregister=deregister,
    )

    # Add health monitor cleanup
    original_shutdown = signal_handler._graceful_shutdown

    async def shutdown_with_health():
        await health_monitor.stop()
        await original_shutdown()

    signal_handler._graceful_shutdown = shutdown_with_health

    # ... (rest of code) ...
```

### Step 4: Create Tests

Create `sworker-wrapper/tests/test_health.py`:

```python
"""Tests for health monitoring."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.health import HealthMonitor, HealthStatus


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a health monitor."""
        return HealthMonitor(check_interval=1.0, report_interval=5.0)

    @pytest.mark.asyncio
    async def test_check_health_healthy(self, monitor):
        """Test health check when all systems healthy."""
        with patch("src.health.get_lifecycle") as mock_lifecycle, \
             patch("src.health.get_task_queue") as mock_queue, \
             patch("src.health.get_model_client") as mock_client:

            # Mock healthy state
            mock_lifecycle.return_value.state = MagicMock(value="running")
            mock_queue.return_value.get_queue_stats = AsyncMock(
                return_value=MagicMock(
                    queued=5,
                    running=1,
                    completed=10,
                    failed=0,
                )
            )
            mock_client.return_value.health_check = AsyncMock(return_value=True)

            report = await monitor.check_health()

            assert report.status == HealthStatus.HEALTHY
            assert report.model_healthy is True
            assert report.queue_size == 5
            assert report.running_tasks == 1

    @pytest.mark.asyncio
    async def test_check_health_degraded(self, monitor):
        """Test health check when model unhealthy."""
        with patch("src.health.get_lifecycle") as mock_lifecycle, \
             patch("src.health.get_task_queue") as mock_queue, \
             patch("src.health.get_model_client") as mock_client:

            mock_lifecycle.return_value.state = MagicMock(value="running")
            mock_queue.return_value.get_queue_stats = AsyncMock(
                return_value=MagicMock(queued=0, running=0, completed=0, failed=0)
            )
            # Model unhealthy for 3+ consecutive checks
            mock_client.return_value.health_check = AsyncMock(return_value=False)

            # Trigger 3 failures
            for _ in range(3):
                report = await monitor.check_health()

            assert report.status == HealthStatus.DEGRADED
            assert report.model_healthy is False

    @pytest.mark.asyncio
    async def test_uptime_calculation(self, monitor):
        """Test uptime calculation."""
        import asyncio
        await asyncio.sleep(0.1)

        assert monitor.uptime_seconds >= 0.1
```

## Test Strategy

### Unit Tests

```bash
cd sworker-wrapper
uv run pytest tests/test_health.py -v
```

### Integration Testing

```bash
# Start wrapper
PORT=16000 SWORKER_COMMAND="python -m http.server 16001" uv run sworker-wrapper &

# Check endpoints
curl http://localhost:16000/health
curl http://localhost:16000/ready
curl http://localhost:16000/live
```

## Acceptance Criteria

- [ ] Health check detects model service status
- [ ] Consecutive failures trigger degraded state
- [ ] Queue stats reported correctly
- [ ] Periodic scheduler reporting works
- [ ] Readiness endpoint respects lifecycle state
- [ ] Liveness endpoint always responds
- [ ] All tests pass

## Next Steps

Proceed to [PYLET-006](PYLET-006-phase1-integration-tests.md) for Phase 1 integration tests.

## Code References

- Current health check: [instance/src/api.py](../../instance/src/api.py)
- Model health: [instance/src/docker_manager.py](../../instance/src/docker_manager.py)
