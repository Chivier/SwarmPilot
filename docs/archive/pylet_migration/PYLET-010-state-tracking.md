# PYLET-011: State Tracking

## Objective

Implement comprehensive state tracking for PyLet-managed instances in the planner. This includes maintaining instance state, syncing with PyLet, handling state transitions, and providing observability.

## Prerequisites

- PYLET-008 completed (Instance lifecycle)
- PYLET-009 completed (Deployment strategy)
- PYLET-010 completed (Migration optimizer)

## Background

The planner needs to track:
1. **Instance State**: PENDING, RUNNING, FAILED, CANCELLED
2. **Registration State**: Whether instance registered with scheduler
3. **Health State**: Healthy, Unhealthy, Unknown
4. **Resource Allocation**: GPU count, worker assignment

State must be synchronized between:
- Planner's in-memory state
- PyLet cluster state
- Scheduler registration state

## Files to Create/Modify

```
planner/
└── src/
    ├── state_tracker.py         # NEW: State tracking system
    ├── state_sync.py            # NEW: State synchronization
    └── metrics.py               # NEW: State metrics export
```

## Implementation Steps

### Step 1: Create State Tracker

Create `planner/src/state_tracker.py`:

```python
"""Instance state tracking for planner.

Maintains synchronized view of instance states across
PyLet, scheduler, and planner.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from loguru import logger


class InstanceState(str, Enum):
    """Instance lifecycle states."""

    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class RegistrationState(str, Enum):
    """Scheduler registration states."""

    UNREGISTERED = "UNREGISTERED"
    REGISTERING = "REGISTERING"
    REGISTERED = "REGISTERED"
    DEREGISTERING = "DEREGISTERING"
    FAILED = "FAILED"


class HealthState(str, Enum):
    """Instance health states."""

    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    DEGRADED = "DEGRADED"


@dataclass
class InstanceStateRecord:
    """Complete state record for an instance."""

    # Identifiers
    pylet_id: str
    instance_id: str
    model_id: str

    # States
    lifecycle_state: InstanceState = InstanceState.PENDING
    registration_state: RegistrationState = RegistrationState.UNREGISTERED
    health_state: HealthState = HealthState.UNKNOWN

    # Metadata
    endpoint: Optional[str] = None
    worker_id: Optional[str] = None
    gpu_count: int = 1

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None

    # Metrics
    request_count: int = 0
    error_count: int = 0

    def update_state(
        self,
        lifecycle: Optional[InstanceState] = None,
        registration: Optional[RegistrationState] = None,
        health: Optional[HealthState] = None,
    ) -> None:
        """Update state fields."""
        if lifecycle:
            self.lifecycle_state = lifecycle
        if registration:
            self.registration_state = registration
        if health:
            self.health_state = health
            self.last_health_check = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    @property
    def is_operational(self) -> bool:
        """Check if instance is fully operational."""
        return (
            self.lifecycle_state == InstanceState.RUNNING
            and self.registration_state == RegistrationState.REGISTERED
            and self.health_state in (HealthState.HEALTHY, HealthState.UNKNOWN)
        )


class StateTracker:
    """Tracks state of all managed instances."""

    def __init__(self):
        """Initialize state tracker."""
        self._records: dict[str, InstanceStateRecord] = {}
        self._lock = asyncio.Lock()
        self._listeners: list[Callable] = []

    async def create_record(
        self,
        pylet_id: str,
        instance_id: str,
        model_id: str,
        gpu_count: int = 1,
    ) -> InstanceStateRecord:
        """Create a new instance record.

        Args:
            pylet_id: PyLet instance ID.
            instance_id: Instance name/ID.
            model_id: Model identifier.
            gpu_count: GPU allocation.

        Returns:
            New state record.
        """
        async with self._lock:
            record = InstanceStateRecord(
                pylet_id=pylet_id,
                instance_id=instance_id,
                model_id=model_id,
                gpu_count=gpu_count,
            )
            self._records[pylet_id] = record
            logger.debug(f"Created state record for {instance_id}")
            return record

    async def update_record(
        self,
        pylet_id: str,
        lifecycle: Optional[InstanceState] = None,
        registration: Optional[RegistrationState] = None,
        health: Optional[HealthState] = None,
        endpoint: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> Optional[InstanceStateRecord]:
        """Update an instance record.

        Args:
            pylet_id: PyLet instance ID.
            lifecycle: New lifecycle state.
            registration: New registration state.
            health: New health state.
            endpoint: Instance endpoint.
            worker_id: Worker assignment.

        Returns:
            Updated record or None if not found.
        """
        async with self._lock:
            record = self._records.get(pylet_id)
            if not record:
                logger.warning(f"No record found for {pylet_id}")
                return None

            old_state = (
                record.lifecycle_state,
                record.registration_state,
                record.health_state,
            )

            record.update_state(lifecycle, registration, health)

            if endpoint:
                record.endpoint = endpoint
            if worker_id:
                record.worker_id = worker_id

            new_state = (
                record.lifecycle_state,
                record.registration_state,
                record.health_state,
            )

            if old_state != new_state:
                await self._notify_listeners(record, old_state, new_state)

            return record

    async def delete_record(self, pylet_id: str) -> Optional[InstanceStateRecord]:
        """Delete an instance record.

        Args:
            pylet_id: PyLet instance ID.

        Returns:
            Deleted record or None.
        """
        async with self._lock:
            return self._records.pop(pylet_id, None)

    async def get_record(self, pylet_id: str) -> Optional[InstanceStateRecord]:
        """Get an instance record."""
        async with self._lock:
            return self._records.get(pylet_id)

    async def get_all_records(self) -> list[InstanceStateRecord]:
        """Get all instance records."""
        async with self._lock:
            return list(self._records.values())

    async def get_model_records(self, model_id: str) -> list[InstanceStateRecord]:
        """Get all records for a model."""
        async with self._lock:
            return [r for r in self._records.values() if r.model_id == model_id]

    async def get_operational_count(self, model_id: str) -> int:
        """Get count of operational instances for a model."""
        records = await self.get_model_records(model_id)
        return sum(1 for r in records if r.is_operational)

    def add_listener(self, callback: Callable) -> None:
        """Add state change listener."""
        self._listeners.append(callback)

    async def _notify_listeners(
        self,
        record: InstanceStateRecord,
        old_state: tuple,
        new_state: tuple,
    ) -> None:
        """Notify listeners of state change."""
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(record, old_state, new_state)
                else:
                    listener(record, old_state, new_state)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    async def get_summary(self) -> dict:
        """Get state summary by model.

        Returns:
            Dict with model stats.
        """
        async with self._lock:
            summary = {}
            for record in self._records.values():
                if record.model_id not in summary:
                    summary[record.model_id] = {
                        "total": 0,
                        "operational": 0,
                        "pending": 0,
                        "failed": 0,
                    }

                stats = summary[record.model_id]
                stats["total"] += 1

                if record.is_operational:
                    stats["operational"] += 1
                elif record.lifecycle_state == InstanceState.PENDING:
                    stats["pending"] += 1
                elif record.lifecycle_state == InstanceState.FAILED:
                    stats["failed"] += 1

            return summary
```

### Step 2: Create State Synchronizer

Create `planner/src/state_sync.py`:

```python
"""State synchronization with PyLet and scheduler.

Periodically syncs planner state with external systems.
"""

import asyncio
from typing import Optional

from loguru import logger

from src.pylet_client_async import AsyncPyLetClient
from src.state_tracker import (
    StateTracker,
    InstanceState,
    RegistrationState,
    HealthState,
)


class StateSynchronizer:
    """Synchronizes state between planner, PyLet, and scheduler."""

    def __init__(
        self,
        tracker: StateTracker,
        pylet_client: AsyncPyLetClient,
        scheduler_url: str,
        sync_interval: float = 10.0,
    ):
        """Initialize synchronizer.

        Args:
            tracker: State tracker to update.
            pylet_client: PyLet client for cluster queries.
            scheduler_url: Scheduler URL for registration checks.
            sync_interval: Seconds between sync cycles.
        """
        self._tracker = tracker
        self._pylet = pylet_client
        self._scheduler_url = scheduler_url
        self._sync_interval = sync_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background sync task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info("State synchronizer started")

    async def stop(self) -> None:
        """Stop background sync task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("State synchronizer stopped")

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            try:
                await self.sync()
            except Exception as e:
                logger.error(f"Sync error: {e}")

            await asyncio.sleep(self._sync_interval)

    async def sync(self) -> None:
        """Perform full state synchronization."""
        await self._sync_with_pylet()
        await self._sync_with_scheduler()
        await self._cleanup_stale_records()

    async def _sync_with_pylet(self) -> None:
        """Sync state with PyLet cluster."""
        # Get all planner-managed instances from PyLet
        pylet_instances = await self._pylet.list_model_instances_by_label(
            "managed_by", "swarmpilot-planner"
        )

        pylet_state = {inst.pylet_id: inst for inst in pylet_instances}

        # Update existing records
        records = await self._tracker.get_all_records()
        for record in records:
            if record.pylet_id in pylet_state:
                pylet_inst = pylet_state[record.pylet_id]

                # Map PyLet status to our state
                lifecycle = self._map_pylet_status(pylet_inst.status)

                await self._tracker.update_record(
                    pylet_id=record.pylet_id,
                    lifecycle=lifecycle,
                    endpoint=pylet_inst.endpoint,
                )
            else:
                # Instance no longer in PyLet - mark as cancelled
                await self._tracker.update_record(
                    pylet_id=record.pylet_id,
                    lifecycle=InstanceState.CANCELLED,
                )

    async def _sync_with_scheduler(self) -> None:
        """Sync registration state with scheduler."""
        import httpx

        records = await self._tracker.get_all_records()

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(
                    f"{self._scheduler_url}/instances"
                )
                if response.status_code == 200:
                    registered = set(response.json().get("instance_ids", []))

                    for record in records:
                        is_registered = record.instance_id in registered

                        if is_registered:
                            new_state = RegistrationState.REGISTERED
                        elif record.registration_state == RegistrationState.REGISTERED:
                            new_state = RegistrationState.UNREGISTERED
                        else:
                            continue

                        await self._tracker.update_record(
                            pylet_id=record.pylet_id,
                            registration=new_state,
                        )

            except Exception as e:
                logger.warning(f"Failed to sync with scheduler: {e}")

    async def _cleanup_stale_records(self) -> None:
        """Remove records for terminated instances."""
        records = await self._tracker.get_all_records()

        for record in records:
            if record.lifecycle_state in (
                InstanceState.CANCELLED,
                InstanceState.FAILED,
            ):
                # Keep failed records for debugging, remove cancelled
                if record.lifecycle_state == InstanceState.CANCELLED:
                    await self._tracker.delete_record(record.pylet_id)
                    logger.debug(f"Cleaned up record for {record.instance_id}")

    def _map_pylet_status(self, status: str) -> InstanceState:
        """Map PyLet status to instance state."""
        mapping = {
            "PENDING": InstanceState.PENDING,
            "ASSIGNED": InstanceState.ASSIGNED,
            "RUNNING": InstanceState.RUNNING,
            "COMPLETED": InstanceState.CANCELLED,
            "FAILED": InstanceState.FAILED,
            "CANCELLED": InstanceState.CANCELLED,
            "UNKNOWN": InstanceState.UNKNOWN,
        }
        return mapping.get(status.upper(), InstanceState.UNKNOWN)
```

### Step 3: Create Metrics Export

Create `planner/src/state_metrics.py`:

```python
"""State metrics export for monitoring.

Provides Prometheus-compatible metrics for instance state.
"""

from typing import Optional

from src.state_tracker import (
    StateTracker,
    InstanceState,
    RegistrationState,
    HealthState,
)


class StateMetrics:
    """Exports instance state metrics."""

    def __init__(self, tracker: StateTracker):
        """Initialize metrics exporter.

        Args:
            tracker: State tracker to export from.
        """
        self._tracker = tracker

    async def get_metrics(self) -> dict:
        """Get current metrics.

        Returns:
            Dict of metric name to value.
        """
        records = await self._tracker.get_all_records()
        summary = await self._tracker.get_summary()

        metrics = {
            # Instance counts
            "swarmpilot_instances_total": len(records),
            "swarmpilot_instances_operational": sum(
                1 for r in records if r.is_operational
            ),

            # Per-model counts
            "swarmpilot_model_instances": {
                model_id: stats["total"]
                for model_id, stats in summary.items()
            },
            "swarmpilot_model_operational": {
                model_id: stats["operational"]
                for model_id, stats in summary.items()
            },

            # State distribution
            "swarmpilot_lifecycle_state": self._count_by_state(
                records, "lifecycle_state"
            ),
            "swarmpilot_registration_state": self._count_by_state(
                records, "registration_state"
            ),
            "swarmpilot_health_state": self._count_by_state(
                records, "health_state"
            ),
        }

        return metrics

    def _count_by_state(self, records: list, field: str) -> dict:
        """Count records by state field."""
        counts = {}
        for record in records:
            state = getattr(record, field).value
            counts[state] = counts.get(state, 0) + 1
        return counts

    async def format_prometheus(self) -> str:
        """Format metrics as Prometheus text.

        Returns:
            Prometheus-formatted metrics string.
        """
        metrics = await self.get_metrics()
        lines = []

        # Total instances
        lines.append(
            f"swarmpilot_instances_total {metrics['swarmpilot_instances_total']}"
        )
        lines.append(
            f"swarmpilot_instances_operational {metrics['swarmpilot_instances_operational']}"
        )

        # Per-model
        for model_id, count in metrics["swarmpilot_model_instances"].items():
            lines.append(
                f'swarmpilot_model_instances{{model_id="{model_id}"}} {count}'
            )

        # Lifecycle states
        for state, count in metrics["swarmpilot_lifecycle_state"].items():
            lines.append(
                f'swarmpilot_lifecycle_state{{state="{state}"}} {count}'
            )

        return "\n".join(lines)
```

### Step 4: Create Tests

Create `planner/tests/test_state_tracker.py`:

```python
"""Tests for state tracking."""

import pytest
from datetime import datetime

from src.state_tracker import (
    StateTracker,
    InstanceState,
    RegistrationState,
    HealthState,
)


class TestStateTracker:
    """Tests for StateTracker."""

    @pytest.fixture
    def tracker(self):
        """Create state tracker."""
        return StateTracker()

    @pytest.mark.asyncio
    async def test_create_record(self, tracker):
        """Test creating instance record."""
        record = await tracker.create_record(
            pylet_id="p1",
            instance_id="inst-1",
            model_id="model-a",
        )

        assert record.pylet_id == "p1"
        assert record.lifecycle_state == InstanceState.PENDING
        assert record.registration_state == RegistrationState.UNREGISTERED

    @pytest.mark.asyncio
    async def test_update_record(self, tracker):
        """Test updating instance record."""
        await tracker.create_record("p1", "inst-1", "model-a")

        updated = await tracker.update_record(
            pylet_id="p1",
            lifecycle=InstanceState.RUNNING,
            registration=RegistrationState.REGISTERED,
        )

        assert updated.lifecycle_state == InstanceState.RUNNING
        assert updated.registration_state == RegistrationState.REGISTERED

    @pytest.mark.asyncio
    async def test_is_operational(self, tracker):
        """Test operational check."""
        await tracker.create_record("p1", "inst-1", "model-a")

        # Not operational initially
        record = await tracker.get_record("p1")
        assert not record.is_operational

        # Update to operational state
        await tracker.update_record(
            pylet_id="p1",
            lifecycle=InstanceState.RUNNING,
            registration=RegistrationState.REGISTERED,
            health=HealthState.HEALTHY,
        )

        record = await tracker.get_record("p1")
        assert record.is_operational

    @pytest.mark.asyncio
    async def test_get_summary(self, tracker):
        """Test state summary."""
        await tracker.create_record("p1", "inst-1", "model-a")
        await tracker.create_record("p2", "inst-2", "model-a")
        await tracker.create_record("p3", "inst-3", "model-b")

        # Make one operational
        await tracker.update_record(
            "p1",
            lifecycle=InstanceState.RUNNING,
            registration=RegistrationState.REGISTERED,
        )

        summary = await tracker.get_summary()

        assert "model-a" in summary
        assert summary["model-a"]["total"] == 2
        assert summary["model-a"]["operational"] == 1
```

## Test Strategy

### Unit Tests

```bash
cd planner
uv run pytest tests/test_state_tracker.py -v
```

### Integration with Sync

```python
async def test_state_sync():
    """Test state synchronization."""
    tracker = StateTracker()
    pylet = AsyncPyLetClient()
    await pylet.init()

    sync = StateSynchronizer(
        tracker=tracker,
        pylet_client=pylet,
        scheduler_url="http://localhost:8000",
    )

    await sync.start()
    await asyncio.sleep(15)  # Let it sync

    summary = await tracker.get_summary()
    print(f"State: {summary}")

    await sync.stop()
```

## Acceptance Criteria

- [ ] StateTracker tracks all instance states
- [ ] State transitions logged and observable
- [ ] StateSynchronizer syncs with PyLet
- [ ] StateSynchronizer syncs with scheduler
- [ ] Metrics export working
- [ ] Stale records cleaned up
- [ ] All tests pass

## Next Steps

Proceed to [PYLET-012](PYLET-012-phase2-integration-tests.md) for Phase 2 integration testing.

## Code References

- Instance manager: [planner/src/instance_manager.py](../../planner/src/instance_manager.py)
- PyLet client: [planner/src/pylet_client.py](../../planner/src/pylet_client.py)
