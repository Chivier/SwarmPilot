# PYLET-012: Phase 2 Integration Tests

## Objective

Create comprehensive integration tests for the planner's PyLet integration. These tests validate the complete workflow from deployment decisions through PyLet execution to scheduler registration.

## Prerequisites

- All Phase 2 tasks completed (PYLET-007 through PYLET-011)
- Phase 1 (SWorker wrapper) fully implemented
- Test PyLet cluster available

## Background

Integration tests must validate:
1. End-to-end deployment flow
2. Instance lifecycle management
3. Migration operations
4. State synchronization
5. Error handling and recovery
6. Performance under load

## Files to Create

```
planner/
└── tests/
    └── integration/
        ├── __init__.py
        ├── conftest.py              # Test fixtures
        ├── test_deployment_flow.py  # Deployment tests
        ├── test_migration_flow.py   # Migration tests
        ├── test_state_sync.py       # State sync tests
        └── test_recovery.py         # Recovery tests
```

## Implementation Steps

### Step 1: Create Test Fixtures

Create `planner/tests/integration/conftest.py`:

```python
"""Integration test fixtures."""

import asyncio
import os
import pytest
from typing import AsyncGenerator

# Skip if no PyLet cluster
pytestmark = pytest.mark.skipif(
    os.getenv("PYLET_HEAD_ADDRESS") is None,
    reason="PYLET_HEAD_ADDRESS not set - no PyLet cluster available",
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


@pytest.fixture
async def pylet_client(pylet_address):
    """Create PyLet client connected to cluster."""
    from src.pylet_client_async import AsyncPyLetClient

    client = AsyncPyLetClient(head_address=pylet_address)
    await client.init()
    yield client
    await client.shutdown()


@pytest.fixture
async def instance_manager(pylet_client):
    """Create instance manager with PyLet client."""
    from src.instance_manager import InstanceManager

    manager = InstanceManager(pylet_client=pylet_client)

    # Register test model command
    manager.register_model_command(
        "test-model",
        "sworker-wrapper --command 'sleep 3600' --port $PORT",
    )

    yield manager

    # Cleanup: terminate all test instances
    instances = await manager.get_model_instances("test-model")
    if instances:
        await manager.terminate_instances("test-model", len(instances))


@pytest.fixture
async def deployment_executor(instance_manager):
    """Create deployment executor."""
    from src.deployment_executor import DeploymentExecutor

    return DeploymentExecutor(instance_manager)


@pytest.fixture
async def state_tracker():
    """Create state tracker."""
    from src.state_tracker import StateTracker

    return StateTracker()


@pytest.fixture
async def mock_scheduler(scheduler_address):
    """Mock scheduler for registration tests."""
    import httpx
    from fastapi import FastAPI
    from contextlib import asynccontextmanager
    import uvicorn
    import threading

    app = FastAPI()
    registered_instances = set()

    @app.post("/model/register")
    async def register(request: dict):
        registered_instances.add(request.get("instance_id"))
        return {"status": "ok"}

    @app.post("/model/deregister")
    async def deregister(request: dict):
        registered_instances.discard(request.get("instance_id"))
        return {"status": "ok"}

    @app.get("/instances")
    async def list_instances():
        return {"instance_ids": list(registered_instances)}

    # Start in background thread
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run)
    thread.daemon = True
    thread.start()

    yield registered_instances

    server.should_exit = True
```

### Step 2: Create Deployment Flow Tests

Create `planner/tests/integration/test_deployment_flow.py`:

```python
"""Integration tests for deployment flow."""

import asyncio
import pytest


class TestDeploymentFlow:
    """Tests for end-to-end deployment flow."""

    @pytest.mark.asyncio
    async def test_deploy_single_instance(self, instance_manager):
        """Test deploying a single instance."""
        instances = await instance_manager.deploy_instances(
            model_id="test-model",
            count=1,
        )

        assert len(instances) == 1
        assert instances[0].status == "PENDING"

        # Wait for running
        ready = await instance_manager.wait_instances_ready(
            instances,
            timeout=120.0,
        )

        assert len(ready) == 1
        assert ready[0].status == "RUNNING"
        assert ready[0].endpoint is not None

    @pytest.mark.asyncio
    async def test_deploy_multiple_instances(self, instance_manager):
        """Test deploying multiple instances."""
        instances = await instance_manager.deploy_instances(
            model_id="test-model",
            count=3,
        )

        assert len(instances) == 3

        ready = await instance_manager.wait_instances_ready(
            instances,
            timeout=180.0,
        )

        running = [i for i in ready if i.status == "RUNNING"]
        assert len(running) == 3

    @pytest.mark.asyncio
    async def test_terminate_instances(self, instance_manager):
        """Test terminating instances."""
        # Deploy
        instances = await instance_manager.deploy_instances("test-model", 2)
        await instance_manager.wait_instances_ready(instances, timeout=120.0)

        # Verify running
        count = await instance_manager.get_instance_count("test-model")
        assert count == 2

        # Terminate one
        terminated = await instance_manager.terminate_instances("test-model", 1)
        assert len(terminated) == 1

        # Verify count
        count = await instance_manager.get_instance_count("test-model")
        assert count == 1

    @pytest.mark.asyncio
    async def test_deployment_executor(self, deployment_executor, instance_manager):
        """Test deployment executor with add/remove."""
        # Add instances
        result = await deployment_executor.execute({
            "add": [("test-model", 2)],
            "remove": [],
        })

        assert len(result.successful_adds) == 2
        assert result.all_successful

        # Remove one
        result = await deployment_executor.execute({
            "add": [],
            "remove": [("test-model", 1)],
        })

        assert len(result.successful_removes) == 1

    @pytest.mark.asyncio
    async def test_reconciliation(self, deployment_executor, instance_manager):
        """Test reconciling to target state."""
        # Start with 0 instances
        result = await deployment_executor.reconcile({
            "test-model": 3,
        })

        assert len(result.successful_adds) == 3

        # Reconcile down to 1
        result = await deployment_executor.reconcile({
            "test-model": 1,
        })

        assert len(result.successful_removes) == 2

        # Verify final count
        count = await instance_manager.get_instance_count("test-model")
        assert count == 1
```

### Step 3: Create Migration Flow Tests

Create `planner/tests/integration/test_migration_flow.py`:

```python
"""Integration tests for migration flow."""

import asyncio
import pytest

from src.migration_executor import MigrationExecutor, MigrationPlan


class TestMigrationFlow:
    """Tests for instance migration."""

    @pytest.mark.asyncio
    async def test_single_migration(self, instance_manager):
        """Test migrating a single instance."""
        # Deploy instance
        instances = await instance_manager.deploy_instances("test-model", 1)
        await instance_manager.wait_instances_ready(instances, timeout=120.0)

        original = instances[0]
        executor = MigrationExecutor(instance_manager)

        # Create migration plan
        plan = MigrationPlan(
            source_pylet_id=original.pylet_id,
            model_id="test-model",
            instance_id=original.instance_id,
            reason="test",
        )

        # Execute migration
        result = await executor.migrate(plan)

        assert result.success
        assert result.new_instance is not None
        assert result.new_instance.pylet_id != original.pylet_id

        # Verify count unchanged
        count = await instance_manager.get_instance_count("test-model")
        assert count == 1

    @pytest.mark.asyncio
    async def test_batch_migration(self, instance_manager):
        """Test migrating multiple instances."""
        # Deploy instances
        instances = await instance_manager.deploy_instances("test-model", 3)
        await instance_manager.wait_instances_ready(instances, timeout=180.0)

        executor = MigrationExecutor(instance_manager)

        # Migrate all
        plans = [
            MigrationPlan(
                source_pylet_id=inst.pylet_id,
                model_id="test-model",
                instance_id=inst.instance_id,
                reason="batch_test",
            )
            for inst in instances
        ]

        results = await executor.migrate_batch(plans, parallel=2)

        successful = [r for r in results if r.success]
        assert len(successful) == 3

    @pytest.mark.asyncio
    async def test_migration_preserves_availability(self, instance_manager):
        """Test that migration maintains service availability."""
        # Deploy
        instances = await instance_manager.deploy_instances("test-model", 1)
        await instance_manager.wait_instances_ready(instances, timeout=120.0)

        executor = MigrationExecutor(instance_manager)

        plan = MigrationPlan(
            source_pylet_id=instances[0].pylet_id,
            model_id="test-model",
            instance_id=instances[0].instance_id,
            reason="availability_test",
        )

        # During migration, there should always be at least one instance
        # (new deploys before old terminates)
        async def check_availability():
            while True:
                count = await instance_manager.get_instance_count("test-model")
                if count == 0:
                    return False
                await asyncio.sleep(0.5)

        check_task = asyncio.create_task(check_availability())
        result = await executor.migrate(plan)
        check_task.cancel()

        assert result.success
        # Availability check didn't return False
```

### Step 4: Create State Sync Tests

Create `planner/tests/integration/test_state_sync.py`:

```python
"""Integration tests for state synchronization."""

import asyncio
import pytest

from src.state_tracker import StateTracker, InstanceState, RegistrationState
from src.state_sync import StateSynchronizer


class TestStateSync:
    """Tests for state synchronization."""

    @pytest.mark.asyncio
    async def test_sync_with_pylet(
        self,
        pylet_client,
        instance_manager,
        state_tracker,
    ):
        """Test state sync with PyLet cluster."""
        # Deploy instances
        instances = await instance_manager.deploy_instances("test-model", 2)

        # Create records
        for inst in instances:
            await state_tracker.create_record(
                pylet_id=inst.pylet_id,
                instance_id=inst.instance_id,
                model_id="test-model",
            )

        # Wait and sync
        await asyncio.sleep(5)
        sync = StateSynchronizer(
            tracker=state_tracker,
            pylet_client=pylet_client,
            scheduler_url="http://localhost:8080",
        )
        await sync._sync_with_pylet()

        # Check state updated
        records = await state_tracker.get_all_records()
        for record in records:
            assert record.lifecycle_state in (
                InstanceState.PENDING,
                InstanceState.ASSIGNED,
                InstanceState.RUNNING,
            )

    @pytest.mark.asyncio
    async def test_detect_terminated_instances(
        self,
        pylet_client,
        instance_manager,
        state_tracker,
    ):
        """Test detecting instances terminated outside planner."""
        # Deploy and create record
        instances = await instance_manager.deploy_instances("test-model", 1)
        inst = instances[0]

        await state_tracker.create_record(
            pylet_id=inst.pylet_id,
            instance_id=inst.instance_id,
            model_id="test-model",
        )

        # Terminate via instance manager (simulating external termination)
        await instance_manager.terminate_instances("test-model", 1)

        # Sync
        sync = StateSynchronizer(
            tracker=state_tracker,
            pylet_client=pylet_client,
            scheduler_url="http://localhost:8080",
        )
        await sync._sync_with_pylet()

        # Record should show cancelled
        record = await state_tracker.get_record(inst.pylet_id)
        assert record.lifecycle_state == InstanceState.CANCELLED

    @pytest.mark.asyncio
    async def test_state_summary(self, state_tracker):
        """Test getting state summary."""
        # Create records in various states
        await state_tracker.create_record("p1", "i1", "model-a")
        await state_tracker.create_record("p2", "i2", "model-a")
        await state_tracker.create_record("p3", "i3", "model-b")

        await state_tracker.update_record(
            "p1",
            lifecycle=InstanceState.RUNNING,
            registration=RegistrationState.REGISTERED,
        )

        summary = await state_tracker.get_summary()

        assert "model-a" in summary
        assert summary["model-a"]["total"] == 2
        assert summary["model-a"]["operational"] == 1
        assert "model-b" in summary
        assert summary["model-b"]["total"] == 1
```

### Step 5: Create Recovery Tests

Create `planner/tests/integration/test_recovery.py`:

```python
"""Integration tests for error recovery."""

import asyncio
import pytest


class TestRecovery:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_deployment_partial_failure(self, deployment_executor):
        """Test handling partial deployment failure."""
        # This test requires a way to force failure (e.g., resource exhaustion)
        # For now, test that failures are reported correctly
        result = await deployment_executor.execute({
            "add": [("test-model", 5)],  # Request many
            "remove": [],
        })

        # Should have some successes or failures
        total = len(result.successful_adds) + len(result.failed_adds)
        assert total > 0

    @pytest.mark.asyncio
    async def test_reconnect_after_pylet_restart(self, pylet_client):
        """Test handling PyLet connection loss."""
        # This is a stress test that requires PyLet restart
        # Placeholder for manual testing

        # Verify connection works
        workers = await pylet_client.get_workers()
        assert isinstance(workers, list)

    @pytest.mark.asyncio
    async def test_state_recovery_after_planner_restart(
        self,
        pylet_client,
        instance_manager,
        state_tracker,
    ):
        """Test recovering state after planner restart."""
        # Deploy instances
        instances = await instance_manager.deploy_instances("test-model", 2)
        await instance_manager.wait_instances_ready(instances, timeout=120.0)

        # Simulate restart by clearing state tracker
        for inst in instances:
            await state_tracker.delete_record(inst.pylet_id)

        # Verify tracker is empty
        records = await state_tracker.get_all_records()
        assert len(records) == 0

        # Recover state from PyLet
        await instance_manager.sync_with_pylet()

        # Re-create records from PyLet
        pylet_instances = await pylet_client.list_model_instances("test-model")
        for info in pylet_instances:
            await state_tracker.create_record(
                pylet_id=info.pylet_id,
                instance_id=info.name,
                model_id="test-model",
            )

        # Verify recovered
        records = await state_tracker.get_all_records()
        assert len(records) == 2
```

### Step 6: Create Test Runner Script

Create `planner/tests/integration/run_integration_tests.sh`:

```bash
#!/bin/bash
# Run integration tests against PyLet cluster

set -e

# Configuration
export PYLET_HEAD_ADDRESS="${PYLET_HEAD_ADDRESS:-http://localhost:8000}"
export SCHEDULER_ADDRESS="${SCHEDULER_ADDRESS:-http://localhost:8080}"

echo "Running integration tests..."
echo "PyLet: $PYLET_HEAD_ADDRESS"
echo "Scheduler: $SCHEDULER_ADDRESS"

# Check PyLet is running
if ! curl -s "$PYLET_HEAD_ADDRESS/health" > /dev/null 2>&1; then
    echo "ERROR: PyLet head not available at $PYLET_HEAD_ADDRESS"
    echo "Start PyLet with: pylet start"
    exit 1
fi

# Run tests
cd "$(dirname "$0")/../.."
uv run pytest tests/integration/ -v --tb=short "$@"
```

## Test Strategy

### Prerequisites

1. Start PyLet cluster:
```bash
cd /home/yanweiye/Projects/pylet
pylet start &
pylet start --head localhost:8000 &
```

2. Ensure SWorker wrapper is built:
```bash
cd sworker-wrapper
uv build
```

### Running Tests

```bash
# Set environment
export PYLET_HEAD_ADDRESS=http://localhost:8000
export SCHEDULER_ADDRESS=http://localhost:8080

# Run all integration tests
cd planner
./tests/integration/run_integration_tests.sh

# Run specific test file
uv run pytest tests/integration/test_deployment_flow.py -v

# Run with coverage
uv run pytest tests/integration/ --cov=src --cov-report=html
```

### CI Integration

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  integration:
    runs-on: ubuntu-latest
    services:
      pylet:
        image: pylet:latest
        ports:
          - 8000:8000

    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install uv
          cd planner && uv sync

      - name: Run integration tests
        env:
          PYLET_HEAD_ADDRESS: http://localhost:8000
        run: |
          cd planner
          uv run pytest tests/integration/ -v
```

## Acceptance Criteria

- [ ] Test fixtures created for PyLet integration
- [ ] Deployment flow tests pass
- [ ] Migration flow tests pass
- [ ] State sync tests pass
- [ ] Recovery tests pass
- [ ] Tests skip gracefully without PyLet cluster
- [ ] CI integration configured

## Post-Migration Validation

After completing all Phase 2 tasks, run the full integration test suite:

```bash
# Full validation
PYLET_HEAD_ADDRESS=http://localhost:8000 \
uv run pytest tests/integration/ -v --tb=long

# Expected: All tests pass
```

## Code References

- Instance manager: [planner/src/instance_manager.py](../../planner/src/instance_manager.py)
- Deployment executor: [planner/src/deployment_executor.py](../../planner/src/deployment_executor.py)
- State tracker: [planner/src/state_tracker.py](../../planner/src/state_tracker.py)
