# PYLET-013: Planner Module README & Quick Start

## Objective

Create comprehensive README documentation for the `planner/` module that enables users to run the planner service with minimal configuration. This provides a standalone entry point for Phase 2 functionality.

## Prerequisites

- PYLET-006 through PYLET-012 completed
- PyLet client integration working
- Scheduler client integration working

## Background

After Phase 2 implementation, users should be able to:
1. Run the planner service standalone
2. Configure PyLet and scheduler connections
3. Test deployment operations with minimal setup
4. Understand the planner's role in the system

## Files to Create/Modify

```
planner/
├── README.md                    # NEW: Module documentation
├── pyproject.toml               # EXISTS: Dependencies
├── src/
│   ├── __init__.py
│   ├── pylet_client.py          # EXISTS: PyLet integration
│   ├── scheduler_client.py      # EXISTS: Scheduler client
│   └── instance_manager.py      # EXISTS: Instance lifecycle
└── examples/
    ├── quickstart.py            # NEW: Minimal example
    └── deploy_model.py          # NEW: Deploy model example
```

## Implementation Steps

### Step 1: Create Planner README

Create `planner/README.md`:

```markdown
# SwarmPilot Planner

The Planner service manages model instance lifecycle via PyLet and coordinates with the scheduler for request routing.

## Quick Start

### Prerequisites

- Python 3.10+
- PyLet cluster running (head + worker nodes)
- Scheduler service running
- Model weights available

### Installation

```bash
cd planner

# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Minimal Configuration

Create a `.env` file or set environment variables:

```bash
# Required
PYLET_HEAD_ADDRESS=http://localhost:8000
SCHEDULER_URL=http://localhost:8080

# Optional
LOG_LEVEL=INFO
```

### Run Planner

```bash
# Start the planner service
uv run python -m src.main

# Or run directly
python -m src.main
```

### Deploy a Model (Programmatic)

```python
import asyncio
from src.pylet_client import AsyncPyLetClient
from src.scheduler_client import SchedulerClient
from src.instance_manager import InstanceManager

async def main():
    # Initialize clients
    pylet = AsyncPyLetClient(head_address="http://localhost:8000")
    await pylet.init()

    scheduler = SchedulerClient(scheduler_url="http://localhost:8080")

    # Create instance manager
    manager = InstanceManager(
        pylet_client=pylet,
        scheduler_client=scheduler,
    )

    # Register model command
    manager.register_model_command(
        "Qwen/Qwen3-0.6B",
        "vllm serve Qwen/Qwen3-0.6B --port $PORT"
    )

    # Deploy 2 instances
    instances = await manager.deploy_instances(
        model_id="Qwen/Qwen3-0.6B",
        count=2,
        gpu_count=1,
    )

    print(f"Deployed {len(instances)} instances")
    for inst in instances:
        print(f"  - {inst.instance_id}: {inst.endpoint}")

    # Cleanup
    await pylet.shutdown()
    await scheduler.close()

asyncio.run(main())
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Planner Service                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │InstanceManager │  │  PyLetClient   │  │  SchedulerClient   │  │
│  │                │  │                │  │                    │  │
│  │ - deploy()     │──│ - submit()     │  │ - register()       │  │
│  │ - terminate()  │  │ - cancel()     │  │ - deregister()     │  │
│  │ - sync()       │  │ - list()       │  │ - health()         │  │
│  └────────┬───────┘  └────────┬───────┘  └─────────┬──────────┘  │
│           │                   │                     │            │
└───────────┼───────────────────┼─────────────────────┼────────────┘
            │                   │                     │
            ▼                   ▼                     ▼
     ┌──────────────┐    ┌──────────────┐     ┌──────────────┐
     │   Optimizer  │    │  PyLet Head  │     │  Scheduler   │
     │  (capacity)  │    │  (cluster)   │     │  (routing)   │
     └──────────────┘    └──────────────┘     └──────────────┘
```

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PYLET_HEAD_ADDRESS` | Yes | - | PyLet head node URL |
| `SCHEDULER_URL` | Yes | - | Scheduler service URL |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `SYNC_INTERVAL` | No | `30` | State sync interval (seconds) |
| `DEPLOY_TIMEOUT` | No | `300` | Instance startup timeout (seconds) |

### Model Configuration

Models are configured via the `register_model_command()` method:

```python
# vLLM backend
manager.register_model_command(
    "meta-llama/Llama-2-7b",
    "vllm serve meta-llama/Llama-2-7b --port $PORT --tensor-parallel-size 2"
)

# SGLang backend
manager.register_model_command(
    "meta-llama/Llama-2-7b",
    "python -m sglang.launch_server --model meta-llama/Llama-2-7b --port $PORT"
)
```

## API Reference

### InstanceManager

Main class for managing model instances.

```python
class InstanceManager:
    def __init__(
        self,
        pylet_client: AsyncPyLetClient,
        scheduler_client: SchedulerClient,
    ):
        """Initialize instance manager."""

    async def deploy_instances(
        self,
        model_id: str,
        count: int,
        gpu_count: int = 1,
        wait_ready: bool = True,
        register: bool = True,
    ) -> list[ManagedInstance]:
        """Deploy model instances with scheduler registration."""

    async def terminate_instances(
        self,
        model_id: str,
        count: int,
        deregister: bool = True,
    ) -> list[ManagedInstance]:
        """Terminate instances with scheduler deregistration."""

    async def sync_with_pylet(self) -> None:
        """Sync instance state with PyLet cluster."""
```

### PyLetClient

Wrapper for PyLet operations.

```python
class AsyncPyLetClient:
    async def init(self) -> None:
        """Initialize connection to PyLet head."""

    async def deploy_model(
        self,
        model_id: str,
        model_command: str,
        count: int = 1,
        gpu_count: int = 1,
    ) -> list[InstanceInfo]:
        """Deploy model instances via PyLet."""

    async def cancel_instances(
        self,
        instances: list[InstanceInfo],
        wait: bool = True,
    ) -> None:
        """Cancel instances."""
```

### SchedulerClient

HTTP client for scheduler registration.

```python
class SchedulerClient:
    async def register_instance(
        self,
        info: RegistrationInfo,
    ) -> bool:
        """Register instance with scheduler."""

    async def deregister_instance(
        self,
        instance_id: str,
    ) -> bool:
        """Deregister instance from scheduler."""
```

## Examples

### Deploy and Monitor

```python
import asyncio
from src.instance_manager import InstanceManager

async def deploy_and_monitor():
    # Setup (see Quick Start)
    manager = await setup_manager()

    # Deploy
    instances = await manager.deploy_instances("model-a", count=3)

    # Monitor
    while True:
        await manager.sync_with_pylet()
        running = await manager.get_instance_count("model-a")
        print(f"Running instances: {running}")
        await asyncio.sleep(10)
```

### Scale Up/Down

```python
async def scale_model(manager, model_id, target_count):
    current = await manager.get_instance_count(model_id)

    if target_count > current:
        # Scale up
        await manager.deploy_instances(
            model_id,
            count=target_count - current
        )
    elif target_count < current:
        # Scale down
        await manager.terminate_instances(
            model_id,
            count=current - target_count
        )

    print(f"Scaled {model_id}: {current} -> {target_count}")
```

### Integration with Optimizer

```python
async def apply_optimization(manager, optimizer):
    # Get optimization decisions
    changes = optimizer.compute_changes(
        current_deployment,
        target_distribution,
        capacity_matrix,
    )

    # Apply removals first
    for model_id, count in changes["remove"]:
        await manager.terminate_instances(model_id, count)

    # Then additions
    for model_id, count in changes["add"]:
        await manager.deploy_instances(model_id, count)
```

## Troubleshooting

### PyLet Connection Failed

```
Error: Failed to connect to PyLet at http://localhost:8000
```

**Solutions:**
1. Verify PyLet head is running: `curl http://localhost:8000/health`
2. Check network connectivity
3. Verify PYLET_HEAD_ADDRESS is correct

### Scheduler Registration Failed

```
Error: Failed to register instance test-0
```

**Solutions:**
1. Verify scheduler is running: `curl http://localhost:8080/health`
2. Check scheduler logs for errors
3. Verify SCHEDULER_URL is correct

### Instance Stuck in PENDING

**Solutions:**
1. Check PyLet worker availability: `pylet workers`
2. Verify GPU resources available: `nvidia-smi`
3. Check PyLet logs: `pylet logs <instance-id>`

### State Sync Issues

```
Warning: Instance pylet-123 not found in PyLet
```

**Solutions:**
1. Instance may have been cancelled externally
2. Run `await manager.sync_with_pylet()` to update state
3. Check PyLet for actual instance status

## Development

### Running Tests

```bash
cd planner

# Unit tests
uv run pytest tests/ -v

# Integration tests (requires PyLet cluster)
PYLET_HEAD_ADDRESS=http://localhost:8000 \
SCHEDULER_URL=http://localhost:8080 \
uv run pytest tests/integration/ -v
```

### Code Style

```bash
# Format
uv run black src/ tests/
uv run ruff check --fix src/ tests/

# Type check
uv run mypy src/
```

## See Also

- [PyLet Documentation](/home/yanweiye/Projects/pylet/README.md)
- [Scripts Module](../scripts/README.md)
- [Migration Guide](../docs/pylet_migration.md)
- [Phase 2 Tasks](../docs/pylet_migration/)
```

### Step 2: Create Quick Start Example

Create `planner/examples/quickstart.py`:

```python
#!/usr/bin/env python3
"""Quick Start: Deploy a model via the Planner.

This example demonstrates the minimal setup to deploy a model
using the SwarmPilot Planner with PyLet.

Prerequisites:
    - PyLet cluster running (head + workers)
    - Scheduler service running

Usage:
    # Set environment variables
    export PYLET_HEAD_ADDRESS=http://localhost:8000
    export SCHEDULER_URL=http://localhost:8080

    # Run example
    python examples/quickstart.py [MODEL_ID]

Example:
    python examples/quickstart.py Qwen/Qwen3-0.6B
"""

import asyncio
import os
import sys

from loguru import logger


async def main(model_id: str) -> None:
    """Deploy a model and verify it's working."""
    from src.pylet_client import AsyncPyLetClient
    from src.scheduler_client import SchedulerClient
    from src.instance_manager import InstanceManager

    pylet_address = os.getenv("PYLET_HEAD_ADDRESS", "http://localhost:8000")
    scheduler_url = os.getenv("SCHEDULER_URL", "http://localhost:8080")

    logger.info("=== SwarmPilot Planner Quick Start ===")
    logger.info(f"Model: {model_id}")
    logger.info(f"PyLet: {pylet_address}")
    logger.info(f"Scheduler: {scheduler_url}")

    # Initialize clients
    pylet = AsyncPyLetClient(head_address=pylet_address)
    await pylet.init()

    scheduler = SchedulerClient(scheduler_url=scheduler_url)

    manager = InstanceManager(
        pylet_client=pylet,
        scheduler_client=scheduler,
    )

    manager.register_model_command(
        model_id,
        f"vllm serve {model_id} --port $PORT --host 0.0.0.0"
    )

    try:
        logger.info(f"Deploying 1 instance of {model_id}...")
        instances = await manager.deploy_instances(
            model_id=model_id,
            count=1,
            gpu_count=1,
        )

        if instances:
            instance = instances[0]
            logger.info(f"Instance deployed: {instance.instance_id}")
            logger.info(f"Endpoint: {instance.endpoint}")

        logger.info("=== Quick Start Complete ===")

    finally:
        await scheduler.close()
        await pylet.shutdown()


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    asyncio.run(main(model))
```

### Step 3: Create Deploy Model Example

Create `planner/examples/deploy_model.py`:

```python
#!/usr/bin/env python3
"""Deploy Model Example: Full deployment workflow.

Shows complete deployment workflow including:
    - Multiple instance deployment
    - Health monitoring
    - Scaling operations
    - Graceful shutdown

Usage:
    python examples/deploy_model.py --model MODEL_ID --count N
"""

import argparse
import asyncio
import os
import signal

from loguru import logger


async def main(args: argparse.Namespace) -> None:
    """Run deployment workflow."""
    from src.pylet_client import AsyncPyLetClient
    from src.scheduler_client import SchedulerClient
    from src.instance_manager import InstanceManager

    pylet_address = os.getenv("PYLET_HEAD_ADDRESS", "http://localhost:8000")
    scheduler_url = os.getenv("SCHEDULER_URL", "http://localhost:8080")

    pylet = AsyncPyLetClient(head_address=pylet_address)
    await pylet.init()
    scheduler = SchedulerClient(scheduler_url=scheduler_url)
    manager = InstanceManager(pylet_client=pylet, scheduler_client=scheduler)

    manager.register_model_command(
        args.model,
        f"vllm serve {args.model} --port $PORT --host 0.0.0.0"
    )

    shutdown_event = asyncio.Event()

    def handle_signal(sig):
        logger.info(f"Received signal, shutting down...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_event_loop().add_signal_handler(sig, handle_signal)

    try:
        logger.info(f"Deploying {args.count} instances of {args.model}...")
        instances = await manager.deploy_instances(
            model_id=args.model,
            count=args.count,
            gpu_count=args.gpu,
        )

        for inst in instances:
            logger.info(f"  - {inst.instance_id}: {inst.endpoint}")

        logger.info("Monitoring instances (Ctrl+C to stop)...")
        while not shutdown_event.is_set():
            await manager.sync_with_pylet()
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass

    finally:
        logger.info("Terminating all instances...")
        count = await manager.get_instance_count(args.model)
        if count > 0:
            await manager.terminate_instances(args.model, count)
        await scheduler.close()
        await pylet.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model instances")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--count", type=int, default=1, help="Instance count")
    parser.add_argument("--gpu", type=int, default=1, help="GPUs per instance")

    asyncio.run(main(parser.parse_args()))
```

## Test Strategy

### Documentation Tests

```bash
# Verify README renders correctly
cat planner/README.md | head -50

# Check code examples are syntactically correct
python -m py_compile planner/examples/quickstart.py
python -m py_compile planner/examples/deploy_model.py
```

## Acceptance Criteria

- [ ] planner/README.md created with:
  - [ ] Quick Start section with minimal setup
  - [ ] Architecture diagram
  - [ ] Configuration reference
  - [ ] API reference for main classes
  - [ ] Examples section
  - [ ] Troubleshooting guide
- [ ] planner/examples/quickstart.py created and executable
- [ ] planner/examples/deploy_model.py created with full workflow
- [ ] All code examples are syntactically correct
- [ ] Examples work end-to-end with real cluster

## Next Steps

Phase 2 complete. Proceed to Phase 3:
- [PYLET-014](PYLET-014-scheduler-task-queue.md) for scheduler task queue (future)

## Code References

- PyLet client: [planner/src/pylet_client.py](../../planner/src/pylet_client.py)
- Scheduler client: [planner/src/scheduler_client.py](../../planner/src/scheduler_client.py)
- Instance manager: [planner/src/instance_manager.py](../../planner/src/instance_manager.py)
