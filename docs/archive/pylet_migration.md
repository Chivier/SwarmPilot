# Migration Plan: SwarmPilot to PyLet Integration

## Overview

This document describes the migration of SwarmPilot's instance management to PyLet, a lightweight cluster management framework. The migration simplifies instance management by using raw PyLet workers to expose model services directly to the scheduler.

## Requirements

1. All migration should follow test-driven development (TDD) principle
2. All migration should update documentation to stay current
3. All code should follow Google's Python Style Guide
4. Maintain backward compatibility during migration

## Architecture Comparison

### Current Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Scheduler  │────▶│  Instance   │────▶│   Model     │
│  (routing)  │◀────│  (queue)    │◀────│   (vLLM)    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│   Planner   │
│ (optimizer) │
└─────────────┘
```

### Target Architecture (with PyLet)

```
┌─────────────────────┐     ┌─────────────────────────────────┐
│      Scheduler      │────▶│        PyLet Worker             │
│  ┌───────────────┐  │     │  ┌───────────────────────────┐  │
│  │  Task Queue   │  │     │  │      Model Service        │  │
│  │  (Phase 3)    │  │     │  │      (vLLM/sglang)        │  │
│  └───────────────┘  │     │  │      exposed on $PORT     │  │
│     (routing)       │◀────│  └───────────────────────────┘  │
└─────────────────────┘     └─────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Planner   │
│ (optimizer) │
│      │      │
│      ▼      │
│ pylet.submit│
│ pylet.cancel│
└─────────────┘
```

**Key Change**: No wrapper layer. Model services are exposed directly via PyLet's automatic port allocation. Task queue functionality will be implemented at the scheduler side in Phase 3.

## PyLet Overview

PyLet is a lightweight cluster management framework providing:

- **Head Node**: Central controller managing worker registration and instance scheduling
- **Worker Node**: Process manager running on each machine with GPU resources
- **Instance**: A unit of work (process) running on a worker

### Key PyLet Concepts

| Concept | Description |
|---------|-------------|
| `pylet.init(address)` | Connect to PyLet head node |
| `pylet.submit(command, ...)` | Submit instance to cluster |
| `instance.wait()` | Wait for instance completion |
| `instance.cancel()` | Request instance cancellation |
| `instance.endpoint` | Get `host:port` when running |
| `$PORT` env var | Auto-allocated port for service discovery |

### PyLet Instance States

```
PENDING → ASSIGNED → RUNNING → COMPLETED
                  ↓            ↓
               UNKNOWN      FAILED
                  ↓            ↓
               CANCELLED ◀─────┘
```

## Migration Phases

### Phase 1: Direct Model Deployment via PyLet

Replace the instance service with direct PyLet-managed model services. Models expose their HTTP API directly on PyLet-assigned ports.

**Scope:**
- Deploy model services (vLLM, sglang) directly via PyLet
- Model registers itself with scheduler on startup
- No wrapper layer - raw model service exposed

### Phase 2: Planner PyLet Integration

Integrate planner with PyLet for instance lifecycle management.

**Scope:**
- Use PyLet client to submit/cancel instances
- Track PyLet instance IDs
- State synchronization with PyLet cluster
- Planner-managed scheduler registration/deregistration (instead of model self-registration)

### Phase 3: Scheduler-Side Task Queue (Future)

Move task queue functionality from instance to scheduler.

**Scope:**
- Implement request-level queuing in scheduler
- Priority-based task scheduling
- Callback handling for completed tasks

## Migration Tasks

All tasks are documented in separate files in `docs/pylet_migration/`:

### Phase 1: Direct Model Deployment

| Task ID | Title | Description |
|---------|-------|-------------|
| [PYLET-001](pylet_migration/PYLET-001-direct-model-deployment.md) | Direct Model Deployment | Deploy models directly via PyLet |
| [PYLET-002](pylet_migration/PYLET-002-model-registration.md) | Model Registration | Model self-registration with scheduler |
| [PYLET-003](pylet_migration/PYLET-003-signal-handling.md) | Signal Handling | Graceful shutdown on SIGTERM/SIGINT |
| [PYLET-004](pylet_migration/PYLET-004-health-monitoring.md) | Health Monitoring | Health check integration |
| [PYLET-005](pylet_migration/PYLET-005-phase1-integration-tests.md) | Phase 1 Integration Tests | End-to-end testing |
| [PYLET-005A](pylet_migration/PYLET-005A-scripts-readme.md) | Scripts README & Quick Start | Documentation for scripts module |

### Phase 2: Planner PyLet Integration

| Task ID | Title | Description |
|---------|-------|-------------|
| [PYLET-006](pylet_migration/PYLET-006-pylet-client-integration.md) | PyLet Client Integration | Add pylet dependency |
| [PYLET-007](pylet_migration/PYLET-007-instance-lifecycle.md) | Instance Lifecycle | Replace instance management |
| [PYLET-008](pylet_migration/PYLET-008-deployment-strategy.md) | Deployment Strategy | Integrate with optimizer |
| [PYLET-009](pylet_migration/PYLET-009-migration-optimizer.md) | Migration Optimizer | Update for pylet |
| [PYLET-010](pylet_migration/PYLET-010-state-tracking.md) | State Tracking | Maintain instance state |
| [PYLET-011](pylet_migration/PYLET-011-phase2-integration-tests.md) | Phase 2 Integration Tests | End-to-end testing |
| [PYLET-012](pylet_migration/PYLET-012-planner-managed-registration.md) | Planner-Managed Registration | Move register/deregister to planner |
| [PYLET-013](pylet_migration/PYLET-013-planner-readme.md) | Planner README & Quick Start | Documentation for planner module |

### Phase 3: Scheduler-Side Task Queue

| Task ID | Title | Description |
|---------|-------|-------------|
| [PYLET-014](pylet_migration/PYLET-014-scheduler-task-queue.md) | Scheduler Task Queue Design | Overall design and architecture |
| [PYLET-015](pylet_migration/PYLET-015-worker-queue-thread.md) | Worker Queue Thread | Implement `WorkerQueueThread` class |
| [PYLET-016](pylet_migration/PYLET-016-callback-mechanism.md) | Library-Based Callbacks | Implement `TaskResultCallback` |
| [PYLET-017](pylet_migration/PYLET-017-worker-queue-manager.md) | Worker Queue Manager | Implement `WorkerQueueManager` |
| [PYLET-018](pylet_migration/PYLET-018-queue-aware-scheduling.md) | Queue-Aware Scheduling | Integrate queue state with existing algorithms |
| [PYLET-019](pylet_migration/PYLET-019-api-integration.md) | API Integration | Update `/task/submit` endpoint |
| [PYLET-020](pylet_migration/PYLET-020-graceful-shutdown.md) | Graceful Shutdown | Handle worker deregistration |
| [PYLET-021](pylet_migration/PYLET-021-phase3-integration-tests.md) | Phase 3 Integration Tests | End-to-end testing |

## Phase 1: Direct Model Deployment Design

### Deployment Pattern

Models are deployed directly via PyLet without a wrapper:

```bash
# Direct model deployment
pylet submit "vllm serve Qwen/Qwen3-0.6B --port $PORT" --gpu 1
```

### Port Allocation

PyLet automatically allocates a port via the `$PORT` environment variable:

```
$PORT (from PyLet) → Model service HTTP API (vLLM, sglang, etc.)
```

### Model Service Interface

The model service exposes its native API directly:

```
# vLLM native endpoints
POST /v1/completions     - Generate completions
POST /v1/chat/completions - Chat completions
GET  /health             - Health check
GET  /v1/models          - List models
```

### Registration Flow

```
PyLet starts instance
        │
        ▼
┌───────────────────┐
│ Model starts on   │
│ $PORT             │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Model registers   │
│ with scheduler    │
│ (startup script)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Scheduler routes  │
│ requests directly │
│ to model          │
└───────────────────┘
```

## Phase 2: Planner Integration Design

### Planner Instance Map

The planner maintains a **scheduler-to-instances mapping** to track which scheduler owns which instances. This enables:
1. **Minimal change computation**: Only send changed instances to affected schedulers
2. **State synchronization**: Recover from failures by querying schedulers

```python
@dataclass
class SchedulerInstanceMap:
    """Maps schedulers to their owned instances.

    Key design:
    - One scheduler per model_id
    - Planner tracks which instances belong to which scheduler
    - On deployment change, planner computes minimal diff and syncs
    """

    # scheduler_endpoint -> set of instance_ids
    _map: dict[str, set[str]] = field(default_factory=dict)

    # model_id -> scheduler_endpoint
    _model_schedulers: dict[str, str] = field(default_factory=dict)

    def get_scheduler_for_model(self, model_id: str) -> str | None:
        """Get scheduler endpoint for a model."""
        return self._model_schedulers.get(model_id)

    def set_scheduler_for_model(self, model_id: str, scheduler_endpoint: str) -> None:
        """Set scheduler endpoint for a model."""
        self._model_schedulers[model_id] = scheduler_endpoint
        if scheduler_endpoint not in self._map:
            self._map[scheduler_endpoint] = set()

    def get_instances(self, scheduler_endpoint: str) -> set[str]:
        """Get instance IDs owned by a scheduler."""
        return self._map.get(scheduler_endpoint, set())

    def add_instance(self, scheduler_endpoint: str, instance_id: str) -> None:
        """Record that an instance belongs to a scheduler."""
        if scheduler_endpoint not in self._map:
            self._map[scheduler_endpoint] = set()
        self._map[scheduler_endpoint].add(instance_id)

    def remove_instance(self, scheduler_endpoint: str, instance_id: str) -> None:
        """Remove instance from scheduler's tracking."""
        if scheduler_endpoint in self._map:
            self._map[scheduler_endpoint].discard(instance_id)
```

### Instance List Sync Flow

The planner sends **complete target instance lists** to schedulers. Schedulers compute the diff internally.

```python
async def sync_deployment(
    target_deployment: dict[str, list[InstanceInfo]],
    scheduler_map: SchedulerInstanceMap,
) -> None:
    """Sync target deployment to schedulers.

    Args:
        target_deployment: model_id -> list of target instances
        scheduler_map: Current scheduler-instance mapping

    Flow:
    1. Cancel PyLet instances that will be removed FIRST
    2. Send new instance list to each scheduler
    3. Scheduler handles diff internally (add/remove/reschedule)
    4. Update planner's tracking map
    """
    for model_id, target_instances in target_deployment.items():
        scheduler_endpoint = scheduler_map.get_scheduler_for_model(model_id)
        if not scheduler_endpoint:
            continue

        # Get current instances for this scheduler
        current_instance_ids = scheduler_map.get_instances(scheduler_endpoint)
        target_instance_ids = {inst.instance_id for inst in target_instances}

        # 1. Cancel PyLet instances that will be removed FIRST
        to_remove = current_instance_ids - target_instance_ids
        for instance_id in to_remove:
            await pylet.cancel(instance_id)  # Cancel PyLet instance
            scheduler_map.remove_instance(scheduler_endpoint, instance_id)

        # 2. Start new PyLet instances
        to_add = target_instance_ids - current_instance_ids
        for inst in target_instances:
            if inst.instance_id in to_add:
                await pylet.submit(inst.command, gpu=inst.gpu, name=inst.instance_id)
                scheduler_map.add_instance(scheduler_endpoint, inst.instance_id)

        # 3. Send complete instance list to scheduler
        # Scheduler computes diff and handles add/remove/reschedule
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{scheduler_endpoint}/instance/sync",
                json={"instances": [asdict(i) for i in target_instances]},
            )
            result = response.json()
            logger.info(
                f"Synced {model_id}: +{len(result['added'])} "
                f"-{len(result['removed'])}, {result['rescheduled']} rescheduled"
            )
```

### Deployment Change Algorithm (Updated)

```python
async def apply_deployment_changes(
    deployment_changes: dict,
    scheduler_map: SchedulerInstanceMap,
) -> None:
    """Apply deployment changes using instance list sync.

    Key principles:
    1. Cancel PyLet instances BEFORE sending instance list
    2. Scheduler rescheduled tasks from removed instances to others
    3. Minimize changes - only send to affected schedulers
    """
    # Group changes by model_id
    for model_id, change in deployment_changes.items():
        scheduler_endpoint = scheduler_map.get_scheduler_for_model(model_id)

        # Compute target instance list
        current_instances = list(scheduler_map.get_instances(scheduler_endpoint))
        target_instances = current_instances.copy()

        # Remove instances
        for _ in range(change.get("remove", 0)):
            if target_instances:
                to_remove = target_instances.pop()
                # Cancel PyLet instance FIRST
                await pylet.cancel(to_remove)

        # Add instances
        for i in range(change.get("add", 0)):
            instance = await pylet.submit(
                f"vllm serve {model_id} --port $PORT",
                gpu=1,
                name=f"{model_id}-{uuid.uuid4().hex[:8]}",
            )
            await instance.wait_running()
            target_instances.append(InstanceInfo(
                instance_id=instance.name,
                endpoint=instance.endpoint,
                model_id=model_id,
            ))

        # Send complete target list to scheduler
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{scheduler_endpoint}/instance/sync",
                json={"instances": [asdict(i) for i in target_instances]},
            )

        # Update planner's tracking
        scheduler_map._map[scheduler_endpoint] = {i.instance_id for i in target_instances}
```

## Phase 3: Scheduler-Side Task Queue

This phase moves task queuing from instances to the scheduler, enabling centralized queue management with per-worker task threads.

### Key Design Constraints

Based on requirements review, Phase 3 adheres to these critical constraints:

| Constraint | Description |
|------------|-------------|
| **Single Model per Scheduler** | Each scheduler handles exactly one model type. Model ID is verified on instance sync. |
| **Instance List Sync API** | Planner submits complete target instance list via `POST /instance/sync`. Scheduler computes diff and handles add/remove internally. |
| **Reuse Existing Algorithms** | NO new scheduling algorithms. Use existing `min_expected_time`, `probabilistic`, `round_robin`, etc. Rescheduling reuses same algorithms. |
| **Unified Transparent Proxy** | ALL requests enter through the Proxy. Management requests handled directly; ALL other requests routed to Dispatcher. |
| **Priority Queue (Arrival Time)** | Worker queues are priority queues ordered by `enqueue_time`. Earlier arrival = higher priority = front of queue. |
| **Resilient Queues** | Queues don't exit on network errors. Tasks stay in queue with retry flag. Planner cancels PyLet instance before sending new list. |

See [PYLET-014](pylet_migration/PYLET-014-scheduler-task-queue.md) for complete design documentation.

### Motivation

After Phase 1 and 2, PyLet workers expose model services directly without a wrapper layer. This means:
- **No local task queue**: Workers no longer have a task queue (the instance service is deprecated)
- **Direct model API**: Workers expose raw model endpoints (e.g., vLLM's `/v1/completions`)
- **Synchronous execution**: Model APIs process one request at a time per worker

The scheduler must now manage task queuing to:
1. Queue tasks when workers are busy
2. Maintain FIFO ordering per worker
3. Handle callbacks for task completion
4. Support priority-based scheduling

### Architecture Overview

**Unified Proxy Architecture**: ALL requests enter through the Transparent Proxy. Management requests are handled directly by the scheduler; ALL other requests go through the Dispatcher to Worker Queues (queued & load-balanced).

```
                    ┌─────────────────────────────────────────────────────┐
                    │              PLANNER                                 │
                    │  ┌─────────────────────────────────────────────┐    │
                    │  │ Instance Management via Scheduler API        │    │
                    │  │ POST /instance/sync (target instance list)   │    │
                    │  │ Scheduler computes diff: add/remove          │    │
                    │  │ Removed instance tasks → rescheduled         │    │
                    │  └─────────────────────────────────────────────┘    │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│   SCHEDULER (Single Model Type: any - LLM, image gen, embeddings, etc.)   │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │              Transparent Proxy (Unified Entry Point)               │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │  Route by Path:                                              │  │   │
│  │  │  • /instance/*, /health, /info → Management API              │  │   │
│  │  │  • ALL other paths → Dispatcher → Worker Queue               │  │   │
│  │  │    (queued & load-balanced to workers)                       │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────┬─────────────────────────────────────┘   │
│                                │                                          │
│          ┌─────────────────────┴─────────────────────┐                   │
│          │                                           │                   │
│          ▼                                           ▼                   │
│  ┌───────────────┐             ┌──────────────────────────────────┐     │
│  │ Management    │             │           Dispatcher             │     │
│  │ API           │             │  ALL non-management requests     │     │
│  │ /instance/*   │             │  Scheduling Strategy             │     │
│  │ /health       │             │  (existing algos: min_time,      │     │
│  │ /info         │             │   probabilistic, round_robin)    │     │
│  └───────────────┘             └───────────────┬──────────────────┘     │
│                                                │                         │
│         ┌──────────────────────────────────────┼──────────────────┐     │
│         │                                      │                  │     │
│         ▼                                      ▼                  ▼     │
│  ┌─────────────┐                        ┌─────────────┐   ┌─────────────┐
│  │Worker Queue │                        │Worker Queue │   │Worker Queue │
│  │  Thread 1   │                        │  Thread 2   │   │  Thread N   │
│  │ ┌─────────┐ │                        │ ┌─────────┐ │   │ ┌─────────┐ │
│  │ │Priority │ │                        │ │Priority │ │   │ │Priority │ │
│  │ │(arriv.) │ │                        │ │(arriv.) │ │   │ │(arriv.) │ │
│  │ └─────────┘ │                        │ └─────────┘ │   │ └─────────┘ │
│  └──────┬──────┘                        └──────┬──────┘   └──────┬──────┘
│         │                                      │                  │      │
└─────────┼──────────────────────────────────────┼──────────────────┼──────┘
          │                                      │                  │
          ▼                                      ▼                  ▼
   ┌─────────────┐                        ┌─────────────┐   ┌─────────────┐
   │PyLet Worker │                        │PyLet Worker │   │PyLet Worker │
   │  (model_a)  │                        │  (model_a)  │   │  (model_a)  │
   │  Model API  │                        │  Model API  │   │  Model API  │
   └─────────────┘                        └─────────────┘   └─────────────┘
```

### Key Changes from Current Architecture

| Aspect | Current (Instance-Side Queue) | New (Scheduler-Side Queue) |
|--------|-------------------------------|----------------------------|
| **Queue Location** | Each instance has its own queue | Scheduler maintains per-worker queues |
| **Task Dispatch** | Async HTTP POST to `/task/submit` | Thread-based sync call to worker API |
| **Result Handling** | Instance callbacks scheduler | Thread invokes library callback |
| **Queue Management** | Instance manages FIFO locally | Scheduler manages FIFO per worker |
| **Worker Complexity** | Wrapper with queue + callback | Raw model API (vLLM, sglang) |

### Worker Queue Thread Design

Each registered worker gets a dedicated `WorkerQueueThread` that:

1. **Maintains a FIFO queue** for tasks assigned to that worker
2. **Processes tasks sequentially** in a dedicated thread (not blocking event loop)
3. **Makes synchronous HTTP calls** to the worker's model API
4. **Invokes callbacks** via library API when tasks complete

See [PYLET-015](pylet_migration/PYLET-015-worker-queue-thread.md) for detailed implementation.

### Callback Mechanism

The new callback mechanism is **library-based** rather than HTTP-based:

- Worker threads invoke callbacks via `asyncio.run_coroutine_threadsafe()`
- No HTTP overhead for result notification
- Direct integration with task registry and WebSocket manager

See [PYLET-016](pylet_migration/PYLET-016-callback-mechanism.md) for detailed implementation.

### Task Submission Flow (New)

```
1. Client submits task via POST /task/submit
   │
   ▼
2. Scheduler creates TaskRecord (status=PENDING)
   │
   ▼
3. Central Dispatcher selects worker using SchedulingStrategy
   │
   ├── Gets predictions for each worker
   ├── Considers queue depth at each worker
   └── Selects optimal worker
   │
   ▼
4. Task assigned to worker's WorkerQueueThread
   │
   ├── TaskRecord.assigned_instance = worker_id
   ├── TaskRecord.status = QUEUED
   └── WorkerQueueThread.enqueue(task)
   │
   ▼
5. WorkerQueueThread processes task (in dedicated thread)
   │
   ├── Waits for previous tasks to complete (FIFO)
   ├── TaskRecord.status = RUNNING
   ├── Calls worker API synchronously
   └── Invokes callback with result
   │
   ▼
6. TaskResultCallback handles completion (in main event loop)
   │
   ├── Updates TaskRecord (status, result/error)
   ├── Updates instance statistics
   ├── Records throughput
   └── Broadcasts to WebSocket subscribers
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Queue Storage** | In-memory (thread-safe Queue) | Simple, fast, sufficient for per-worker queues |
| **Thread Model** | One thread per worker | Avoids blocking event loop, simple concurrency |
| **Callback Pattern** | Library-based (`asyncio.run_coroutine_threadsafe`) | No HTTP overhead, direct integration |
| **Queue Depth Tracking** | Scheduler-side only | Workers are stateless (raw model API) |
| **Task Redistribution** | On deregister, return to central queue | Maintains FIFO ordering |

See [PYLET-014](pylet_migration/PYLET-014-scheduler-task-queue.md) for complete design documentation

## Migration Checklist

### Pre-Migration

- [ ] Review all task files in `docs/pylet_migration/`
- [ ] Set up PyLet development environment
- [ ] Create feature branch for migration

### Phase 1 Checklist

- [ ] PYLET-001: Direct model deployment working
- [ ] PYLET-002: Model registration with scheduler
- [ ] PYLET-003: Signal handling for graceful shutdown
- [ ] PYLET-004: Health monitoring
- [ ] PYLET-005: Integration tests pass
- [ ] PYLET-005A: Scripts README & Quick Start guide

### Phase 2 Checklist

- [ ] PYLET-006: PyLet client integration
- [ ] PYLET-007: Instance lifecycle management
- [ ] PYLET-008: Deployment strategy integration
- [ ] PYLET-009: Migration optimizer update
- [ ] PYLET-010: State tracking
- [ ] PYLET-011: Integration tests pass
- [ ] PYLET-012: Planner-managed registration/deregistration
- [ ] PYLET-013: Planner README & Quick Start guide

### Phase 3 Checklist

- [ ] PYLET-014: Scheduler task queue design
- [ ] PYLET-015: Worker queue thread implementation
- [ ] PYLET-016: Library-based callback mechanism
- [ ] PYLET-017: Worker queue manager
- [ ] PYLET-018: Queue-aware scheduling strategies
- [ ] PYLET-019: API integration (`/task/submit` update)
- [ ] PYLET-020: Graceful shutdown handling
- [ ] PYLET-021: Phase 3 integration tests pass

### Post-Migration

- [ ] Performance benchmarks
- [ ] Documentation update
- [ ] Deprecation notices for old instance service

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model startup failures | Health check with retry logic |
| Scheduler registration timeout | Retry with exponential backoff |
| Port conflicts | Use PyLet's automatic port allocation |
| Signal handling issues | Test graceful shutdown thoroughly |

## Dependencies

### Phase 1

- PyLet library (local development version)
- Model services (vLLM, sglang)

### Phase 2

- All Phase 1 dependencies
- PyLet async API (`pylet.aio`)

### Phase 3

- All Phase 2 dependencies
- Scheduler modifications

## File Structure After Migration

```
swarmpilot-refresh/
├── planner/
│   └── src/
│       ├── pylet_client.py       # Phase 2: PyLet integration wrapper
│       ├── scheduler_client.py   # Phase 2: Scheduler registration client
│       ├── deployment_service.py # Phase 2: Use pylet + scheduler client
│       └── ...
│
├── scheduler/
│   └── src/
│       ├── services/
│       │   ├── worker_queue_thread.py   # Phase 3: Per-worker queue thread
│       │   ├── worker_queue_manager.py  # Phase 3: Worker thread management
│       │   ├── task_result_callback.py  # Phase 3: Library-based callbacks
│       │   ├── central_queue.py         # MODIFIED: Integration with worker queues
│       │   └── task_dispatcher.py       # MODIFIED: Use worker queue threads
│       ├── algorithms/                  # Phase 3: No changes (reuse existing)
│       └── ...
│
├── instance/                      # DEPRECATED after Phase 1
│   └── ...
│
└── docs/
    ├── pylet_migration.md        # This document
    └── pylet_migration/          # Task files
        ├── PYLET-001-direct-model-deployment.md
        ├── ...
        ├── PYLET-014-scheduler-task-queue.md
        ├── PYLET-015-worker-queue-thread.md
        ├── PYLET-016-callback-mechanism.md
        ├── PYLET-017-worker-queue-manager.md
        ├── PYLET-018-queue-aware-scheduling.md
        ├── PYLET-019-api-integration.md
        ├── PYLET-020-graceful-shutdown.md
        └── PYLET-021-phase3-integration-tests.md
```

## References

- [PyLet Repository](/home/yanweiye/Projects/pylet)
- [SwarmPilot Instance Service](../instance/src/)
- [SwarmPilot Planner Service](../planner/src/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
