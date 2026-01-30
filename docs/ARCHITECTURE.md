# Architecture

System design overview of SwarmPilot's three services and how they interact.

## System Topology

```
                  ┌──────────┐
                  │  Client  │
                  └────┬─────┘
                       │ HTTP / WebSocket
                       ▼
                ┌─────────────┐         ┌──────────────┐
                │  Scheduler  │◀───────▶│  Predictor   │
                │  :8000/v1   │  lib    │  :8001       │
                └──────┬──────┘         └──────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐
     │Instance │ │Instance │ │Instance │
     │ (worker)│ │ (worker)│ │ (worker)│
     └─────────┘ └─────────┘ └─────────┘
                       ▲
                       │ deploys / drains / migrates
                ┌──────┴──────┐
                │   Planner   │───── PyLet Cluster
                │  :8002/v1   │
                └─────────────┘
```

## Services

### Scheduler (port 8000)

Routes incoming tasks to compute instances. All endpoints use the `/v1/` prefix.

**Key components:**

| Component | File | Purpose |
|-----------|------|---------|
| `InstanceRegistry` | `registry/instance_registry.py` | Tracks registered instances and their state |
| `TaskRegistry` | `registry/task_registry.py` | Tracks task lifecycle (pending -> running -> completed) |
| `WorkerQueueManager` | `services/worker_queue_manager.py` | Manages per-instance task queues |
| `WorkerQueueThread` | `services/worker_queue_thread.py` | Executes queued tasks on worker instances |
| `PredictorClient` | `clients/predictor_library_client.py` | Calls Predictor for runtime estimates |
| `ConnectionManager` | `services/websocket_manager.py` | WebSocket connections for real-time results |
| `PlannerRegistrar` | `services/planner_registrar.py` | Registers this scheduler with a Planner |

**Task lifecycle:**
1. Client submits task via `POST /v1/task/submit`
2. Scheduler calls Predictor to estimate runtime per instance
3. Scheduling strategy selects the best instance
4. Task is queued in the selected instance's `WorkerQueueThread`
5. Worker sends HTTP request to the instance
6. Instance returns result via `POST /v1/callback/task_result`
7. Client retrieves result via `GET /v1/task/info` or WebSocket `/v1/task/get_result`

### Predictor (port 8001)

MLP-based runtime prediction. Endpoints have **no** prefix (mounted at root).

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **ExpectError** | MSE-based MLP that predicts `(expected_runtime_ms, error_margin_ms)` |
| **Quantile** | Quantile regression MLP that predicts runtime at configurable quantiles |
| **Model Cache** | LRU cache of loaded predictor models for fast inference |
| **Model Storage** | On-disk persistence of trained models as `.json` files |
| **Preprocessor** | Feature transformation pipeline applied before prediction/training |

**Prediction flow:**
1. Scheduler sends features + platform_info via `POST /predict`
2. Predictor loads trained model (from cache or disk)
3. MLP forward pass produces runtime estimate
4. Result returned to Scheduler for scheduling decisions

### Planner (port 8002)

Deployment optimization using mathematical programming. Core endpoints use `/v1/` prefix. PyLet endpoints are mounted under `/v1/` via a router.

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **Optimizer** | Simulated Annealing or Integer Programming to find optimal instance-to-model mapping |
| **PyLet** | Cluster manager that provisions, drains, and terminates instances |
| **SchedulerRegistry** | Maps model IDs to scheduler URLs for multi-scheduler setups |
| **AvailableInstanceStore** | Tracks instances available for migration |
| **InstanceTimeline** | Records deployment events over time |

**Deployment flow:**
1. Operator calls `POST /v1/deploy` with optimization parameters
2. Planner runs optimizer to compute target instance allocation
3. PyLet reconciles current cluster state toward target
4. New instances register with the appropriate Scheduler
5. Removed instances are drained before termination

---

## Scheduling Strategies

The Scheduler supports 7 built-in strategies, selectable at runtime via `POST /v1/strategy/set`.

| Strategy Name | Key | Description |
|---------------|-----|-------------|
| Adaptive Bootstrap | `adaptive_bootstrap` | **Default.** Uses bootstrapped prediction intervals to balance load |
| Minimum Expected Time | `min_time` | Greedy: assigns to instance with shortest predicted queue time |
| Probabilistic | `probabilistic` | Monte Carlo sampling at a target quantile (default 0.9) |
| Round Robin | `round_robin` | Cyclic assignment across instances |
| Random | `random` | Uniform random instance selection |
| Power of Two | `po2` | Pick 2 random instances, choose the one with shorter queue |
| Serverless | `severless` | Min expected time with serverless scaling semantics |

The default strategy is configured via `SCHEDULING_STRATEGY` (default: `adaptive_bootstrap`).

---

## Communication Patterns

### Scheduler <-> Predictor

The Scheduler embeds the Predictor as a library by default (no HTTP calls). When running the Predictor as a standalone service, the Scheduler uses a WebSocket client for low-latency batch predictions.

### Scheduler <-> Instances

- **Registration:** Instance sends `POST /v1/instance/register` on startup
- **Task execution:** Scheduler's `WorkerQueueThread` sends HTTP POST to instance endpoint
- **Result callback:** Instance sends `POST /v1/callback/task_result` when done
- **Drain/Remove:** Scheduler coordinates graceful shutdown before instance removal

### Planner <-> Scheduler

- **Registration:** Scheduler registers with Planner via `POST /v1/scheduler/register` on startup (requires `PLANNER_REGISTRATION_URL`, `SCHEDULER_MODEL_ID`, `SCHEDULER_SELF_URL`)
- **Forwarding:** Planner includes dummy Scheduler-compatible endpoints so PyLet-managed instances can register directly with the Planner

### Planner <-> PyLet

The Planner uses the PyLet Python SDK (`pylet.init()`, `pylet.submit()`, `pylet.cancel()`) to provision and terminate instances on a compute cluster.

---

## File Structure

```
swarmpilot/
├── scheduler/
│   ├── api.py                    # FastAPI endpoints
│   ├── cli.py                    # CLI: sscheduler
│   ├── config.py                 # Environment-based config
│   ├── models.py                 # Pydantic request/response models
│   ├── algorithms/               # Scheduling strategies
│   │   ├── base.py               # Abstract SchedulingStrategy
│   │   ├── factory.py            # Strategy factory
│   │   ├── adaptive_bootstrap.py
│   │   ├── min_expected_time.py
│   │   ├── probabilistic.py
│   │   ├── round_robin.py
│   │   ├── random.py
│   │   ├── power_of_two.py
│   │   └── serverless.py
│   ├── registry/
│   │   ├── instance_registry.py  # Instance state
│   │   └── task_registry.py      # Task state
│   ├── services/
│   │   ├── worker_queue_manager.py
│   │   ├── worker_queue_thread.py
│   │   ├── websocket_manager.py
│   │   ├── planner_registrar.py
│   │   └── task_result_callback.py
│   ├── clients/
│   │   ├── predictor_library_client.py
│   │   └── training_client.py
│   └── utils/
├── predictor/
│   ├── cli.py                    # CLI: spredictor
│   ├── config.py                 # Pydantic settings (PREDICTOR_* prefix)
│   ├── models.py                 # Pydantic models
│   ├── api/
│   │   ├── app.py                # FastAPI application
│   │   └── routes/
│   │       ├── prediction.py     # POST /predict
│   │       ├── training.py       # POST /train
│   │       ├── models.py         # GET /list
│   │       ├── health.py         # GET /health
│   │       ├── cache.py          # /cache/stats, /cache/clear
│   │       └── websocket.py      # WS /ws/predict
│   ├── predictor/
│   │   ├── base.py               # Abstract predictor
│   │   ├── expect_error.py       # ExpectError MLP
│   │   └── quantile.py           # Quantile regression MLP
│   └── storage/
│       └── model_storage.py      # On-disk model persistence
├── planner/
│   ├── api.py                    # FastAPI endpoints
│   ├── pylet_api.py              # PyLet router (mounted at /v1)
│   ├── cli.py                    # CLI: splanner
│   ├── config.py                 # Environment-based config
│   ├── models.py                 # Pydantic models
│   ├── core/
│   │   └── swarm_optimizer.py    # SA and IP optimizers
│   ├── pylet/
│   │   ├── client.py             # PyLet API client
│   │   ├── deployment_service.py # High-level deployment
│   │   ├── deployment_executor.py
│   │   ├── instance_manager.py
│   │   ├── migration_executor.py
│   │   └── scheduler_client.py
│   └── scheduler_registry.py     # Model -> Scheduler URL mapping
└── graph/                        # Client library
```
