# Deployment Interface Redesign

Date: 2026-03-12

## Motivation

PyLet provides an elegant, minimal API for remote instance management (`pylet.submit()` -> `instance.endpoint`). However, SwarmPilot wraps it in three abstraction layers and exposes multiple overlapping REST endpoints (`/deploy`, `/deploy_manually`, `/optimize`, `/scale`), making the deployment experience unnecessarily complex.

This design introduces a PyLet-inspired Python SDK and CLI extension that provides three clear deployment paths while preserving the Planner's core optimization capability.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Entry points | Python SDK + `splanner` CLI extension | SDK for scripting, CLI for operations |
| Connection target | Planner (8002) | Already holds PyLet integration and Scheduler Registry |
| Scheduler registration | opt-in: auto-discover / manual override / None | Custom workloads don't need scheduling |
| Lifecycle model | Async handle + `.wait_ready()` | Non-blocking by default, user chooses when to wait |
| Optimization deployment | `register()` + `deploy()` (always full) | Planner's SA+IP optimizer needs global view |
| CLI namespace | Extend existing `splanner` | Avoid new CLI binary, reuse existing infra |

## Three Deployment Paths

| | `register()` + `deploy()` | `serve()` | `run()` |
|---|---|---|---|
| Optimization | Planner computes optimal allocation | Skipped, user specifies replicas | Skipped |
| Scheduler | Auto-register all | Auto / manual / None | None |
| Use case | Production multi-model deployment | Quick test, single model, custom inference | Arbitrary workload |
| Replicas | Optimizer decides | User specifies | 1 |

## SDK API

### Initialization

```python
import swarmpilot

# Global init (single cluster)
swarmpilot.init("http://planner:8002")

# Non-global (multi-cluster)
cluster = swarmpilot.connect("http://another-planner:8002")
cluster.serve(...)
```

### Path 1: Optimized Deployment

```python
# Lazy registration (no deployment yet)
swarmpilot.register("Qwen/Qwen2.5-7B", gpu=1, target_qps=100)
swarmpilot.register("meta-llama/Llama-3-8B", gpu=2, target_qps=50)

# Full optimization + deployment (always deploys ALL registered models)
result = swarmpilot.deploy()
result.wait_ready()
result.plan                                # optimizer output details
result["Qwen/Qwen2.5-7B"].scheduler       # "http://sched-1:8000"
result["Qwen/Qwen2.5-7B"].endpoints       # ["http://w1:30001", ...]

# Update requirements and re-optimize
swarmpilot.register("Qwen/Qwen2.5-7B", gpu=1, target_qps=200)
result = swarmpilot.deploy()  # incremental adjustment
```

### Path 2: Manual Deployment

```python
# Standard model serving
inst = swarmpilot.serve("Qwen/Qwen2.5-7B", gpu=1)
inst.wait_ready()
inst.endpoint     # "http://w1:30001"
inst.scheduler    # "http://sched-1:8000" (auto-discovered)

# Multi-replica
insts = swarmpilot.serve("Qwen/Qwen2.5-7B", gpu=1, replicas=3)
insts.scheduler   # shared scheduler address
insts.endpoints   # list of all replica endpoints

# Custom name
inst = swarmpilot.serve("Qwen/Qwen2.5-7B", gpu=1, name="qwen-prod")

# Custom inference command + scheduler
inst = swarmpilot.serve(
    "python my_inference_server.py --model custom --port $PORT",
    gpu=2, name="custom-inference",
    scheduler="http://sched:8000"
)

# No scheduler (Planner-managed only)
inst = swarmpilot.serve("Qwen/Qwen2.5-7B", gpu=1, scheduler=None)
inst.scheduler    # None
```

### Path 3: Custom Workload

```python
proc = swarmpilot.run("python crawler.py --port $PORT", gpu=0, name="crawler")
proc.wait_ready()
proc.endpoint     # "http://w2:34521"
proc.scheduler    # None (always)
```

### Common Operations

```python
swarmpilot.ps()                             # list all instances
swarmpilot.scale("Qwen/Qwen2.5-7B", replicas=5)
swarmpilot.terminate("crawler")             # by name
insts.terminate()                           # by handle
```

## CLI Design (splanner extension)

### Path 1: Optimized Deployment

```bash
splanner register Qwen/Qwen2.5-7B --gpu 1 --target-qps 100
splanner register meta-llama/Llama-3-8B --gpu 2 --target-qps 50
splanner deploy                        # full optimization + deployment
splanner deploy --wait --timeout 300
```

### Path 2: Manual Deployment

```bash
splanner serve Qwen/Qwen2.5-7B --gpu 1
splanner serve Qwen/Qwen2.5-7B --gpu 1 --replicas 3
splanner serve Qwen/Qwen2.5-7B --gpu 1 --name qwen-prod
splanner serve "python my_server.py --port \$PORT" \
    --gpu 2 --name custom-inference --scheduler http://sched:8000
splanner serve Qwen/Qwen2.5-7B --gpu 1 --no-scheduler
splanner serve Qwen/Qwen2.5-7B --gpu 1 --wait --timeout 300
```

### Path 3: Custom Workload

```bash
splanner run "python crawler.py --port \$PORT" --gpu 0 --name crawler
```

### Common Operations

```bash
splanner ps
splanner ps --model Qwen/Qwen2.5-7B
splanner info qwen-7b-abc12
splanner scale Qwen/Qwen2.5-7B --replicas 5
splanner terminate crawler
splanner terminate --model Qwen/Qwen2.5-7B
splanner terminate --all
```

### CLI-SDK Mapping

| CLI | SDK |
|---|---|
| `splanner register MODEL ...` | `swarmpilot.register(model, ...)` |
| `splanner deploy` | `swarmpilot.deploy()` |
| `splanner serve MODEL/CMD ...` | `swarmpilot.serve(model_or_cmd, ...)` |
| `splanner run CMD ...` | `swarmpilot.run(cmd, ...)` |
| `splanner ps` | `swarmpilot.ps()` |
| `splanner scale MODEL --replicas N` | `swarmpilot.scale(model, replicas=N)` |
| `splanner terminate NAME` | `swarmpilot.terminate(name)` |
| `splanner info NAME` | Instance attributes |
| `--no-scheduler` | `scheduler=None` |
| `--scheduler URL` | `scheduler="URL"` |
| `--wait --timeout N` | `.wait_ready(timeout=N)` |

## Planner API Endpoints

New endpoints (old endpoints preserved as legacy):

```
# Optimized deployment path
POST   /v1/register              # Register model requirements (lazy)
POST   /v1/deploy                # Trigger optimization + full deployment
GET    /v1/registered            # List registered model requirements

# Manual deployment path
POST   /v1/serve                 # Manual model/service deployment
POST   /v1/run                   # Start custom workload

# Common
GET    /v1/instances             # List all instances
GET    /v1/instances/{name}      # Instance details
POST   /v1/scale                 # Scale replicas
POST   /v1/terminate             # Terminate instances
```

### POST /v1/serve

```json
// Request
{
  "model_or_command": "Qwen/Qwen2.5-7B",
  "name": null,
  "replicas": 1,
  "gpu": 1,
  "scheduler": "auto",
  "memory": null,
  "env": {}
}

// Response
{
  "name": "Qwen-Qwen2.5-7B",
  "model": "Qwen/Qwen2.5-7B",
  "command": "vllm serve Qwen/Qwen2.5-7B --port $PORT",
  "replicas": [
    {"endpoint": "http://w1:30001", "status": "pending"}
  ],
  "scheduler": "http://sched-1:8000",
  "status": "deploying"
}
```

### POST /v1/run

```json
// Request
{
  "command": "python crawler.py --port $PORT",
  "name": "crawler",
  "gpu": 0,
  "memory": null,
  "env": {}
}

// Response
{
  "name": "crawler",
  "command": "python crawler.py --port $PORT",
  "endpoint": null,
  "scheduler": null,
  "status": "deploying"
}
```

### Internal Flow

```
serve(model_or_cmd, scheduler="auto")
  |
  +-- PyLetClient.deploy(command, gpu, replicas)
  |     \-- pylet.submit("... --port $PORT", gpu=gpu)
  |
  +-- health check (wait for instance ready)
  |
  +-- scheduler resolution:
  |     +-- "auto"       -> scheduler_registry.get_scheduler_info(model)
  |     +-- None         -> skip registration
  |     \-- "http://..." -> use specified address
  |
  \-- register to scheduler (if not None)
        \-- POST {scheduler}/v1/instance/register


deploy() (optimized path)
  |
  +-- collect all registered model requirements
  |
  +-- SwarmOptimizer.optimize(requirements, cluster_capacity)
  |     \-- SA + IP solver -> optimal replica allocation
  |
  +-- for each model in plan:
  |     +-- PyLetClient.deploy(model, gpu, replicas)
  |     +-- health check
  |     \-- register to scheduler (auto-discover)
  |
  \-- return DeploymentResult


run(command, name)
  |
  +-- PyLetClient.submit(command, gpu, name)
  |
  \-- no registration, return Process handle
```

## Data Models (SDK Return Types)

```python
class Instance:
    name: str                    # user-specified or auto-generated
    model: str | None            # model name; None for custom commands
    command: str                 # actual command executed
    endpoint: str | None         # None while pending
    scheduler: str | None        # scheduler address or None
    status: str                  # "pending" | "deploying" | "running" | "failed"
    gpu: int

    def wait_ready(self, timeout: int = 300) -> None: ...
    def terminate(self) -> None: ...


class InstanceGroup:
    name: str
    model: str | None
    command: str
    instances: list[Instance]
    scheduler: str | None        # shared across all replicas

    @property
    def endpoints(self) -> list[str]: ...

    def wait_ready(self, timeout: int = 300) -> None: ...
    def scale(self, replicas: int) -> None: ...
    def terminate(self) -> None: ...


class Process:
    name: str
    command: str
    endpoint: str | None
    scheduler: None              # always None
    status: str
    gpu: int

    def wait_ready(self, timeout: int = 300) -> None: ...
    def terminate(self) -> None: ...


class DeploymentResult:
    plan: dict                   # optimization plan details
    groups: dict[str, InstanceGroup]  # model_id -> InstanceGroup
    status: str                  # "deploying" | "ready" | "partial_failure"

    def __getitem__(self, model: str) -> InstanceGroup: ...
    def wait_ready(self, timeout: int = 600) -> None: ...
    def terminate(self) -> None: ...


class ClusterState:
    instances: list[Instance]
    processes: list[Process]
    groups: list[InstanceGroup]
```

## Error Handling

```python
from swarmpilot.errors import (
    DeployError,         # PyLet deployment failure
    RegistrationError,   # Scheduler registration failure
    TimeoutError,        # wait_ready timeout
    SchedulerNotFound,   # auto-discover can't find scheduler for model
    OptimizationError,   # Planner optimizer infeasible (insufficient resources)
    ModelNotDeployed,    # predictor op on model with no scheduler mapping
)

# Partial success: 2 of 3 replicas succeed
try:
    insts = swarmpilot.serve("Qwen/Qwen2.5-7B", gpu=1, replicas=3)
    insts.wait_ready(timeout=300)
except DeployError as e:
    e.succeeded   # [Instance(...), Instance(...)]
    e.failed      # [{"replica": 2, "error": "GPU exhausted"}]

# Optimizer infeasible
try:
    result = swarmpilot.deploy()
except OptimizationError as e:
    e.reason      # "Insufficient GPU: need 8, available 5"

# Scheduler not found (serve with auto-discover)
try:
    inst = swarmpilot.serve("unknown-model", gpu=1)
except SchedulerNotFound as e:
    e.model       # "unknown-model"
    e.hint        # "Use scheduler='http://...' or scheduler=None"

# Predictor operation before model deployment
try:
    swarmpilot.predictor.upload(model="Qwen/Qwen2.5-7B",
                                 name="llm-output-predictor",
                                 feature="input_text",
                                 path="./my_preprocessor/")
except ModelNotDeployed as e:
    e.model       # "Qwen/Qwen2.5-7B"
    e.hint        # "Deploy the model first with swarmpilot.serve() or swarmpilot.deploy()"
```

## Predictor Management

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Preprocessor granularity | Per-model (all ops require model) | Preprocessor is uploaded/registered to a specific Scheduler via model mapping |
| Preprocessor format | Self-contained folder (class + model file) | Co-located files simplify server-side path resolution |
| Upload method | Local upload (tar.gz) + remote path pull (both land on Scheduler local) | Scheduler must have preprocessor locally to load it |
| MLP training | Auto + manual override | Default auto-training continues; users can retrain or supply custom data |
| Post-training strategy | Auto-switch to Probabilistic | MLP training completion triggers strategy change on Scheduler |
| API routing | Planner init → SDK direct-connects to Scheduler | Minimizes Planner load; Predictor is embedded in Scheduler |
| Predictor optional | System works without Predictor (Round-Robin) | Predictor is an enhancement, not a requirement |
| Ordering | Deploy first, then predictor setup | Scheduler address must be known before predictor ops |
| Mapping check | SDK validates scheduler mapping before ALL predictor ops | Raises ModelNotDeployed with actionable hint if mapping missing |
| Scheduler lifecycle | External (user-managed) | Scheduler may run on different machines; not managed by SDK |
| Registration failure | Return error, no rollback | PyLet instances are not auto-cancelled on Scheduler registration failure |
| Inference routing | Scheduler proxies all non-management API calls to instances | No SDK wrapper needed; user sends requests to inst.scheduler directly |

### Preprocessor Folder Structure

```
my_preprocessor/
├── preprocessor.py   # Class inheriting BasePreprocessorV2
└── model.pt          # Model weights (same dir for easy path resolution)
```

The class in `preprocessor.py` must inherit from `BasePreprocessorV2` and implement the standard interface. The model file must be in the same directory so the server can resolve its path relative to the class file.

### SDK API

```python
import swarmpilot

swarmpilot.init("http://planner:8002")
# SDK fetches scheduler addresses from Planner;
# predictor operations direct-connect to Scheduler

# ── Preprocessor Management ──
# All preprocessor ops are per-model (routed to model's Scheduler)

# Upload local preprocessor to a model's Scheduler
# SDK packs folder → uploads to Scheduler → Scheduler stores locally
swarmpilot.predictor.upload(model="Qwen/Qwen2.5-7B",
                             name="llm-output-predictor",
                             feature="input_text",
                             path="./my_preprocessor/")

# Register from remote path — Scheduler pulls and stores locally
# (path must be accessible from Scheduler's machine, e.g. shared NFS)
swarmpilot.predictor.register(model="Qwen/Qwen2.5-7B",
                               name="llm-output-predictor",
                               feature="input_text",
                               path="/shared/nfs/preprocessors/my_pp/")

# Same preprocessor for another model (uploads to that model's Scheduler)
swarmpilot.predictor.upload(model="meta-llama/Llama-3-8B",
                             name="llm-output-predictor",
                             feature="input_text",
                             path="./my_preprocessor/")

# Remove preprocessor from a model
swarmpilot.predictor.remove(model="Qwen/Qwen2.5-7B",
                             name="llm-output-predictor")

# List preprocessors for a model
swarmpilot.predictor.preprocessors(model="Qwen/Qwen2.5-7B")
# [PreprocessorInfo(name="llm-output-predictor", feature="input_text", path="...")]

# ── MLP Online Training ──

# Check training status
swarmpilot.predictor.status("Qwen/Qwen2.5-7B")
# ModelStatus(
#     model="Qwen/Qwen2.5-7B",
#     samples_collected=87,
#     last_trained="2026-03-12T10:30:00",
#     prediction_types=["expect_error", "quantile"],
#     metrics={"mse": 0.03, "mae": 12.5},
#     strategy="round_robin",
#     preprocessors=[("input_text", "llm-output-predictor")]
# )

# Manual retrain (using accumulated samples)
# On success: Scheduler auto-switches to Probabilistic strategy
result = swarmpilot.predictor.retrain("Qwen/Qwen2.5-7B")
result.samples_trained   # 87
result.metrics           # {"mse": 0.02, "mae": 11.3}
result.strategy          # "probabilistic" (auto-switched)

# Train with custom data (also triggers strategy switch)
result = swarmpilot.predictor.train("Qwen/Qwen2.5-7B", data=my_data)

# List all trained models
swarmpilot.predictor.models()
# [ModelInfo(model="Qwen/Qwen2.5-7B", ...), ...]

# Manual prediction (debugging)
pred = swarmpilot.predictor.predict("Qwen/Qwen2.5-7B", features={...})
pred.expected_runtime_ms  # 1234.5
pred.quantiles            # {0.5: 1200, 0.9: 1500, 0.95: 1700}
```

### CLI Extension

```bash
# Preprocessor management (all require --model)
splanner predictor upload --model Qwen/Qwen2.5-7B \
    --name llm-output-predictor --feature input_text --path ./my_preprocessor/
splanner predictor register --model Qwen/Qwen2.5-7B \
    --name llm-output-predictor --feature input_text --path /shared/nfs/my_pp/
splanner predictor remove --model Qwen/Qwen2.5-7B --name llm-output-predictor
splanner predictor preprocessors --model Qwen/Qwen2.5-7B

# MLP training
splanner predictor status Qwen/Qwen2.5-7B
splanner predictor retrain Qwen/Qwen2.5-7B
splanner predictor models

# Prediction (debugging)
splanner predictor predict Qwen/Qwen2.5-7B --features '{"input_text": "hello"}'
```

### Scheduler API Endpoints (new)

SDK direct-connects to Scheduler for predictor operations.
All endpoints are on the Scheduler that serves the target model:

```
# Preprocessor management (per-model, on model's Scheduler)
POST   /v1/predictor/preprocessor/upload     # Upload preprocessor (multipart) → store locally
POST   /v1/predictor/preprocessor/register   # Pull from remote path → copy to local storage
DELETE /v1/predictor/preprocessor/{name}     # Remove preprocessor
GET    /v1/predictor/preprocessors           # List preprocessors on this Scheduler

# MLP training
GET    /v1/predictor/status                  # Training status & metrics (Scheduler knows its model)
POST   /v1/predictor/retrain                 # Retrain → auto-switch to Probabilistic
POST   /v1/predictor/train                   # Train with custom data → auto-switch to Probabilistic
GET    /v1/predictor/models                  # List trained models
POST   /v1/predictor/predict                 # Manual prediction (debug)
```

Planner exposes a discovery endpoint for SDK initialization:

```
GET    /v1/schedulers                        # Returns model → scheduler address map
```

### Internal Flow

```
swarmpilot.init("http://planner:8002")
  │
  └─ GET /v1/schedulers → SDK caches scheduler address map
       e.g. {"Qwen/Qwen2.5-7B": "http://sched-1:8000", ...}


swarmpilot.predictor.upload(model, name, feature, path="./local/")
  │
  ├─ SDK: check scheduler mapping for model
  │    ├─ found → continue
  │    └─ not found → raise ModelNotDeployed(...)
  ├─ SDK: tar.gz pack local folder
  ├─ SDK → Scheduler: POST /v1/predictor/preprocessor/upload
  │    (multipart: archive + name + feature metadata)
  └─ Scheduler: extract to local storage dir, register in preprocessor chain
       e.g. {scheduler_storage}/preprocessors/{name}/
            ├── preprocessor.py
            └── model.pt


swarmpilot.predictor.register(model, name, feature, path="/shared/nfs/...")
  │
  ├─ SDK: check scheduler mapping for model
  │    ├─ found → continue
  │    └─ not found → raise ModelNotDeployed(...)
  └─ SDK → Scheduler: POST /v1/predictor/preprocessor/register
       └─ Scheduler: copy from remote path to local storage dir
            shutil.copytree(path, {scheduler_storage}/preprocessors/{name}/)
            then register in preprocessor chain


swarmpilot.predictor.retrain(model)
  │
  ├─ SDK: check scheduler mapping for model
  │    ├─ found → continue
  │    └─ not found → raise ModelNotDeployed(...)
  └─ SDK → Scheduler: POST /v1/predictor/retrain
       └─ Scheduler internal:
            ├─ PredictorLowLevel.train_predictor()
            │    ├─ apply preprocessor chain
            │    ├─ train MLP (ExpectError + Quantile)
            │    └─ save to ModelStorage + invalidate cache
            └─ auto-switch strategy to Probabilistic
                 └─ POST /v1/strategy/set {"strategy": "probabilistic"} (internal)
```

### Strategy Auto-Switch

When MLP training completes (via `retrain()` or `train()`), the Scheduler
automatically switches from Round-Robin to Probabilistic scheduling:

```
Before training:  Round-Robin (no prediction model available)
                         ↓
        retrain() / train() completes successfully
                         ↓
After training:   Probabilistic (uses MLP predictions for routing)
```

This is a one-way transition within a deployment lifecycle. To revert, use:
```python
# SDK does not wrap strategy management — use Scheduler API directly
# POST {scheduler}/v1/strategy/set {"strategy": "round_robin"}
```

### Inference Routing

The Scheduler acts as a transparent proxy: all non-management API calls
(i.e., not `/v1/instance/*`, `/v1/strategy/*`, `/v1/predictor/*`)
are routed to a target instance selected by the current scheduling strategy.

```python
inst = swarmpilot.serve("Qwen/Qwen2.5-7B", gpu=1, replicas=3)
inst.wait_ready()

# User sends inference requests directly to Scheduler
# Scheduler selects instance via Round-Robin or Probabilistic, then proxies
import httpx
resp = httpx.post(f"{inst.scheduler}/v1/chat/completions", json={...})
# Scheduler → selected instance → response back to user
```

### Predictor Data Models

```python
class PreprocessorInfo:
    name: str                    # "llm-output-predictor"
    feature: str                 # "input_text"
    path: str                    # server-side storage path


class ModelStatus:
    model: str
    samples_collected: int
    last_trained: str | None     # ISO 8601
    prediction_types: list[str]  # ["expect_error", "quantile"]
    metrics: dict                # {"mse": 0.03, "mae": 12.5}
    strategy: str                # "round_robin" or "probabilistic"
    preprocessors: list[PreprocessorInfo]


class TrainResult:
    model: str
    samples_trained: int
    metrics: dict
    strategy: str                # strategy after training (always "probabilistic")


class PredictResult:
    model: str
    expected_runtime_ms: float | None
    error_margin_ms: float | None
    quantiles: dict[float, float] | None  # {0.5: 1200, 0.9: 1500, ...}
```

## Legacy Endpoints

Existing endpoints preserved for backward compatibility:

- `POST /v1/deploy` (old optimizer + deploy with capacity matrix)
- `POST /v1/deploy_manually` (old manual target state)
- `POST /v1/optimize` (old optimize-only)
- `POST /v1/scale` (old single-model scale)
- `POST /v1/migrate` (old instance migration)
- `POST /v1/terminate-all` (old terminate all)
- `GET /v1/status` (old status)

These will be documented as legacy. New code should use the endpoints defined above.
