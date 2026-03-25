# SDK Usage

The SwarmPilot Python SDK and CLI for programmatic deployment and management.

## Installation

```bash
pip install swarmpilot
```

## Quick Example

```python
from swarmpilot.sdk import SwarmPilotClient

async with SwarmPilotClient("http://localhost:8002") as sp:
    # Deploy a model
    group = await sp.serve("Qwen/Qwen3-8B-VL-Instruct", gpu=1, replicas=2)
    await group.wait_ready()
    print(group.endpoints)

    # Terminate
    await sp.terminate(all=True)
```

---

## Client Initialization

```python
SwarmPilotClient(
    planner_url: str = "http://localhost:8002",
    scheduler_url: str | None = None,
    timeout: float = 300.0,
)
```

| Parameter | Description |
|-----------|-------------|
| `planner_url` | Base URL of the Planner service |
| `scheduler_url` | Base URL of the Scheduler service (required for predictor operations) |
| `timeout` | Default HTTP timeout in seconds |

The client must be used as an async context manager or explicitly closed via `close()`.

---

## Deployment Operations

### serve() -- Deploy a Model

```python
group = await sp.serve(
    "Qwen/Qwen3-8B-VL-Instruct",
    gpu=1,
    replicas=2,
    scheduler="auto",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_or_command` | `str` | required | Model name or shell command |
| `name` | `str \| None` | `None` | Instance group name |
| `replicas` | `int` | `1` | Number of replicas |
| `gpu` | `int` | `1` | GPUs per replica |
| `scheduler` | `str \| None` | `"auto"` | Scheduler URL, `"auto"`, or `None` |

Returns: `InstanceGroup`

### run() -- Start a Custom Workload

```python
proc = await sp.run(
    "python train.py --epochs 10",
    name="train-job",
    gpu=2,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `command` | `str` | required | Shell command to execute |
| `name` | `str` | required | Unique process name |
| `gpu` | `int` | `0` | GPUs allocated |

Returns: `Process`

### register() + deploy() -- Optimized Multi-Model Deployment

```python
await sp.register("Qwen/Qwen3-8B-VL-Instruct", gpu=1, replicas=3)
await sp.register("meta-llama/Llama-3-8B", gpu=2, replicas=2)
result = await sp.deploy()
print(result.status)
print(result["Qwen/Qwen3-8B-VL-Instruct"].endpoints)
```

`register()` queues model requirements. `deploy()` triggers the optimizer and deploys all registered models.

Returns: `DeploymentResult` (supports `result["model_name"]` dict access)

### scale() -- Scale Replicas

```python
group = await sp.scale("Qwen/Qwen3-8B-VL-Instruct", 5)
```

Returns: `InstanceGroup`

### instances() -- List All Instances

```python
state = await sp.instances()
for inst in state.instances:
    print(f"{inst.name}  {inst.model}  {inst.status}")
```

Returns: `ClusterState`

### schedulers() -- Get Model-to-Scheduler Mapping

```python
mapping = await sp.schedulers()  # {"model_name": "http://scheduler-url"}
```

Returns: `dict[str, str]`

### terminate() -- Terminate Instances

```python
await sp.terminate(name="inst-001")           # by name
await sp.terminate(model="Qwen/Qwen3-8B-VL-Instruct")  # by model
await sp.terminate(all=True)                  # everything
```

At least one of `name`, `model`, or `all` must be set.

---

## Predictor Operations

Requires `scheduler_url` to be set on the client.

```python
async with SwarmPilotClient(
    "http://localhost:8002",
    scheduler_url="http://localhost:8000",
) as sp:
    # Train the predictor
    result = await sp.train("Qwen/Qwen3-8B-VL-Instruct", prediction_type="expect_error")
    print(result.samples_trained, result.metrics)

    # Check predictor status
    status = await sp.predictor_status("Qwen/Qwen3-8B-VL-Instruct")
    print(status.samples_collected, status.prediction_types)

    # Run a prediction
    pred = await sp.predict(
        "Qwen/Qwen3-8B-VL-Instruct",
        features={"token_count": 150},
        prediction_type="expect_error",
    )
    print(pred.expected_runtime_ms, pred.error_margin_ms)
```

See [Predictor](predictor.md) for details on training data requirements and prediction modes.

---

## SDK Data Models

### Instance

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique instance identifier |
| `model` | `str \| None` | Model being served |
| `command` | `str` | Launch command |
| `endpoint` | `str \| None` | HTTP endpoint (once ready) |
| `scheduler` | `str \| None` | Scheduler URL |
| `status` | `str` | Lifecycle status |
| `gpu` | `int` | GPUs allocated |

Methods: `wait_ready(timeout=300)`, `terminate()`

### InstanceGroup

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Group name |
| `model` | `str \| None` | Model served by all replicas |
| `instances` | `list[Instance]` | Individual instances |
| `endpoints` | `list[str]` | Non-None endpoints (property) |

Methods: `wait_ready(timeout=300)`, `scale(replicas)`, `terminate()`

### Process

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Process identifier |
| `command` | `str` | Shell command |
| `endpoint` | `str \| None` | HTTP endpoint |
| `status` | `str` | Lifecycle status |
| `gpu` | `int` | GPUs allocated |

Methods: `terminate()`

### DeploymentResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `plan` | `dict` | Raw optimization plan |
| `groups` | `dict[str, InstanceGroup]` | Model name -> group mapping |
| `status` | `str` | Overall deployment status |

Supports `result["model_name"]` access. Methods: `wait_ready(timeout=600)`

### ClusterState

| Attribute | Type | Description |
|-----------|------|-------------|
| `instances` | `list[Instance]` | All instances |
| `processes` | `list[Process]` | All processes |
| `groups` | `list[InstanceGroup]` | Logical groups |

### TrainResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model identifier |
| `samples_trained` | `int` | Samples used |
| `metrics` | `dict` | Training metrics |
| `strategy` | `str` | Prediction strategy |

### PredictResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model identifier |
| `expected_runtime_ms` | `float \| None` | Point estimate (expect_error) |
| `error_margin_ms` | `float \| None` | Error margin (expect_error) |
| `quantiles` | `dict \| None` | Quantile values (quantile mode) |

### ModelStatus

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model identifier |
| `samples_collected` | `int` | Samples collected |
| `last_trained` | `str \| None` | Last training timestamp |
| `prediction_types` | `list[str]` | Available prediction types |
| `metrics` | `dict` | Training metrics |
| `strategy` | `str` | Active prediction strategy |
| `preprocessors` | `list[PreprocessorInfo]` | Registered preprocessors |

---

## CLI Reference

The `splanner` CLI provides the same deployment operations as the SDK. All commands accept `--planner-url` (or `PLANNER_URL` env var).

| Command | SDK Equivalent | Description |
|---------|---------------|-------------|
| `splanner serve <model> --gpu N --replicas N` | `sp.serve()` | Deploy a model |
| `splanner run "<command>" --name N --gpu N` | `sp.run()` | Start custom workload |
| `splanner register <model> --gpu N --replicas N` | `sp.register()` | Register model requirements |
| `splanner deploy` | `sp.deploy()` | Trigger optimized deployment |
| `splanner ps` | `sp.instances()` | List all instances |
| `splanner scale <model> --replicas N` | `sp.scale()` | Scale replicas |
| `splanner terminate [name] --model M --all` | `sp.terminate()` | Terminate instances |
| `splanner schedulers` | `sp.schedulers()` | Show scheduler mapping |

---

## Error Handling

The SDK raises structured exceptions from `swarmpilot.errors`:

| Exception | Trigger |
|-----------|---------|
| `ModelNotDeployed` | 404 from Planner (model not found) |
| `SchedulerNotFound` | 404 from Scheduler (no scheduler for model) |
| `DeployError` | Instance enters `"failed"` status |
| `SwarmPilotTimeoutError` | Instance not ready within timeout |
| `ValueError` | 400 (bad request) or missing `scheduler_url` for predictor ops |
