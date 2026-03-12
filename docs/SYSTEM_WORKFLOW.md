# System Workflow

End-to-end workflow documentation for SwarmPilot: cluster startup, service
deployment, request routing, and scheduling execution.

For API details see [API_REFERENCE.md](API_REFERENCE.md). For architecture
overview see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Cluster Startup

### 1.1 Standard Cluster Topology

A production SwarmPilot cluster consists of three service tiers:

```
                    ┌──────────────────┐
                    │     Clients      │
                    └───────┬──────────┘
                            │ HTTP / WebSocket
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │ Scheduler  │ │ Scheduler  │ │ Scheduler  │
       │ (llm-7b)   │ │ (llm-32b)  │ │ (gpt-4)    │
       │ :8010      │ │ :8020      │ │ :8030      │
       └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
              │              │              │
              │   ┌──────────┴──────────┐   │
              │   │                     │   │
              ▼   ▼                     ▼   ▼
       ┌────────────┐           ┌────────────┐
       │ Predictor  │           │  Planner   │
       │ :8001      │           │  :8002     │
       │ (embedded) │           │            │
       └────────────┘           └─────┬──────┘
                                      │
                                      ▼
                               ┌────────────┐
                               │   PyLet    │
                               │  Cluster   │
                               │  :5100     │
                               └────────────┘
```

**Single-scheduler deployment** (minimal): one Predictor + one Scheduler.
No Planner needed if instances are registered manually.

**Multi-scheduler deployment** (production): one Predictor (shared) +
N Schedulers (one per model) + one Planner (central optimizer + PyLet).

### 1.2 Startup Sequence

Services **must** start in this order:

```
Step 1: Predictor (:8001)
  │  Loads ML models, exposes /predict, /train, /ws/predict
  │
Step 2: Planner (:8002)
  │  Initializes PyLet (if PYLET_ENABLED=true)
  │  Creates scheduler registry
  │  Ready to accept scheduler registrations
  │
Step 3+: Scheduler(s) (:8010, :8020, ...)
     Each scheduler:
       1. Loads Config (env vars)
       2. Initializes InstanceRegistry, TaskRegistry, WorkerQueueManager
       3. Creates embedded PredictorClient (library mode, no HTTP)
       4. Loads scheduling strategy (default: adaptive_bootstrap)
       5. Registers with Planner via POST /v1/scheduler/register
          - Sends: {model_id, scheduler_url}
          - Fail-hard: exits after 3 failed retries (5s delay each)
       6. Ready for tasks
```

### 1.3 Startup Environment Variables

**Predictor:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTOR_HOST` | `0.0.0.0` | Bind address |
| `PREDICTOR_PORT` | `8000` | Listen port |
| `PREDICTOR_STORAGE_DIR` | `models` | On-disk model directory |
| `PREDICTOR_LOG_LEVEL` | `info` | Log verbosity |

**Planner:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PLANNER_HOST` | `0.0.0.0` | Bind address |
| `PLANNER_PORT` | `8002` | Listen port |
| `PYLET_ENABLED` | `false` | Enable PyLet integration |
| `PYLET_HEAD_URL` | — | PyLet cluster head URL |
| `PYLET_BACKEND` | `vllm` | Model serving backend (`vllm` or `sglang`) |
| `SCHEDULER_URL` | — | Default scheduler URL (optional) |

**Scheduler:**

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_HOST` | `0.0.0.0` | Bind address |
| `SCHEDULER_PORT` | `8000` | Listen port |
| `SCHEDULER_MODEL_ID` | — | Model this scheduler manages |
| `SCHEDULER_SELF_URL` | — | URL reachable by Planner |
| `SCHEDULING_STRATEGY` | `adaptive_bootstrap` | Algorithm name |
| `PLANNER_REGISTRATION_URL` | — | Planner URL for registration |
| `PLANNER_URL` | — | Planner URL for metrics reporting |
| `PREDICTOR_STORAGE_DIR` | `models` | Predictor model directory |
| `TRAINING_ENABLE_AUTO` | `false` | Auto-train predictor models |

### 1.4 Health Checks

```bash
# Predictor
curl http://localhost:8001/health
# → {"status": "ok", "timestamp": "..."}

# Scheduler
curl http://localhost:8010/v1/health
# → {"status": "ok", "instances": 5, "tasks": 12, ...}

# Planner
curl http://localhost:8002/v1/health
# → {"status": "ok", "version": "1.0.0"}

# Verify all schedulers registered
curl http://localhost:8002/v1/scheduler/list
# → {"total": 2, "schedulers": [{"model_id": "llm-7b", ...}, ...]}
```

---

## 2. Service / LLM Deployment

### 2.1 Deployment Methods

SwarmPilot supports three deployment methods, all through the Planner API:

| Method | Endpoint | Use Case |
|--------|----------|----------|
| Optimization-driven | `POST /v1/deploy` | Compute optimal model distribution across machines |
| Manual target state | `POST /v1/deploy_manually` | Specify exact instance counts per model |
| Single-model scale | `POST /v1/scale` | Scale one model to a target replica count |

### 2.2 Optimization-Driven Deployment

```
POST http://planner:8002/v1/deploy
{
  "M": 10,                                    # Number of machines
  "N": 3,                                     # Number of models
  "B": [[cap_matrix]],                        # Capacity matrix [M x N]
  "initial": [-1, -1, ...],                   # Current state (-1 = idle)
  "target": [0.3, 0.5, 0.2],                  # Request distribution
  "a": 0.5,                                   # Change constraint (max 50%)
  "algorithm": "integer_programming",          # or "simulated_annealing"
  "model_ids": ["llm-7b", "llm-32b", "gpt2"]
}
```

The Planner runs `SwarmOptimizer` (integer programming via PuLP, or simulated
annealing) to compute an optimal allocation: `[0, 1, 2, 0, ...]` mapping each
machine to a model index.

### 2.3 Manual Deployment

```
POST http://planner:8002/v1/deploy_manually
{
  "target_state": {"Qwen/7B": 3, "Llama/7B": 2}
}
```

Skips optimization. The Planner directly reconciles the current cluster state
toward the specified target.

### 2.4 Instance Lifecycle

```
DEPLOYING ──→ WAITING_HEALTH ──→ REGISTERING ──→ ACTIVE
    │               │                  │            │
    └→ FAILED       └→ FAILED         └→ FAILED    │
                                                    ▼
                                               DRAINING ──→ TERMINATING ──→ TERMINATED
```

**Full deployment flow:**

```
[1] Planner calls PyLetClient.deploy_model(model_id, count)
      │
      ▼
[2] PyLet submits instances on cluster workers
    Command: "vllm serve {model_id} --port $PORT --host 0.0.0.0"
    or:      "python -m sglang.launch_server --model {model_id} --port $PORT"
      │
      ▼
[3] Planner polls PyLet until instance status = RUNNING
      │
      ▼
[4] Planner polls instance /health endpoint (every 2s, timeout 300s)
    Waits for HTTP 200
      │
      ▼
[5] Planner registers instance with the appropriate Scheduler
    POST /v1/instance/register
    {
      "instance_id": "unique-id",
      "model_id": "Qwen/7B",
      "endpoint": "http://worker1:8080",
      "platform_info": {
        "software_name": "vllm",
        "software_version": "0.4.0",
        "hardware_name": "gpu-node-1"
      }
    }
      │
      ▼
[6] Scheduler adds instance to InstanceRegistry
    Creates dedicated WorkerQueueThread (OS thread)
    Instance status: ACTIVE — ready to serve tasks
```

**Drain flow** (graceful removal):

1. Scheduler marks instance as `DRAINING` — no new tasks assigned
2. Existing queued tasks complete or get rescheduled to other instances
3. Scheduler deregisters instance from `InstanceRegistry`
4. Planner terminates instance via `pylet.cancel_instance()`
5. Instance status: `TERMINATED`

---

## 2.5 SDK & CLI Deployment Flow

In addition to the direct REST API (Section 2), the SDK and CLI
provide higher-level deployment primitives through new endpoints on
the Planner (`/v1/serve`, `/v1/run`, `/v1/scale`, `/v1/terminate`,
`/v1/instances`, `/v1/schedulers`, `/v1/register`, `/v1/deploy`)
and the Scheduler (`/v1/predictor/*`).

### 2.5.1 SDK Flow

```
SwarmPilotClient(planner_url)
    │
    ├─ serve(model, gpu, replicas, scheduler)
    │    → POST /v1/serve
    │    → Planner: resolve scheduler, deploy via PyLet
    │    → Returns: InstanceGroup (name, endpoints, instances)
    │
    ├─ run(command, name, gpu)
    │    → POST /v1/run
    │    → Returns: Process (name, endpoint, status)
    │
    ├─ register(model, gpu, replicas) + deploy()
    │    → POST /v1/register (stores requirements)
    │    → POST /v1/deploy (batch-deploys all registered models)
    │    → Returns: DeploymentResult (plan, groups, status)
    │
    ├─ scale(model, replicas)
    │    → POST /v1/scale
    │    → Returns: InstanceGroup
    │
    ├─ instances()
    │    → GET /v1/instances
    │    → Returns: ClusterState (instances, processes, groups)
    │
    ├─ schedulers()
    │    → GET /v1/schedulers
    │    → Returns: dict[model_id → scheduler_url]
    │
    ├─ terminate(name=, model=, all=)
    │    → POST /v1/terminate
    │
    └─ Predictor operations (require scheduler_url):
         ├─ train(model)         → POST /v1/predictor/train
         ├─ predict(model, ...)  → POST /v1/predictor/predict
         └─ predictor_status(model)
                                 → GET /v1/predictor/status/{model}
```

### 2.5.2 CLI Flow

The `splanner` CLI mirrors the SDK methods as subcommands:

| Command | SDK Method | Endpoint |
|---------|-----------|----------|
| `splanner serve MODEL --gpu N --replicas N` | `serve()` | `POST /v1/serve` |
| `splanner run CMD --name N --gpu N` | `run()` | `POST /v1/run` |
| `splanner register MODEL --gpu N --replicas N` | `register()` | `POST /v1/register` |
| `splanner deploy` | `deploy()` | `POST /v1/deploy` |
| `splanner ps` | `instances()` | `GET /v1/instances` |
| `splanner scale MODEL --replicas N` | `scale()` | `POST /v1/scale` |
| `splanner terminate [NAME] --model M --all` | `terminate()` | `POST /v1/terminate` |
| `splanner schedulers` | `schedulers()` | `GET /v1/schedulers` |

### 2.5.3 Predictor Endpoints

The Scheduler exposes predictor management under `/v1/predictor/`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/predictor/train` | POST | Flush training buffer and train model |
| `/v1/predictor/retrain` | POST | Force retrain (same as train) |
| `/v1/predictor/predict` | POST | Manual single-platform prediction |
| `/v1/predictor/status/{model_id}` | GET | Training status and sample count |
| `/v1/predictor/models` | GET | List all models with predictor data |

After successful training with >= 10 samples, the Scheduler
auto-switches its scheduling strategy to `probabilistic`.

---

## 3. Request Routing

### 3.1 Entry Points

The Scheduler exposes two request entry paths:

**Path A — Explicit task submission** (`POST /v1/task/submit`):

```
POST /v1/task/submit
{
  "task_id": "task-123",
  "model_id": "llm-7b",
  "task_input": {"prompt": "Hello", "max_tokens": 100},
  "metadata": {"priority": 1}
}
```

- Returns **immediately** with PENDING status (non-blocking)
- Client retrieves results via:
  - WebSocket: `ws://scheduler:8000/v1/task/get_result` (real-time push)
  - Polling: `GET /v1/task/info?task_id=task-123`

**Path B — Transparent proxy** (`/{path:path}` catch-all):

```
POST /v1/chat/completions
{"model": "llm-7b", "messages": [...]}
```

- Catches **any** HTTP request not matched by internal routes
- **Synchronously blocks** until backend responds (default timeout: 300s)
- Transparently forwards HTTP status, body, and headers from the backend
- Compatible with OpenAI API format for drop-in usage

### 3.2 Routing Decision Flow

Both entry paths follow the same core routing logic:

```
Request arrives
    │
    ▼
[1] Check InstanceRegistry for available instances
    ├─ Task submit: filters by model_id
    └─ Proxy: uses all active instances
    │
    ▼
[2] Call scheduling_strategy.schedule_task()
    │  ├─ Get ML predictions from PredictorClient
    │  ├─ Collect queue info from InstanceRegistry
    │  └─ Apply algorithm to select best instance
    │
    ▼
[3] Create QueuedTask
    {task_id, model_id, task_input, metadata, enqueue_time, predicted_time_ms}
    │
    ▼
[4] Enqueue to WorkerQueueManager[selected_instance_id]
    │
    ▼
[5a] Task submit: return PENDING, notify via WebSocket on completion
[5b] Proxy: await asyncio.Future until worker completes, return response
```

### 3.3 Route Priority

Internal routes take priority over the proxy catch-all. The proxy is mounted
**last** on the FastAPI app:

1. `/v1/instance/*` — instance management
2. `/v1/task/*` — task management
3. `/v1/strategy/*` — strategy configuration
4. `/v1/callback/*` — worker result callbacks
5. `/v1/health` — health check
6. `/{path:path}` — transparent proxy (catch-all, lowest priority)

---

## 4. Scheduling Execution

### 4.1 End-to-End Workflow

```
Submit ──→ Predict ──→ Select Instance ──→ Update Queue ──→ Dispatch
                                                               │
  Training ←── Callback ←── Result ←── Worker Execution ←──────┘
```

### 4.2 Prediction Phase

The Scheduler embeds the Predictor as a library by default (direct Python
calls, zero network overhead). For each available instance, it calls
`predictor_client.predict()`.

**Two prediction models:**

| Model | Output | Used By |
|-------|--------|---------|
| **ExpectError** | `(expected_runtime_ms, error_margin_ms)` | `min_time`, `serverless` |
| **Quantile** | `{p50, p90, p95, p99}` runtime quantiles | `probabilistic`, `adaptive_bootstrap` |

**Prediction pipeline:**

```
Task metadata + platform_info
    │
    ▼
PreprocessorChain (config-driven, model_id pattern matching)
    │
    ▼
Feature vector extraction
    │
    ▼
MLP forward pass (PyTorch)
    │
    ▼
Prediction(predicted_time_ms, error_margin_ms, quantiles, confidence)
```

The preprocessor chain is configured per model_id pattern via JSON config.
The same chain is used at both training and prediction time for consistency.

### 4.3 Scheduling Strategies

Seven strategies are available, selectable at runtime via
`POST /v1/strategy/set` or the `SCHEDULING_STRATEGY` env var:

#### adaptive_bootstrap (default)

```
if all platforms have trained quantile models:
    delegate to Probabilistic strategy
else:
    use thread-safe round-robin (no prediction needed)
```

Auto-transitions from round-robin (cold start) to probabilistic (warm) as
models train. This is the recommended strategy for most deployments.

#### min_time

```
for each instance i:
    score[i] = queue_expected_time + queue_error_margin + predicted_time
select instance with minimum score
```

Deterministic, greedy selection. Uses ExpectError predictions. Good baseline
when prediction accuracy is high.

#### probabilistic

```
for each of 10 Monte Carlo samples:
    for each instance i:
        sample random percentile p ∈ [0, 1]
        pred_time = interpolate(prediction_quantiles, p)
        queue_time = interpolate(queue_quantiles, p)
        total[i] = pred_time + queue_time
    winner = argmin(total)
select instance with most wins across all samples
```

Captures full probability distributions. Accounts for correlated variability
in both queue length and task runtime. Uses Quantile predictions.

#### round_robin

Cyclic assignment across instances. Ignores predictions entirely. Stateless.

#### random

Uniform random instance selection. Ignores predictions. Stateless.

#### po2 (power of two choices)

```
pick 2 random instances
compare: queue_expected_time + predicted_time
return the better one
```

Low-overhead approximation of min_time. One comparison instead of scanning
all instances.

#### serverless

Similar to min_time but optimized for serverless scaling semantics.

### 4.4 Queue State Update

After selecting an instance, the strategy updates the instance's queue model
to reflect the newly added task:

**ExpectError queue:**
```
new_expected = current_expected + prediction.expected_runtime_ms
new_error    = sqrt(current_error² + prediction.error_margin_ms²)
```
Error margins compound using root-sum-of-squares — uncertainty grows
sub-linearly as queue depth increases.

**Probabilistic queue:**
```
for each quantile level q:
    new_value[q] = current_value[q] + prediction.quantile_value[q]
```
Maintains the full distribution of queue completion times.

**Stateless strategies** (round_robin, random): no queue update.

### 4.5 Task Dispatch

Each registered instance has a dedicated `WorkerQueueThread` (OS thread):

```
WorkerQueueThread._process_loop()
    │
    ▼
[1] Dequeue next task from FIFO queue (blocking, 1s timeout)
    │
    ▼
[2] Callback: mark task RUNNING in TaskRegistry
    │
    ▼
[3] HTTP request to worker endpoint
    POST {worker_endpoint}/{path} with task_input as JSON
    │
    ├─ On success: capture response body, status code, headers
    ├─ On ConnectError/ReadError: retry with exponential backoff
    │   (1s, 2s, 4s — max 3 retries)
    └─ On HTTPStatusError/Timeout: fail immediately (no retry)
    │
    ▼
[4] Create TaskResult
    {task_id, worker_id, status, result, error,
     execution_time_ms, http_status_code, response_headers}
    │
    ▼
[5] Thread-safe callback via asyncio.run_coroutine_threadsafe()
    → Bridges OS thread back to main event loop
```

### 4.6 Result Handling

The `TaskResultCallback` runs asynchronously on the main event loop:

```
handle_result(TaskResult)
    │
    ▼
[1] Update TaskRegistry
    ├─ COMPLETED: store result, set completed_at timestamp
    └─ FAILED: store error message
    │
    ▼
[2] Update InstanceRegistry statistics
    ├─ Decrement pending_tasks
    ├─ Increment completed_tasks (on success)
    └─ Increment failed_tasks (on failure)
    │
    ▼
[3] WebSocket broadcast
    Send WSTaskResultMessage to all subscribed clients
    │
    ▼
[4] Resolve proxy Future (if task came via transparent proxy)
    Proxy handler wakes up and returns response to client
    │
    ▼
[5] Record throughput for Planner reporting
    │
    ▼
[6] Add training sample (if auto-training enabled)
    {model_id, platform_info, features, actual_runtime_ms}
```

### 4.7 Continuous Training Loop

When `TRAINING_ENABLE_AUTO=true`, the Scheduler collects execution results
and retrains predictor models:

```
Collect TrainingSample with actual_runtime_ms
    │
    ▼
Group by (model_id, platform_info)
    │
    ▼
When batch_size samples accumulated (min_samples ≥ 10):
    │
    ▼
Apply PreprocessorChain (same config as prediction time)
    │
    ▼
Train both models:
  - ExpectError MLP: MSE loss, error margin from residuals
  - Quantile MLP: pinball loss, softplus delta (monotonic quantiles)
    │
    ▼
Save to ModelStorage (JSON on disk)
    │
    ▼
Load into cache — next prediction uses updated model
```

This creates a feedback loop: predictions improve as more tasks execute,
leading to better scheduling decisions over time.

---

## 5. Complete Example: End-to-End Request

```bash
# 1. Client submits a task
curl -X POST http://scheduler:8010/v1/task/submit \
  -d '{
    "task_id": "task-42",
    "model_id": "llm-7b",
    "task_input": {"prompt": "Explain quantum computing", "max_tokens": 200},
    "metadata": {"token_count": 150}
  }'
# → {"task_id": "task-42", "status": "PENDING", "queue_position": 3}
```

**Inside the Scheduler:**

```
1. PREDICT: PredictorClient.predict() for each of 3 available instances
   - Instance i1: predicted_time=45ms, error_margin=5ms
   - Instance i2: predicted_time=52ms, error_margin=8ms
   - Instance i3: predicted_time=41ms, error_margin=4ms

2. QUEUE STATE:
   - i1 queue: expected=50ms, error=3ms
   - i2 queue: expected=10ms, error=2ms
   - i3 queue: expected=80ms, error=6ms

3. SELECT (min_time strategy):
   - i1 total: 50 + 3 + 45 = 98ms
   - i2 total: 10 + 2 + 52 = 64ms  ← winner
   - i3 total: 80 + 6 + 41 = 127ms

4. UPDATE QUEUE (i2):
   - new_expected: 10 + 52 = 62ms
   - new_error: sqrt(2² + 8²) = 8.2ms

5. DISPATCH: enqueue to WorkerQueueThread[i2]

6. EXECUTE: POST http://instance-i2:8080/v1/completions
   {"prompt": "Explain quantum computing", "max_tokens": 200}
   → 200 OK, execution_time=48ms

7. CALLBACK:
   - TaskRegistry: task-42 → COMPLETED, result stored
   - InstanceRegistry: i2.pending--, i2.completed++
   - WebSocket: broadcast to subscribed clients
   - Training: add sample (features, actual=48ms) to buffer
```

```bash
# 2. Client retrieves result
curl http://scheduler:8010/v1/task/info?task_id=task-42
# → {"task_id": "task-42", "status": "COMPLETED", "result": {...}}
```

---

## 6. Error Handling Summary

| Stage | Error | Behavior |
|-------|-------|----------|
| No instances (task submit) | Warning logged | Task stays PENDING, queued for later |
| No instances (proxy) | 503 | `{"error": "No backend instances available"}` |
| Scheduling failure | Fallback | Graceful degradation to first available instance |
| Worker connection error | Retry | Exponential backoff (1s, 2s, 4s), max 3 retries |
| Worker HTTP 4xx/5xx | Fail | Propagated transparently (no retry) |
| Worker timeout | Fail | 504 Gateway Timeout (proxy) or FAILED (task submit) |
| Planner registration failure | Fatal | Scheduler exits after 3 retries |
| Instance health check timeout | Fail | Instance stays in WAITING_HEALTH, deployment marked failed |
