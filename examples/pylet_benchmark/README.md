# PyLet Benchmark Example - Direct Scheduler Registration

This is the **simplest** example of the SwarmPilot system: no planner, direct instance registration with scheduler.

## Overview

This example demonstrates the core SwarmPilot components without orchestration complexity:

```
┌─────────────────┐
│  Mock Predictor │  (provides load prediction)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│    Scheduler     │  (routes tasks, no planner)
└────────┬─────────┘
         │
         ├──▶ Sleep Model A (4 instances)
         │
         ├──▶ Sleep Model B (3 instances)
         │
         └──▶ Sleep Model C (3 instances)
```

**Key characteristics:**
- **No Planner**: Instances register directly with scheduler
- **Direct Registration**: Manual registration via `/v1/instance/register` API
- **Single Scheduler**: Centralized scheduling for all models
- **Sleep Models**: CPU-only test workload that simulates latency
- **QPS-based Workload**: Generate configurable load for testing

## Quick Start

### 1. Start the Services

```bash
./examples/pylet_benchmark/start_cluster.sh
```

This starts:
- Mock Predictor (port 8002)
- Scheduler (port 8000)

No instances are started yet - see step 2.

### 2. Deploy Model Instances

```bash
./examples/pylet_benchmark/deploy_model.sh
```

This deploys:
- sleep_model_a: 4 instances (ports 8100-8103)
- sleep_model_b: 3 instances (ports 8104-8106)
- sleep_model_c: 3 instances (ports 8107-8109)

Each instance registers with the scheduler automatically.

### 3. Generate Workload

```bash
python examples/pylet_benchmark/generate_workload.py --qps 5 --duration 60
```

This submits 300 tasks (5 QPS × 60 seconds) using round-robin model selection.

### 4. View Results

The workload generator displays:
- Overall completion rate and latency stats
- Per-model breakdown (total, completed, failed, min/max/avg latency)
- Actual QPS achieved

### 5. Stop the Cluster

```bash
./examples/pylet_benchmark/stop_cluster.sh
```

This terminates all services and instances.

## Scripts

### `start_cluster.sh`

Starts the core services:
1. **Mock Predictor** - Simulates predictor responses for load prediction
2. **Scheduler** - Routes tasks to instances, no planner registration

Services run with:
- Predictable port allocation (configurable via environment variables)
- Health checks after startup
- Logs in `/tmp/pylet_benchmark/`

**Environment Variables:**
- `PREDICTOR_PORT` - Mock Predictor port (default: 8002)
- `SCHEDULER_PORT` - Scheduler port (default: 8000)

### `deploy_model.sh`

Deploys sleep model instances:
1. Starts instance processes listening on sequential ports (8100+)
2. Waits for each instance to be ready (health check)
3. Registers with scheduler via POST `/v1/instance/register`
4. Saves instance PIDs for cleanup

**Instance Distribution** (configurable in script):
- sleep_model_a: 4 instances
- sleep_model_b: 3 instances
- sleep_model_c: 3 instances

Each instance:
- Runs `pylet_sleep_model.py` (standalone FastAPI server)
- Exposes `/health` and `/inference` endpoints
- No auto-registration (manual via deploy script)

### `stop_cluster.sh`

Stops all services and instances:
1. Terminates scheduler (graceful, then force)
2. Terminates predictor (graceful, then force)
3. Terminates all running instances (via PID files)
4. Cleans up PID files

### `generate_workload.py`

CLI workload generator:
- Submits tasks at specified QPS rate
- Round-robin model selection
- Waits for task completion with polling
- Prints per-model statistics

**Usage:**
```bash
python examples/pylet_benchmark/generate_workload.py \
  --scheduler-url http://localhost:8000 \
  --qps 10 \
  --duration 120 \
  --sleep-time-min 0.1 \
  --sleep-time-max 1.0 \
  --models sleep_model_a,sleep_model_b,sleep_model_c
```

**Options:**
- `--scheduler-url` - Scheduler URL (default: http://localhost:8000)
- `--qps` - Queries per second (default: 5)
- `--duration` - Test duration in seconds (default: 60)
- `--sleep-time-min` - Minimum sleep time (default: 0.1s)
- `--sleep-time-max` - Maximum sleep time (default: 1.0s)
- `--models` - Comma-separated model list (default: sleep_model_a,sleep_model_b,sleep_model_c)

## Instance Registration

When `deploy_model.sh` runs, each instance is registered via:

```bash
POST http://localhost:8000/v1/instance/register
Content-Type: application/json

{
    "instance_id": "sleep_model_a-000",
    "model_id": "sleep_model_a",
    "endpoint": "http://localhost:8100",
    "platform_info": {
        "software_name": "python",
        "software_version": "3.11",
        "hardware_name": "cpu"
    }
}
```

This is the **key difference** from the planner-based examples:
- No central deployment orchestrator
- Instances self-identify with model_id
- Scheduler maintains instance pool
- Direct HTTP registration API

## Log Locations

All logs saved in `/tmp/pylet_benchmark/`:

```
/tmp/pylet_benchmark/
├── predictor.log          # Mock Predictor logs
├── scheduler.log          # Scheduler logs
├── instance_sleep_model_a-000.log  # Instance logs
├── instance_sleep_model_a-001.log
├── ...
├── predictor.pid          # Process IDs for cleanup
├── scheduler.pid
└── instance_*.pid
```

View real-time logs:
```bash
tail -f /tmp/pylet_benchmark/scheduler.log
tail -f /tmp/pylet_benchmark/instance_sleep_model_a-000.log
```

## Customization

### Change Instance Counts

Edit `deploy_model.sh`, modify the `MODEL_DISTRIBUTION` array:

```bash
declare -A MODEL_DISTRIBUTION=(
    [sleep_model_a]=8      # 8 instead of 4
    [sleep_model_b]=4      # 4 instead of 3
    [sleep_model_c]=4      # 4 instead of 3
)
```

### Change Scheduler Port

```bash
SCHEDULER_PORT=9000 ./examples/pylet_benchmark/start_cluster.sh
python examples/pylet_benchmark/generate_workload.py --scheduler-url http://localhost:9000
```

### Different QPS and Duration

```bash
# 20 QPS for 300 seconds = 6000 tasks
python examples/pylet_benchmark/generate_workload.py --qps 20 --duration 300
```

### Larger Sleep Times

```bash
# Simulate longer-running workloads
python examples/pylet_benchmark/generate_workload.py \
  --sleep-time-min 0.5 \
  --sleep-time-max 5.0
```

## Architecture Details

### Sleep Model Endpoints

Each instance provides:

- **POST `/inference`** - Execute sleep task
  ```json
  Request: {"sleep_time": 0.5}
  Response: {"success": true, "result": {...}, "execution_time": 0.5}
  ```

- **GET `/health`** - Health check
  ```json
  Response: {"status": "healthy", "model_loaded": true}
  ```

- **GET `/stats`** - Instance statistics
  ```json
  Response: {"requests_received": 10, "requests_completed": 8, ...}
  ```

- **POST `/task/submit`** - Scheduler task interface (used internally)

### Scheduler Routes

- **POST `/v1/task/submit`** - Submit task for scheduling
- **GET `/v1/task/info`** - Query task status
- **POST `/v1/instance/register`** - Register instance
- **POST `/v1/instance/remove`** - Deregister instance
- **GET `/v1/instance/list`** - List all instances
- **GET `/v1/health`** - Health check

### Task Lifecycle

1. **Submission**: `generate_workload.py` → `/v1/task/submit` → Scheduler
2. **Routing**: Scheduler selects instance based on model_id
3. **Execution**: Instance sleeps for specified duration
4. **Completion**: Instance marks task complete
5. **Polling**: Workload generator polls `/v1/task/info` for status

## Performance Tuning

### Increase Throughput

Increase instance count:
```bash
# Edit deploy_model.sh, change MODEL_DISTRIBUTION
./examples/pylet_benchmark/stop_cluster.sh
./examples/pylet_benchmark/deploy_model.sh
```

### Test Overload Scenarios

Submit more QPS than capacity:
```bash
# 20 instances × ~1 req/s = ~20 QPS capacity
# Submit 50 QPS to test queuing
python examples/pylet_benchmark/generate_workload.py --qps 50 --duration 60
```

### Measure Latency Under Load

Use high QPS but short sleep times:
```bash
python examples/pylet_benchmark/generate_workload.py \
  --qps 100 \
  --duration 30 \
  --sleep-time-min 0.01 \
  --sleep-time-max 0.05
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill and retry
kill -9 <PID>
./examples/pylet_benchmark/start_cluster.sh
```

### Instance Registration Fails

Check scheduler is running:
```bash
curl http://localhost:8000/v1/health
```

Check instance logs:
```bash
cat /tmp/pylet_benchmark/instance_sleep_model_a-000.log
```

### Tasks Not Completing

Verify instances are registered:
```bash
curl http://localhost:8000/v1/instance/list | python -m json.tool
```

Check instance health:
```bash
curl http://localhost:8100/health
```

### High Latency

- Check CPU load: `top`
- Increase sleep time range: `--sleep-time-min 0.5`
- Reduce QPS: `--qps 5`

## Comparison with Multi-Scheduler Example

| Aspect | This Example | Multi-Scheduler |
|--------|-------------|-----------------|
| Architecture | Single Scheduler | Per-model Schedulers |
| Planner | None | PyLet Planner (orchestrator) |
| Instance Deployment | Manual (script) | PyLet deployment |
| Registration | Direct API | Automatic |
| Complexity | Simple | Advanced |
| Scalability | Limited | Multi-model scaling |

This example is ideal for:
- **Learning** the core concepts
- **Testing** scheduler and predictor in isolation
- **Benchmarking** basic performance
- **Development** without planner complexity

For production multi-model deployments, see `examples/mock_llm_cluster/`.

## References

- **Scheduler API**: `/scheduler/src/api.py`
- **Sleep Model**: `/examples/pylet_benchmark/pylet_sleep_model.py`
- **Mock Predictor**: `/examples/pylet_benchmark/mock_predictor_server.py`
- **Architecture**: See [PYLET-025](https://taskmgr.example.com) documentation
