# Single Model Example

Single-model deployment with 3 replicas, 1 Scheduler, no Planner.

## Overview

The simplest SwarmPilot deployment pattern:

- **1 Scheduler** serves one model (`Qwen/Qwen3-8B-VL`)
- **3 mock instances** on ports 8100-8102, registered manually
- **No Planner** — instances are started and registered via shell scripts
- **Library Predictor** — Scheduler uses built-in predictor (`PREDICTOR_MODE=library`)

## Architecture

```
         ┌──────────┐
         │  Client   │
         │ (httpx)   │
         └─────┬─────┘
               │ POST /v1/task/submit
               ▼
         ┌──────────┐
         │Scheduler │
         │  :8000   │
         └─────┬────┘
               │ dispatches to
     ┌─────────┼─────────┐
     ▼         ▼         ▼
┌─────────┐┌─────────┐┌─────────┐
│Instance ││Instance ││Instance │
│ :8100   ││ :8101   ││ :8102   │
│qwen-vl-0││qwen-vl-1││qwen-vl-2│
└─────────┘└─────────┘└─────────┘
```

## Quick Start

### 1. Start Scheduler

```bash
./examples/1.single_model/start_cluster.sh
```

### 2. Deploy 3 Mock Instances

```bash
./examples/1.single_model/deploy_model.sh
```

### 3. Try the API

```bash
python examples/1.single_model/api_example.py
```

### 4. Stop Everything

```bash
./examples/1.single_model/stop_cluster.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `start_cluster.sh` | Start Scheduler on port 8000 |
| `deploy_model.sh` | Start 3 mock instances + register with Scheduler |
| `stop_cluster.sh` | Kill all processes via PID files |
| `mock_vllm_server.py` | Minimal FastAPI mock (GET /health, POST /v1/completions) |
| `api_example.py` | httpx demo: list instances, submit task, check status |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_PORT` | `8000` | Scheduler listen port |
| `MODEL_ID` | `Qwen/Qwen3-8B-VL` | Model identifier |
| `PORT` | `8100` | Mock server listen port |

### Ports

| Port | Service |
|------|---------|
| 8000 | Scheduler |
| 8100 | Instance 0 (qwen-vl-0) |
| 8101 | Instance 1 (qwen-vl-1) |
| 8102 | Instance 2 (qwen-vl-2) |

### Logs & PID Files

All stored in `/tmp/1.single_model/`:
- `scheduler.log`, `scheduler.pid`
- `instance-{0,1,2}.log`, `instance-{0,1,2}.pid`

## Using Real vLLM

Replace mock instances with real vLLM servers by editing `deploy_model.sh`.
Uncomment the `vllm serve` block and comment out the mock block:

```bash
for i in 0 1 2; do
    vllm serve "Qwen/Qwen3-8B-VL" \
        --port "${INSTANCE_PORTS[$i]}" --host 0.0.0.0 \
        --gpu-memory-utilization 0.9 \
        > "$LOG_DIR/instance-$i.log" 2>&1 &
done
```

Each instance needs a GPU. Adjust `--gpu-memory-utilization` for your hardware.

## Troubleshooting

### Scheduler not starting

```bash
tail -f /tmp/1.single_model/scheduler.log
```

Check if port 8000 is already in use:
```bash
lsof -i:8000
```

### Instances fail health check

```bash
tail -f /tmp/1.single_model/instance-0.log
```

Verify manually:
```bash
curl http://localhost:8100/health
```

### Tasks not completing

Verify instances are registered:
```bash
curl http://localhost:8000/v1/instance/list | python3 -m json.tool
```

Check Scheduler strategy:
```bash
curl http://localhost:8000/v1/strategy/current
```

### Port conflicts

Stop existing services first:
```bash
./examples/1.single_model/stop_cluster.sh
```

## Related Examples

- [`3.multi_model_direct/`](../3.multi_model_direct/) — Multiple models, one Scheduler per model, no Planner
- [`5.multi_model_planner/`](../5.multi_model_planner/) — Multiple models, Planner-managed deployment
- [`llm_cluster/`](../llm_cluster/) — Full LLM cluster with PyLet orchestration
