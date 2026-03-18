# Multi-Model Direct Deployment

Two models, two schedulers, no Planner — manual instance management.

## Overview

This example demonstrates the **one scheduler per model** architecture
rule by running two independent schedulers side by side. Each scheduler
owns exactly one model, and the client must know which scheduler handles
which model.

- **Scheduler A** (:8010) — `Qwen/Qwen3-8B-VL` (2 instances)
- **Scheduler B** (:8020) — `meta-llama/Llama-3.1-8B` (2 instances)

No Planner, no PyLet, no SDK — everything is managed with shell scripts
and direct HTTP calls.

## Architecture

```
                ┌──────────────────────────┐
                │         Client           │
                │    (api_example.py)       │
                └─────┬──────────┬─────────┘
   "Which scheduler?" │          │ "Which scheduler?"
                      ▼          ▼
         ┌──────────────┐  ┌──────────────┐
         │ Scheduler A  │  │ Scheduler B  │
         │   :8010      │  │   :8020      │
         │ Qwen3-8B-VL  │  │ Llama-3.1-8B │
         └──┬───────┬───┘  └──┬───────┬───┘
            │       │         │       │
            ▼       ▼         ▼       ▼
        ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
        │:8100 │ │:8101 │ │:8200 │ │:8201 │
        │Qwen 0│ │Qwen 1│ │Llama0│ │Llama1│
        └──────┘ └──────┘ └──────┘ └──────┘
```

## Quick Start

```bash
# 1. Start both schedulers
./examples/multi_model_direct/start_cluster.sh

# 2. Launch mock instances and register them
./examples/multi_model_direct/deploy_model.sh

# 3. Send requests
python examples/multi_model_direct/api_example.py

# 4. Tear down
./examples/multi_model_direct/stop_cluster.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `start_cluster.sh` | Start Scheduler A (:8010) and Scheduler B (:8020) |
| `deploy_model.sh` | Launch 4 mock instances, wait for health, register with schedulers |
| `stop_cluster.sh` | Kill all instances and schedulers |
| `mock_vllm_server.py` | Minimal FastAPI mock (GET /health, POST /v1/completions) |
| `api_example.py` | httpx client querying both schedulers independently |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SCHEDULER_QWEN_PORT` | 8010 | Port for Scheduler A (Qwen) |
| `SCHEDULER_LLAMA_PORT` | 8020 | Port for Scheduler B (Llama) |

Instance ports are hardcoded: 8100–8101 (Qwen), 8200–8201 (Llama).

Logs and PID files are written to `/tmp/multi_model_direct/`.

## Troubleshooting

### Scheduler not starting

Check logs:

```bash
tail -f /tmp/multi_model_direct/scheduler-qwen.log
tail -f /tmp/multi_model_direct/scheduler-llama.log
```

### Registration fails (HTTP 4xx)

Ensure `model_id` in the registration payload matches the scheduler's
`SCHEDULER_MODEL_ID`. Each scheduler rejects instances for other models.

```bash
curl http://localhost:8010/v1/instance/list | python3 -m json.tool
curl http://localhost:8020/v1/instance/list | python3 -m json.tool
```

### Port already in use

```bash
./examples/multi_model_direct/stop_cluster.sh
# Then retry
```

### Tasks not completing

Verify instances are registered and healthy:

```bash
curl http://localhost:8010/v1/instance/list | python3 -m json.tool
curl http://localhost:8100/health
```

## When to Use the Planner

This example works for small, static deployments. As you add models or
scale instances, the manual approach becomes painful:

| Pain point | This example | With Planner |
|-----------|--------------|--------------|
| Model-to-scheduler mapping | Client hardcodes URLs | Planner discovers schedulers |
| Instance lifecycle | Manual start/register/drain/stop | Planner manages via PyLet |
| Scaling | Edit scripts, re-run | `POST /v1/scale` |
| Optimal placement | You decide replica counts | Optimizer computes allocation |

See `examples/multi_model_planner/` for the Planner-managed version.
