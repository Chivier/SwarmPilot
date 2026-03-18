# Multi-Model Planner Example

Production pattern — Planner coordinates multi-model deployment across per-model Schedulers.

## Overview

This example demonstrates the full SwarmPilot workflow: a central **Planner** manages two per-model **Schedulers**, each handling its own model's instances. The Planner provides:

- **Scheduler discovery** — clients query one endpoint to find all model→scheduler mappings
- **Automated scaling** — scale any model up or down through the Planner API
- **Unified termination** — tear down all instances across all models in one call
- **Optimized deployment** (with PyLet) — automatic instance placement and lifecycle

## Architecture

```
                ┌──────────────────────────┐
                │         Client           │
                │  (sdk_example.py / curl) │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │     Planner  :8002       │
                │  scheduler discovery     │
                │  scaling, termination    │
                └──────┬───────────┬───────┘
                       │           │
              ┌────────┘           └────────┐
              ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  Scheduler A     │          │  Scheduler B     │
    │  :8010           │          │  :8020           │
    │  Qwen/Qwen3-8B  │          │  Llama-3.1-8B    │
    └────┬────────┬────┘          └────┬────────┬────┘
         │        │                    │        │
         ▼        ▼                    ▼        ▼
    ┌────────┐┌────────┐         ┌────────┐┌────────┐
    │Inst    ││Inst    │         │Inst    ││Inst    │
    │:8100   ││:8101   │         │:8200   ││:8201   │
    └────────┘└────────┘         └────────┘└────────┘
```

## Quick Start

```bash
# 1. Start cluster (Planner + 2 Schedulers)
./examples/multi_model_planner/start_cluster.sh

# 2. Deploy instances
#    With PyLet:  automated via splanner serve
#    Without:     mock instances + manual registration
./examples/multi_model_planner/deploy_model.sh

# 3. Query scheduler mapping
curl http://localhost:8002/v1/schedulers

# 4. Check instances per model
curl http://localhost:8010/v1/instance/list   # Qwen
curl http://localhost:8020/v1/instance/list   # Llama

# 5. Stop everything
./examples/multi_model_planner/stop_cluster.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `start_cluster.sh` | Start Planner + 2 Schedulers (auto-registration) |
| `deploy_model.sh` | Deploy instances (PyLet or mock mode) |
| `stop_cluster.sh` | Graceful shutdown of all services |
| `mock_vllm_server.py` | Minimal mock inference server |
| `sdk_example.py` | Python SDK usage (requires PyLet) |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PLANNER_PORT` | 8002 | Planner listen port |
| `SCHEDULER_QWEN_PORT` | 8010 | Qwen scheduler port |
| `SCHEDULER_LLAMA_PORT` | 8020 | Llama scheduler port |
| `DUMMY_HEALTH_PORT` | 9999 | Temporary health server for Planner init |

### Key Environment Variables (Scheduler)

| Variable | Purpose |
|----------|---------|
| `SCHEDULER_MODEL_ID` | Model this scheduler serves (one per process) |
| `PLANNER_REGISTRATION_URL` | Planner URL for auto-registration |
| `SCHEDULER_SELF_URL` | Advertised URL so Planner can reach this scheduler |
| `PREDICTOR_MODE` | Set to `library` for embedded predictor |

## Comparison with Direct Deployment

| Feature | Direct (`multi_model_direct/`) | Planner (this example) |
|---------|-------------------------------|------------------------|
| Scheduler discovery | Manual (know each port) | `GET /v1/schedulers` |
| Instance deployment | Manual curl registration | `splanner serve` or manual |
| Scaling | Start/stop + re-register | `POST /v1/scale` |
| Teardown | Kill each process | `splanner terminate --all` |
| PyLet integration | None | Full lifecycle management |
| Best for | Dev/testing | Production multi-model |

## Troubleshooting

### Schedulers not registered with Planner

Verify registration:
```bash
curl http://localhost:8002/v1/schedulers | python3 -m json.tool
```

Check scheduler logs for registration errors:
```bash
tail -f /tmp/multi_model_planner/scheduler-qwen.log
tail -f /tmp/multi_model_planner/scheduler-llama.log
```

Ensure `PLANNER_REGISTRATION_URL` (not `PLANNER_URL`) is set correctly.

### Port already in use

```bash
./examples/multi_model_planner/stop_cluster.sh
# or manually:
lsof -i:8002 -i:8010 -i:8020
```

### Mock instances not responding

```bash
curl http://localhost:8100/health
curl http://localhost:8200/health
```

Check mock server logs:
```bash
tail -f /tmp/multi_model_planner/mock-qwen-inst-001.log
```

### deploy_model.sh fails in Mode A

Mode A requires a running PyLet cluster. If PyLet is unavailable, the script falls back to Mode B (mock instances) automatically.

## Files

| File | Lines | Description |
|------|-------|-------------|
| `start_cluster.sh` | ~100 | Planner + Scheduler startup with health checks |
| `deploy_model.sh` | ~100 | Dual-mode deployment (PyLet / mock) |
| `stop_cluster.sh` | ~40 | Graceful shutdown |
| `mock_vllm_server.py` | ~65 | Mock vLLM-compatible inference server |
| `sdk_example.py` | ~45 | Async SDK client usage |
| `README.md` | — | This file |

## Related

- Single model: `examples/single_model/` (no Planner, one Scheduler)
- Multi-model direct: `examples/multi_model_direct/` (no Planner, manual Schedulers)
- LLM cluster: `examples/llm_cluster/` (3 models, PyLet, optimizer)
