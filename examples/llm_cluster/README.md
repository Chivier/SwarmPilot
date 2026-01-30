# LLM Cluster Example

Multi-scheduler setup with 3 LLM models using PyLet for container orchestration.

## Overview

This example demonstrates a complete end-to-end LLM inference system with:
- **3 Per-Model Schedulers**: One scheduler per model (llm_fast, llm_medium, llm_slow)
- **Planner Coordination**: Central planner manages scheduler registration and PyLet deployment
- **Library Predictor**: Schedulers use `PREDICTOR_MODE=library` (no external predictor service)
- **Optimizer-based Deployment**: Planner computes optimal instance distribution

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    Client                       │
│         (generate_workload.py)                  │
└──────────────────┬──────────────────────────────┘
                   │ Discovers schedulers via
                   │ Planner /v1/scheduler/list
                   ▼
┌─────────────────────────────────────────────────┐
│              Planner (:8003)                     │
│  - Scheduler registration                       │
│  - PyLet instance deployment                    │
│  - Optimizer for instance allocation            │
└───────┬──────────────┬──────────────┬───────────┘
        │              │              │
        ▼              ▼              ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │ Scheduler │  │ Scheduler │  │ Scheduler │
  │ llm_fast  │  │ llm_medium│  │ llm_slow  │
  │   :8010   │  │   :8011   │  │   :8012   │
  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
        │              │              │
        ▼              ▼              ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │   PyLet   │  │   PyLet   │  │   PyLet   │
  │ Instances │  │ Instances │  │ Instances │
  │ (fast)    │  │ (medium)  │  │ (slow)    │
  └───────────┘  └───────────┘  └───────────┘
```

## Model Configuration

| Model | Runtime Ratio | QPS Ratio | Latency Distribution | Mean Latency |
|-------|---------------|-----------|----------------------|--------------|
| llm_fast | 1x | 5 (55.6%) | Exponential | ~100ms |
| llm_medium | 5x | 1 (11.1%) | Log-normal | ~500ms |
| llm_slow | 20x | 3 (33.3%) | Gamma | ~2000ms |

Capacity matrix B (inverse of runtime ratios): [20.0, 4.0, 1.0] per worker
Target distribution: [55.56%, 11.11%, 33.33%]

## Quick Start

### Prerequisites

- PyLet cluster running (32 workers, 1 GPU each)
- Python 3.10+
- uv package manager
- Ports available: 8003 (planner), 8010-8012 (schedulers)

### 1. Start PyLet Cluster

```bash
./scripts/start_pylet_test_cluster.sh
```

### 2. Start Services

```bash
./examples/llm_cluster/start_cluster.sh
```

This starts (4 steps):
1. Planner (port 8003, PyLet-enabled)
2. Scheduler for llm_fast (port 8010)
3. Scheduler for llm_medium (port 8011)
4. Scheduler for llm_slow (port 8012)

No external predictor is started — schedulers use `PREDICTOR_MODE=library`.

### 3. Deploy Models

```bash
./examples/llm_cluster/deploy_model.sh [num_instances]
```

Default: 32 instances distributed optimally across models.

The optimizer computes:
- Target distribution: [55.56%, 11.11%, 33.33%]
- Optimal placement: instance assignments minimizing relative error

### 4. Generate Workload

```bash
python examples/llm_cluster/generate_workload.py \
    --total-qps 10.0 \
    --duration 60.0 \
    --wait-timeout 600.0
```

### 5. Stop Services

```bash
./examples/llm_cluster/stop_cluster.sh
```

## Scripts

### generate_workload.py

Standalone CLI script for generating LLM inference workload.

**Features:**
- Discovers per-model schedulers via planner
- QPS ratio support (5:1:3)
- Per-model statistics
- Task completion polling with timeout
- Rich progress display
- OpenAI-compatible task format

**Options:**
```
--planner-url        Planner URL for scheduler discovery (default: http://localhost:8003)
--total-qps          Target QPS (default: 10.0)
--duration           Test duration in seconds (default: 60.0)
--wait-timeout       Completion timeout (default: 600.0)
```

**Output:**
- Overall results (total, completed, failed, QPS)
- Per-model statistics (latencies, execution times)
- Traffic analysis

### start_cluster.sh

Multi-step service startup with health checks.

**Steps:**
1. Planner startup (port 8003) with PyLet integration
2. Scheduler (llm_fast) startup (port 8010)
3. Scheduler (llm_medium) startup (port 8011)
4. Scheduler (llm_slow) startup (port 8012)

Each scheduler self-registers with the planner on startup.

**Configuration (environment variables):**
- `SCHEDULER_FAST_PORT`: Scheduler for llm_fast (default: 8010)
- `SCHEDULER_MEDIUM_PORT`: Scheduler for llm_medium (default: 8011)
- `SCHEDULER_SLOW_PORT`: Scheduler for llm_slow (default: 8012)
- `PLANNER_PORT`: Planner port (default: 8003)
- `PYLET_HEAD_PORT`: PyLet head port (default: 5100)

### stop_cluster.sh

Graceful service shutdown.

**Steps:**
1. Terminate PyLet instances via planner `/v1/terminate-all`
2. Stop Scheduler (llm_fast)
3. Stop Scheduler (llm_medium)
4. Stop Scheduler (llm_slow)
5. Stop Planner

All services write logs to `/tmp/llm_cluster/`.

### deploy_model.sh

Deploy models using planner optimizer.

**Configuration:**
- Capacity matrix: inverse of runtime ratios [20.0, 4.0, 1.0]
- Target distribution: QPS ratios [5:1:3] normalized
- Algorithm: simulated_annealing
- Objective: minimize relative_error

**Output:**
- Optimization score
- Instances per model
- Expected capacity per model

## Troubleshooting

### PyLet workers not connecting

Check logs: `tail -f /tmp/llm_cluster/planner.log`

Verify cluster health:
```bash
curl http://localhost:5100/workers | python3 -m json.tool
```

### Scheduler not healthy

Check per-model scheduler logs:
```bash
tail -f /tmp/llm_cluster/scheduler-fast.log
tail -f /tmp/llm_cluster/scheduler-medium.log
tail -f /tmp/llm_cluster/scheduler-slow.log
```

Verify predictor mode is "library":
```bash
grep "PREDICTOR_MODE" /tmp/llm_cluster/scheduler-fast.log
```

### Planner deployment fails

Verify schedulers are registered:
```bash
curl http://localhost:8003/v1/scheduler/list | python3 -m json.tool
```

Check planner logs for PyLet connection issues:
```bash
tail -f /tmp/llm_cluster/planner.log
```

### Tasks not completing

Verify instances are deployed in each scheduler:
```bash
curl http://localhost:8010/v1/instance/list | python3 -m json.tool
curl http://localhost:8011/v1/instance/list | python3 -m json.tool
curl http://localhost:8012/v1/instance/list | python3 -m json.tool
```

Check instance logs in `/tmp/llm_cluster/` for errors.

## Performance Characteristics

Expected on 32-worker cluster:

| Metric | Value |
|--------|-------|
| Throughput (10 QPS) | ~10 tasks/sec |
| Latency (p50) | ~150ms (submit + exec) |
| Model utilization | ~60-70% of capacity |
| Success rate | >95% |

Actual performance varies with:
- System load
- PyLet worker availability
- Network latency
- Model distribution accuracy

## Files

- `generate_workload.py` - Workload generator CLI
- `start_cluster.sh` - Service startup script
- `stop_cluster.sh` - Service shutdown script
- `deploy_model.sh` - Model deployment script
- `mock_predictor_server.py` - Mock predictor (used by deploy script)
- `mock_vllm_server.py` - Mock vLLM server (deployed via PyLet)
- `README.md` - This file

## Related

- Mock LLM cluster: `examples/mock_llm_cluster/` (multi-scheduler, 2 models)
- Multi-scheduler: `examples/multi_scheduler/` (per-model schedulers, sleep models)
- PyLet benchmark: `examples/pylet_benchmark/` (direct registration, no planner)
