# LLM Cluster Example

Single-scheduler setup with 3 LLM models using PyLet for container orchestration.

## Overview

This example demonstrates a complete end-to-end LLM inference system with:
- **Single Scheduler**: Handles all 3 models in one process
- **3 LLM Models**: llm_fast, llm_medium, llm_slow with different performance characteristics
- **PyLet Orchestration**: Deploys model instances to PyLet workers
- **Optimizer-based Deployment**: Uses planner to compute optimal instance distribution

## Architecture

```
┌─────────────────────────────────────────────┐
│           Client                            │
└──────────────────┬──────────────────────────┘
                   │ Task submissions
                   ▼
┌─────────────────────────────────────────────┐
│  Scheduler (Single, All Models)             │
│  - Receives tasks for llm_fast|medium|slow  │
│  - Routes to appropriate instances          │
│  - Polls predictor for runtime predictions  │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
   ┌────────┐           ┌──────────┐
   │Predictor           │ Planner  │
   │        │           │          │
   └────────┘           │ - Deploy │
                        │ - Monitor│
                        └────┬─────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  PyLet Cluster  │
                    │  32 Workers     │
                    │  - llm_fast     │
                    │  - llm_medium   │
                    │  - llm_slow     │
                    └─────────────────┘
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
- Ports available: 8000 (scheduler), 8002 (predictor), 8003 (planner)

### 1. Start PyLet Cluster

```bash
./scripts/start_pylet_test_cluster.sh
```

### 2. Start Services

```bash
./examples/llm_cluster/start_cluster.sh
```

This starts:
- Mock Predictor (port 8002)
- Scheduler (port 8000)
- Planner (port 8003, PyLet-enabled)

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
- QPS ratio support (5:1:3)
- Per-model statistics
- Task completion polling with timeout
- Rich progress display
- OpenAI-compatible task format

**Options:**
```
--scheduler-url       Scheduler URL (default: http://localhost:8000)
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
1. Port availability checks
2. PyLet cluster verification
3. Predictor startup (port 8002)
4. Scheduler startup (port 8000) with library predictor mode
5. Planner startup (port 8003) with PyLet integration
6. Health checks on all services

**Configuration (environment variables):**
- `PREDICTOR_PORT`: Mock predictor port (default: 8002)
- `SCHEDULER_PORT`: Scheduler port (default: 8000)
- `PLANNER_PORT`: Planner port (default: 8003)
- `PYLET_HEAD_PORT`: PyLet head port (default: 5100)

### stop_cluster.sh

Graceful service shutdown.

**Steps:**
1. Terminate PyLet instances via planner
2. Stop Planner
3. Stop Scheduler
4. Stop Predictor

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

Check logs: `tail -f /tmp/llm_cluster/pylet_head.log`

Verify cluster health:
```bash
curl http://localhost:5100/workers | python3 -m json.tool
```

### Scheduler not healthy

Check predictor mode:
```bash
grep "PREDICTOR_MODE" /tmp/llm_cluster/scheduler.log
```

Should be "library" for this example.

### Planner deployment fails

Verify scheduler is healthy:
```bash
curl http://localhost:8000/v1/health
```

Check planner logs for PyLet connection issues:
```bash
tail -f /tmp/llm_cluster/planner.log
```

### Tasks not completing

Verify instances are deployed:
```bash
curl http://localhost:8003/v1/instance/list | python3 -m json.tool
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
- `mock_predictor_server.py` - Mock predictor (pre-existing)
- `mock_vllm_server.py` - Mock vLLM server (pre-existing)
- `README.md` - This file

## Related

- Mock cluster: `examples/mock_llm_cluster/` (multi-scheduler)
- Multi-scheduler: `examples/multi_scheduler/` (per-model schedulers)
- PyLet benchmark: `examples/pylet_benchmark/` (direct registration, no planner)
