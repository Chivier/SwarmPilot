# Service Management Scripts

## Quick Start

### Local Simulation (Single Machine)

```bash
cd experiments/13.workflow_benchmark

# Start services
./scripts/start_all_services.sh

# Run experiment
python tools/cli.py run-text2video-sim --duration 300 --num-workflows 600

# Stop services
./scripts/stop_all_services.sh
```

### Real Cluster (Multi-Node)

```bash
# Step 1: Start infrastructure (run on EACH node)
./scripts/start_real_service.sh                 # Without auto-optimize
./scripts/start_real_service.sh --auto-optimize # With auto-optimize

# Step 2: Deploy models (run from client node 29.209.114.166)
# Type1 (Text2Video):
./scripts/manual_deploy_type1.sh --model-path-a /path/to/llm --model-path-b /path/to/t2vid

# Type2 (Deep Research):
./scripts/manual_deploy_type2.sh --model-path-a /path/to/small_llm --model-path-b /path/to/large_llm

# Step 3: Stop services
./scripts/stop_real_services.sh
```

---

## Scripts Overview

| Script | Mode | Description |
|--------|------|-------------|
| `start_all_services.sh` | Local | Start all services with models |
| `stop_all_services.sh` | Local | Stop all local services |
| `deploy_models_local.sh` | Local | Redeploy models to running instances |
| `start_real_service.sh` | Cluster | Start infrastructure only (no models) |
| `manual_deploy_type1.sh` | Cluster | Deploy Text2Video models |
| `manual_deploy_type2.sh` | Cluster | Deploy Deep Research models |
| `stop_real_services.sh` | Cluster | Stop all cluster services |

---

## Cluster Configuration

| Role | IP | Port |
|------|-----|------|
| Scheduler A | 29.209.114.51 | 8100 |
| Scheduler B | 29.209.113.228 | 8100 |
| Predictor | 29.209.113.113 | 8100 |
| Planner/Client | 29.209.114.166 | 8100 |

**Registration Strategy:**
```
Ports 8200-8203 → Scheduler
Ports 8204-8207 → Planner
```

**Models:**
| Workflow | Group A | Group B |
|----------|---------|---------|
| Type1 (Text2Video) | `llm_service_small_model` | `t2vid` |
| Type2 (Deep Research) | `llm_service_small_model` | `llm_service_large_model` |

---

## Type5: OOD Recovery Experiment

Run the OOD recovery experiment comparing baseline vs recovery modes:

```bash
cd experiments/13.workflow_benchmark

# Run both modes for comparison (recommended)
python -m scripts.run_type5_ood_experiment \
    --num-tasks 500 --qps 2.83 --mode both

# Run with custom services
python -m scripts.run_type5_ood_experiment \
    --scheduler-url http://custom:8100 \
    --predictor-url http://custom:8000 \
    --skip-service-check \
    --num-tasks 500 --qps 2.83
```

**Optimal Configurations:**
| Instances | QPS | Expected Improvement |
|-----------|-----|---------------------|
| 48 | 2.83 | ~23% |
| 128 | 7.90 | ~38% |

See [type5_ood_recovery/README.md](../type5_ood_recovery/README.md) for detailed documentation.

---

## Troubleshooting

```bash
# Check logs
tail -f logs/*.log

# Health check
curl http://localhost:8100/health

# Force stop
pkill -f "scheduler.*start"
pkill -f "instance.*start"
```
