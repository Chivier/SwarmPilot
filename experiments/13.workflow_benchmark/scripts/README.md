# Service Management Scripts

This directory contains scripts for managing the SwarmPilot services required by the Unified Workflow Benchmark Framework.

## Overview

The scripts have been adapted from the original experiment scripts to work from the `experiments/13.workflow_benchmark` directory. All paths are automatically adjusted to work relative to this location.

## Scripts

### 1. `start_all_services.sh`

Starts all required services for running workflow experiments in simulation or real mode.

**Services Started**:
- Predictor (port 8101)
- Planner (port 8202)
- Scheduler A (port 8100)
- Scheduler B (port 8200)
- Instance Group A (ports 8210+)
- Instance Group B (ports 8300+)

**Usage**:
```bash
# From experiments/13.workflow_benchmark directory
./scripts/start_all_services.sh [N1] [N2] [MODEL_ID_A] [MODEL_ID_B]

# Or with environment variables
N1=10 N2=6 ./scripts/start_all_services.sh

# Examples
./scripts/start_all_services.sh                              # Default: N1=4, N2=2
./scripts/start_all_services.sh 8 4                          # 8 Group A, 4 Group B
./scripts/start_all_services.sh 10 6 llm_service_small_model t2vid  # Real models
```

**Parameters**:
- `N1`: Number of instances in Group A (default: 4)
- `N2`: Number of instances in Group B (default: 2)
- `MODEL_ID_A`: Model ID for Group A (default: sleep_model_a)
- `MODEL_ID_B`: Model ID for Group B (default: sleep_model_b)

**Environment Variables**:
- `N1`: Override number of Group A instances
- `N2`: Override number of Group B instances
- `MODEL_ID_A`: Override Group A model ID
- `MODEL_ID_B`: Override Group B model ID
- `AUTO_OPTIMIZE_ENABLED`: Enable/disable planner optimization (default: True)

**Output**:
- Logs: `logs/` directory (in experiment root)
- PID files: `logs/*.pid` for each service

---

### 2. `stop_all_services.sh`

Stops all running services gracefully (or forcefully if needed).

**Usage**:
```bash
# From experiments/13.workflow_benchmark directory
./scripts/stop_all_services.sh
```

**What It Does**:
1. Stops all instance services (parallel)
2. Stops scheduler services (parallel)
3. Stops planner service
4. Stops predictor service
5. Cleans up any remaining processes
6. Stops Docker containers for models

**Cleanup**:
- Removes PID files
- Stops Docker containers (sleep_model, llm_service, t2vid)
- Uses SIGTERM first, then SIGKILL if needed

---

### 3. `deploy_models_local.sh`

Deploys models to running instances. Called automatically by `start_all_services.sh` but can be used standalone for redeployment.

**Usage**:
```bash
# From experiments/13.workflow_benchmark directory
./scripts/deploy_models_local.sh [OPTIONS]

# Examples
./scripts/deploy_models_local.sh \
    --model-id-a sleep_model_a \
    --model-id-b sleep_model_b \
    --n1 4 \
    --n2 2

./scripts/deploy_models_local.sh \
    --model-id-a llm_service_small_model \
    --model-id-b t2vid \
    --model-path-a /path/to/llm/model \
    --model-path-b /path/to/t2vid/model \
    --n1 10 \
    --n2 6
```

**Options**:
- `--scheduler-a-url URL`: Scheduler A URL (default: http://localhost:8100)
- `--scheduler-b-url URL`: Scheduler B URL (default: http://localhost:8200)
- `--planner-url URL`: Planner URL (default: http://localhost:8202)
- `--model-id-a ID`: Model ID for Group A (default: sleep_model_a)
- `--model-id-b ID`: Model ID for Group B (default: sleep_model_b)
- `--n1 NUM`: Number of Group A instances (default: 4)
- `--n2 NUM`: Number of Group B instances (default: 2)
- `--port-a-start PORT`: Starting port for Group A (default: 8210)
- `--port-b-start PORT`: Starting port for Group B (default: 8300)
- `--model-path PATH`: Model path for both groups
- `--model-path-a PATH`: Model path for Group A only
- `--model-path-b PATH`: Model path for Group B only
- `-h, --help`: Show help message

**Deployment Strategy**:
- First half of instances: register to Scheduler A/B
- Second half of instances: register to Planner
- All deployments run in parallel for speed

---

## Typical Workflows

### Simulation Mode (Sleep Models)

```bash
# 1. Start services with sleep models (fast startup)
cd experiments/13.workflow_benchmark
./scripts/start_all_services.sh 4 2 sleep_model_a sleep_model_b

# 2. Run experiment
python tools/cli.py run-text2video-sim --duration 300 --num-workflows 600

# 3. Stop services
./scripts/stop_all_services.sh
```

### Real Mode (Actual Models)

```bash
# 1. Start services with real models
cd experiments/13.workflow_benchmark
./scripts/start_all_services.sh 10 6 llm_service_small_model t2vid

# 2. Run experiment (note: real mode uses fewer workflows)
python tools/cli.py run-text2video-real --duration 300 --num-workflows 100

# 3. Stop services
./scripts/stop_all_services.sh
```

### Redeploying Models Without Restart

```bash
# If services are already running, redeploy different models
./scripts/deploy_models_local.sh \
    --model-id-a llm_service_large_model \
    --model-id-b t2vid \
    --n1 10 \
    --n2 6
```

---

## Directory Structure

```
experiments/13.workflow_benchmark/
├── scripts/
│   ├── README.md                    # This file
│   ├── start_all_services.sh        # Start all services
│   ├── stop_all_services.sh         # Stop all services
│   └── deploy_models_local.sh       # Deploy models to instances
│
├── logs/                            # Service logs and PID files (created automatically)
│   ├── predictor.log
│   ├── predictor.pid
│   ├── planner.log
│   ├── planner.pid
│   ├── scheduler-a.log
│   ├── scheduler-a.pid
│   ├── scheduler-b.log
│   ├── scheduler-b.pid
│   ├── instance-a-*.log
│   ├── instance-a-*.pid
│   ├── instance-b-*.log
│   └── instance-b-*.pid
│
└── output/                          # Experiment outputs
    └── metrics.json
```

---

## Port Allocation

| Service | Default Port | Description |
|---------|--------------|-------------|
| Predictor | 8101 | Performance prediction service |
| Planner | 8202 | Auto-optimization planner |
| Scheduler A | 8100 | Scheduler for Group A (e.g., LLM models) |
| Scheduler B | 8200 | Scheduler for Group B (e.g., T2V models) |
| Group A Instances | 8210-82xx | Instances running Model A |
| Group B Instances | 8300-83xx | Instances running Model B |

---

## Troubleshooting

### Issue: Services won't start

**Check logs**:
```bash
tail -f logs/predictor.log
tail -f logs/scheduler-a.log
```

**Common causes**:
- Port already in use
- Missing dependencies (run `uv sync` in each service directory)
- Insufficient resources (reduce N1/N2)

### Issue: Health checks fail

**Manual health check**:
```bash
curl http://localhost:8101/health  # Predictor
curl http://localhost:8100/health  # Scheduler A
curl http://localhost:8200/health  # Scheduler B
curl http://localhost:8202/health  # Planner
curl http://localhost:8210/health  # First Group A instance
```

### Issue: Models won't deploy

**Check instance health**:
```bash
curl http://localhost:8210/health
curl http://localhost:8300/health
```

**Redeploy manually**:
```bash
./scripts/deploy_models_local.sh --n1 4 --n2 2
```

### Issue: Processes won't stop

**Force cleanup**:
```bash
# Kill all related processes
pkill -f "predictor.*start"
pkill -f "scheduler.*start"
pkill -f "planner.*uvicorn"
pkill -f "instance.*start"

# Stop Docker containers
docker stop $(docker ps -q)
```

---

## Advanced Configuration

### Custom Port Allocation

Edit the script variables at the top of `start_all_services.sh`:
```bash
PREDICTOR_PORT=8101
SCHEDULER_A_PORT=8100
SCHEDULER_B_PORT=8200
PLANNER_PORT=8202
INSTANCE_GROUP_A_START_PORT=8210
INSTANCE_GROUP_B_START_PORT=8300
```

### Disable Auto-Optimization

```bash
AUTO_OPTIMIZE_ENABLED=False ./scripts/start_all_services.sh
```

### Custom Log Directory

The log directory is automatically set to `logs/` in the experiment directory. To change it, edit:
```bash
LOG_DIR="$EXPERIMENT_DIR/logs"  # In start_all_services.sh
```

---

## Notes

- **Path Adaptation**: All paths are automatically adjusted relative to the experiment directory (`experiments/13.workflow_benchmark`)
- **Parallel Startup**: Instances start in parallel for faster initialization
- **Health Checks**: Scripts wait for services to be healthy before proceeding
- **Graceful Shutdown**: Stop script uses SIGTERM first, SIGKILL only if needed
- **Docker Integration**: Scripts handle Docker container lifecycle for model services

---

## References

- Original Text2Video scripts: `experiments/03.Exp4.Text2Video/`
- Original Deep Research scripts: `experiments/07.Exp2.Deep_Research_Migration_Test/`
- Main documentation: `../docs/QUICKSTART.md`, `../README.md`
