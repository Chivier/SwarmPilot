# Experiment: 01.quick-start-up

## Overview

This experiment tests all three scheduling strategies supported by the SwarmPilot scheduler:
- **Round Robin**: Simple cyclic distribution
- **Minimum Time**: Queue-aware scheduling based on expected completion time
- **Probabilistic**: Tail-latency-aware scheduling with quantile-based predictions

## Objectives

1. Deploy a complete SwarmPilot environment with:
   - 1 Predictor service
   - 1 Scheduler service
   - 16 Instance services running sleep-model

2. Test all three scheduling strategies with:
   - 100 tasks per strategy
   - Task execution times following normal distribution (1.5s-5s)
   - Experiment mode enabled (no trained models required)

3. Measure and compare:
   - Average task completion time (submit → complete)
   - P95 and P99 completion times
   - Total execution time (first submit → last complete)
   - Task distribution across instances

## Architecture

```
┌─────────────┐
│   Client    │
│ (test.py)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌─────────────┐
│  Scheduler  │─────▶│  Predictor  │
│   :8000     │      │    :8001    │
└──────┬──────┘      └─────────────┘
       │
       ├─────────────┬─────────────┬─────...
       ▼             ▼             ▼
   Instance-000  Instance-001  Instance-002  ... Instance-015
    :5000/:6000   :5001/:6001   :5002/:6002      :5015/:6015
```

## Configuration

### Services
- **Predictor**: Port 8001, experiment mode enabled
- **Scheduler**: Port 8000, connected to predictor
- **Instances**: Ports 5000-5015 (API) and 6000-6015 (models)

### Task Generation
- **Distribution**: Normal distribution N(μ=3.25, σ=0.583)
- **Range**: 1.5s to 5.0s (99.7% of values within 3σ)
- **Count**: 100 tasks per strategy

### Strategy-Specific Settings

#### Round Robin
- No predictor dependency
- exp_runtime = actual task execution time
- Expected: Even distribution across instances

#### Minimum Time
- Prediction type: expect_error
- exp_runtime = dataset mean (3.25s = 3250ms) for all tasks
- Expected: Queue-aware distribution, minimizes average completion time

#### Probabilistic
- Prediction type: quantile
- Target quantile: P90 (0.9)
- exp_runtime = actual task execution time
- Expected: Tail-latency-aware distribution, optimizes P90 completion time

## Directory Structure

```
01.quick-start-up/
├── start_all_services.sh      # Start all services
├── stop_all_services.sh       # Stop all services
├── test_scheduling.py         # Test script (sequential mode)
├── test_scheduling_poisson.py # Test script (Poisson process mode, WebSocket-based)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── logs/                      # Service logs (created on startup)
│   ├── predictor.log
│   ├── scheduler.log
│   └── instance-*.log
└── results/                   # Test results (created on test run)
    ├── results_*.json         # Sequential mode results
    └── results_poisson_*.json # Poisson mode results
```

## Setup

### Prerequisites

1. Install uv (Python package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Ensure Docker is installed and running:
```bash
docker --version
```

3. Navigate to experiment directory:
```bash
cd experiments/01.quick-start-up
```

### Install Python Dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Running the Experiment

There are two test modes available:

1. **Sequential Mode** (`test_scheduling.py`): Submit all tasks at once, wait for completion sequentially
2. **Poisson Process Mode** (`test_scheduling_poisson.py`): Submit tasks following Poisson process with configurable QPS, use WebSocket for event-driven result collection

### Step 1: Start All Services

```bash
./start_all_services.sh
```

This script will:
1. Start predictor service on port 8001
2. Start scheduler service on port 8000
3. Build sleep-model Docker image (if not already built)
4. Start 16 instance services on ports 5000-5015
5. Start sleep-model on each instance
6. Register all instances with scheduler

Expected output:
```
=========================================
Starting 01.quick-start-up Experiment
=========================================

Step 1: Starting Predictor Service
...
Step 6: Registering all instances with scheduler
...
All services started successfully!
```

### Step 2: Run Tests

#### Option A: Sequential Mode (Original)

```bash
python test_scheduling.py
```

The test script will:
1. Verify scheduler is healthy
2. Generate 100 task times from normal distribution
3. Test each strategy sequentially:
   - **Clear all tasks** from scheduler (POST /task/clear)
   - **Set strategy** (POST /strategy/set)
   - Submit all tasks at once
   - Poll for completion
   - Collect metrics
4. Save results to `results/results_TIMESTAMP.json`
5. Print comparison table

Expected runtime: ~15-20 minutes (depends on task distribution)

#### Option B: Poisson Process Mode (Recommended)

```bash
# Default: 100 tasks, 10 QPS
python test_scheduling_poisson.py

# Custom QPS and task count
python test_scheduling_poisson.py --qps 20 --num-tasks 200

# Test specific strategies
python test_scheduling_poisson.py --strategies min_time probabilistic

# Full options
python test_scheduling_poisson.py --qps 15 --num-tasks 150 --strategies round_robin min_time probabilistic
```

**Features:**
- **Event-driven architecture**: Uses WebSocket for real-time result updates (no polling)
- **Poisson process submission**: Tasks are submitted following exponential inter-arrival times
- **Two-threaded design**:
  - Thread 1 (Receiver): Starts first, subscribes to all task IDs via WebSocket
  - Thread 2 (Submitter): Starts second, submits tasks following Poisson process
- **Configurable QPS**: Control task submission rate (queries per second)
- **Shared task data**: All strategies test with the same pre-generated task inputs

The test script will:
1. Verify scheduler is healthy
2. Generate task times from normal distribution (shared across strategies)
3. For each strategy:
   - **Clear all tasks** from scheduler (POST /task/clear)
   - **Set strategy** (POST /strategy/set)
   - Start WebSocket receiver thread (subscribe to all task IDs)
   - Start Poisson submitter thread (submit tasks with exponential inter-arrival times)
   - Collect results in real-time via WebSocket
   - Calculate metrics
4. Save results to `results/results_poisson_TIMESTAMP.json`
5. Print comparison table with actual QPS achieved

Expected runtime: Depends on QPS and task count
- 100 tasks @ 10 QPS: ~10s submission + task execution time
- 200 tasks @ 20 QPS: ~10s submission + task execution time

**Command-line arguments:**
- `--qps FLOAT`: Target queries per second (default: 10.0)
- `--num-tasks INT`: Number of tasks per strategy (default: 100)
- `--strategies STR [STR ...]`: Strategies to test (default: all three)

### Step 3: Stop All Services

```bash
./stop_all_services.sh
```

This script will:
1. Stop all instance services
2. Stop scheduler service
3. Stop predictor service
4. Clean up Docker containers

## Results

Results are saved in JSON format to `results/results_TIMESTAMP.json`:

```json
{
  "experiment": "01.quick-start-up",
  "timestamp": "2025-01-15T10:30:00",
  "config": {
    "num_tasks": 100,
    "task_mean": 3.25,
    "task_std": 0.583,
    "task_min": 1.5,
    "task_max": 5.0
  },
  "results": [
    {
      "strategy": "round_robin",
      "num_tasks": 100,
      "num_completed": 100,
      "avg_completion_time": 12.5,
      "median_completion_time": 11.8,
      "p95_completion_time": 18.2,
      "p99_completion_time": 20.1,
      "total_time": 25.3,
      "instance_distribution": {
        "instance-000": 6,
        "instance-001": 7,
        ...
      }
    },
    ...
  ]
}
```

## Metrics Explained

- **avg_completion_time**: Average time from task submission to completion (seconds)
- **median_completion_time**: Median completion time (50th percentile)
- **p95_completion_time**: 95th percentile completion time
- **p99_completion_time**: 99th percentile completion time
- **total_time**: Time from first task submission to last task completion
- **instance_distribution**: Number of tasks assigned to each instance

## Expected Behavior

### Round Robin
- **Distribution**: Even (~6-7 tasks per instance)
- **Completion time**: Moderate, no optimization
- **Use case**: Simple, predictable load balancing

### Minimum Time
- **Distribution**: Uneven (queue-aware)
- **Completion time**: Optimized average, may have high tail latency
- **Use case**: Minimizing average task latency

### Probabilistic
- **Distribution**: Probabilistic (accounts for variability)
- **Completion time**: Optimized P90, better tail latency control
- **Use case**: SLA requirements, tail latency optimization

## Troubleshooting

### Services won't start
- Check if ports 5000-5015, 6000-6015, 8000-8001 are available
- Check Docker is running: `docker ps`
- Check logs in `logs/` directory

### Tasks fail or timeout
- Check instance health: `curl http://localhost:5000/health`
- Check scheduler health: `curl http://localhost:8000/health`
- Increase timeout in test script

### Docker image not found
- Ensure sleep-model is built: `cd ../../instance && ./build_sleep_model.sh`
- Check image exists: `docker images | grep sleep_model`

## Technical Notes

### WebSocket-Based Event-Driven Architecture (Poisson Mode)

The Poisson process test mode uses a two-threaded architecture for efficient, event-driven task execution:

#### Thread Architecture

```
Main Thread
    │
    ├─→ Thread 2 (WebSocket Receiver) [Starts First]
    │   ├─→ Connect to ws://scheduler:8100/task/get_result
    │   ├─→ Send subscribe message with all task IDs
    │   ├─→ Listen for results asynchronously
    │   └─→ Put results into Queue
    │
    └─→ Thread 1 (Poisson Submitter) [Starts Second]
        ├─→ Generate exponential inter-arrival times
        ├─→ Submit tasks following Poisson process
        └─→ Track submission metrics
```

#### Why This Design?

1. **WebSocket First**: Ensures no results are missed by establishing the subscription before any tasks are submitted
2. **Event-Driven**: No polling overhead, results arrive as soon as tasks complete
3. **Poisson Process**: Realistic workload simulation with configurable QPS
4. **Non-Blocking**: Both threads run independently, maximizing throughput

#### WebSocket Protocol

**Subscribe Message (Client → Server):**
```json
{
  "type": "subscribe",
  "task_ids": ["task-001", "task-002", ...]
}
```

**Result Message (Server → Client):**
```json
{
  "type": "result",
  "task_id": "task-001",
  "status": "completed",
  "result": {"output": "..."},
  "error": null,
  "execution_time_ms": 234.56,
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

**Error Message (Server → Client):**
```json
{
  "type": "error",
  "task_id": "task-001",
  "status": "failed",
  "error": "Error message",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

#### Poisson Process Implementation

Task submission follows a Poisson process:
- Inter-arrival times: Exponential distribution with rate λ = QPS
- Formula: `wait_time = random.exponential(1 / QPS)`
- Example: QPS=10 → average 0.1s between tasks, but with variance

This simulates realistic production workloads where requests arrive randomly but with a predictable average rate.

### Experiment Mode
This experiment uses "experiment mode" in the predictor, which allows testing without trained models:
- Platform info set to `{"software_name": "exp", "software_version": "exp", "hardware_name": "exp"}`
- Predictor returns synthetic predictions based on `exp_runtime` metadata
- For expect_error: `error_margin = exp_runtime * 0.05`
- For quantile: P50=exp_runtime, P90=exp_runtime*1.05, P95=exp_runtime*1.075, P99=exp_runtime*1.12

### Strategy Switching
When switching strategies, the scheduler:
1. Clears all pending tasks
2. Reinitializes instance queue information
3. Cannot switch if tasks are currently running

Therefore, the test script waits for all tasks to complete before switching to the next strategy.

### Port Allocation
- Predictor: 8001
- Scheduler: 8000
- Instance N: 5000 + N (API), 6000 + N (model container)

## References

- [Scheduler Documentation](../../scheduler/docs/)
- [Instance Documentation](../../instance/docs/)
- [Predictor Documentation](../../predictor/docs/)
