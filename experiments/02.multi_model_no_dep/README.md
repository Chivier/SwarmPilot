# Experiment 02: Multi-Model No-Dependency

## Overview

This experiment extends experiment 01 by introducing a **dual-scheduler** architecture where two independent schedulers manage separate instance groups, each handling different workload characteristics. The design eliminates dependencies between the two groups, allowing for independent evaluation of scheduling strategies under different load patterns.

## Experimental Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Predictor (8101)                        │
│                   Shared by both groups                      │
└──────────────────┬─────────────────┬────────────────────────┘
                   │                 │
       ┌───────────▼─────────┐   ┌──▼──────────────────┐
       │  Scheduler A (8100) │   │  Scheduler B (8200) │
       │  Bimodal Workload   │   │  Pareto Workload    │
       └───────────┬─────────┘   └──┬──────────────────┘
                   │                 │
       ┌───────────▼───────────┐ ┌──▼──────────────────┐
       │   Group A Instances   │ │   Group B Instances │
       │   instance-000..N1-1  │ │   instance-N1..N1+N2│
       │   Ports: 8210-82xx    │ │   Ports: 8300-83xx  │
       │   Model: sleep_model  │ │   Model: sleep_model│
       └───────────────────────┘ └─────────────────────┘
```

### Key Components

1. **Predictor Service** (Port 8101)
   - Shared prediction service for both schedulers
   - Provides execution time predictions for scheduling decisions

2. **Scheduler A** (Port 8100)
   - Manages Group A instances (N1 instances)
   - Handles bimodal workload distribution
   - Independent Poisson arrival process

3. **Scheduler B** (Port 8200)
   - Manages Group B instances (N2 instances)
   - Handles Pareto long-tail workload distribution
   - Independent Poisson arrival process

4. **Instance Groups**
   - **Group A**: N1 instances on ports 8210-82xx
     - Register with Scheduler A
     - Handle bimodal workload (peaks at 2s and 8.5s)

   - **Group B**: N2 instances on ports 8300-83xx
     - Register with Scheduler B
     - Handle Pareto long-tail workload (1-10s range)

### Workload Characteristics

#### Workload A: Bimodal Distribution
- **Type**: Bimodal (same as experiment 01)
- **Left Peak**: 1-3 seconds, mean=2.0s, std=0.4s (50%)
- **Right Peak**: 7-10 seconds, mean=8.5s, std=0.6s (50%)
- **Use Case**: Simulates mixed short/long task scenarios

#### Workload B: Pareto Long-Tail Distribution
- **Type**: Pareto (power-law distribution)
- **Range**: 1-10 seconds
- **Characteristics**:
  - ~80% of tasks complete quickly (short tail)
  - ~20% of tasks take significantly longer (long tail)
  - Alpha parameter: 1.5 (controls tail heaviness)
- **Use Case**: Simulates real-world scenarios with occasional heavy tasks

### Task Submission

- **Independent Poisson Processes**: Each scheduler receives tasks via its own Poisson process
- **Configurable QPS**:
  - `--qps1`: Queries per second for Scheduler A
  - `--qps2`: Queries per second for Scheduler B
- **Randomized Order**: Task arrival times are independent and follow exponential inter-arrival distribution

### Scheduling Strategies

Three strategies are tested on both schedulers:

1. **round_robin**: Distribute tasks evenly across instances
2. **min_time**: Assign tasks to the instance with minimum predicted completion time
3. **probabilistic**: Balance load using probabilistic selection based on predicted times

## Configuration

### Instance Group Configuration

The experiment supports flexible instance group sizing:

- **N1**: Number of instances in Group A (default: 10)
- **N2**: Number of instances in Group B (default: 6)
- **Total**: N1 + N2 instances

Example configurations:
- Balanced: N1=8, N2=8 (16 total)
- Skewed: N1=12, N2=4 (16 total)
- Small test: N1=4, N2=2 (6 total)

### QPS Configuration

Independent query rates for each scheduler:
- **qps1**: Target QPS for Scheduler A (default: 8.0)
- **qps2**: Target QPS for Scheduler B (default: 5.0)

## Usage

### Prerequisites

1. **System Requirements**:
   - Docker installed and running
   - Python 3.9+ with uv package manager
   - Ports 8100-8399 available

2. **Project Setup**:
   ```bash
   # Install dependencies
   cd experiments/02.multi_model_no_dep
   uv sync
   ```

3. **Build Docker Image** (if not already built):
   ```bash
   cd ../../instance
   ./build_sleep_model.sh
   ```

### Starting Services

Start all services with default configuration (N1=10, N2=6):
```bash
./start_all_services.sh
```

Start with custom instance counts:
```bash
./start_all_services.sh 8 8   # N1=8, N2=8
./start_all_services.sh 12 4  # N1=12, N2=4
```

### Running Experiments

Run with default parameters:
```bash
uv run python test_dual_scheduler.py
```

Run with custom configuration:
```bash
# Custom instance counts and QPS
uv run python test_dual_scheduler.py --n1 8 --n2 8 --qps1 10.0 --qps2 6.0

# More tasks per scheduler
uv run python test_dual_scheduler.py --num-tasks 200

# Test specific strategies only
uv run python test_dual_scheduler.py --strategies round_robin probabilistic
```

### Command-Line Options

```
--n1 INT              Number of instances in Group A (default: 10)
--n2 INT              Number of instances in Group B (default: 6)
--qps1 FLOAT          QPS for Scheduler A (default: 8.0)
--qps2 FLOAT          QPS for Scheduler B (default: 5.0)
--num-tasks INT       Tasks per scheduler per strategy (default: 100)
--strategies [LIST]   Strategies to test (default: all three)
```

### Stopping Services

```bash
./stop_all_services.sh
```

## Results

### Output Format

Results are saved as JSON files in `results/` directory with timestamp:
```
results/results_dual_YYYYMMDD_HHMMSS.json
```

### Metrics Collected

For each scheduler and strategy:
- **Submission metrics**: Number of tasks, actual QPS, submission time
- **Completion metrics**: Average, median, P95, P99 completion times
- **Total execution time**: Time from first submission to last completion
- **Instance distribution**: Task distribution across instances (both submitted and completed)

### Example Results Structure

```json
{
  "experiment": "02.multi_model_no_dep",
  "timestamp": "2025-01-15T10:30:00",
  "config": {
    "n1": 10,
    "n2": 6,
    "qps1": 8.0,
    "qps2": 5.0,
    "num_tasks": 100,
    "workload_a": {
      "type": "bimodal",
      "description": "...",
      "mean": 5.25,
      "std": 3.24
    },
    "workload_b": {
      "type": "pareto",
      "description": "...",
      "mean": 4.12,
      "std": 2.87
    }
  },
  "results": [
    {
      "strategy": "round_robin",
      "scheduler_a": { /* metrics */ },
      "scheduler_b": { /* metrics */ }
    },
    ...
  ]
}
```

## Logs

Logs are organized in the `logs/` directory:

```
logs/
├── predictor.log          # Predictor service log
├── scheduler-a.log        # Scheduler A log
├── scheduler-b.log        # Scheduler B log
├── instance_8200.log      # Group A instance logs
├── instance_8201.log
├── ...
├── instance_8300.log      # Group B instance logs
├── instance_8301.log
└── ...
```

## Key Differences from Experiment 01

1. **Dual Schedulers**: Two independent schedulers instead of one
2. **Different Workloads**: Bimodal vs. Pareto long-tail distributions
3. **Independent Processes**: Separate Poisson processes for each scheduler
4. **Instance Grouping**: Explicit grouping with separate port ranges
5. **Flexible Sizing**: Configurable instance counts (N1, N2) instead of fixed 16

## Reuse from Experiment 01

This experiment maximizes code reuse:
- ✅ Service CLIs (scheduler/instance/predictor) - unchanged
- ✅ WebSocket subscription mechanism - reused
- ✅ Poisson process logic - duplicated for two schedulers
- ✅ Bimodal distribution - copied from exp01
- ✅ Metrics calculation - reused
- 🆕 Pareto distribution - new implementation
- 🆕 Dual-scheduler coordination - new architecture

## Troubleshooting

### Port Conflicts
If you see "address already in use" errors:
```bash
# Check what's using the ports
netstat -tulpn | grep 810
netstat -tulpn | grep 820
netstat -tulpn | grep 830

# Stop any existing services
./stop_all_services.sh
```

### Docker Issues
```bash
# Check Docker is running
docker ps

# Clean up old containers
docker ps -a | grep sleep_model | awk '{print $1}' | xargs docker rm -f

# Rebuild image if needed
cd ../../instance && ./build_sleep_model.sh
```

### Service Health Checks
```bash
# Check predictor
curl http://localhost:8101/health

# Check Scheduler A
curl http://localhost:8100/health

# Check Scheduler B
curl http://localhost:8200/health

# Check an instance
curl http://localhost:8200/health
```

### Log Analysis
```bash
# View recent logs
tail -f logs/scheduler-a.log
tail -f logs/scheduler-b.log

# Search for errors
grep -i error logs/*.log
```

## Expected Behavior

1. **Scheduler A** (Bimodal workload):
   - Round-robin: Even distribution, moderate completion times
   - Min-time: May show instance imbalance due to prediction accuracy
   - Probabilistic: Balanced load with good P95/P99 times

2. **Scheduler B** (Pareto workload):
   - More variation in completion times due to long-tail nature
   - Probabilistic strategy may show better P95/P99 handling
   - Instance distribution may be less even due to tail tasks

## Future Enhancements

Possible extensions:
1. **Cross-scheduler dependencies**: Tasks that depend on both groups
2. **Dynamic rebalancing**: Move instances between groups based on load
3. **More workload types**: Step function, periodic bursts, etc.
4. **Resource constraints**: Memory/CPU limits per instance group
5. **Failure scenarios**: Instance failures, network partitions

## References

- Experiment 01: `../01.quick-start-up/README.md`
- Scheduler documentation: `../../scheduler/README.md`
- Instance documentation: `../../instance/README.md`
- Predictor documentation: `../../predictor/README.md`
