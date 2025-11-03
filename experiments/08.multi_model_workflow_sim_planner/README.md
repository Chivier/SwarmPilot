# Experiment 08: Multi-Model Workflow with Dynamic Instance Migration

## Quick Start

### Run All 6 Comparison Experiments (Recommended)

```bash
# 1. Navigate to experiment directory
cd experiments/08.multi_model_workflow_sim_planner

# 2. Run all 6 configurations automatically (1.5-3 hours)
uv run python3 run_comparison_experiments.py --num-workflows-per-phase 100

# 3. Analyze and compare results
uv run python3 analyze_comparison_results.py

# 4. View the comparison report
cat results/comparison_analysis_*.md
```

**What this does:** Automatically tests 3 scheduling strategies (min_time, probabilistic, round_robin) × 2 modes (with/without migration) = 6 configurations, then generates comprehensive comparison reports.

### Run Single Experiment

```bash
# 1. Start all services
./start_all_services.sh

# 2. Run experiment with dynamic migration (default)
uv run python3 test_dynamic_migration.py \
    --strategy min_time \
    --num-workflows-per-phase 100

# 3. Run experiment with static distribution (no migration)
uv run python3 test_dynamic_migration.py \
    --strategy probabilistic \
    --num-workflows-per-phase 100 \
    --disable-migration

# 4. Stop services when done
./stop_all_services.sh
```

### View Results

```bash
# Individual experiment results
ls -lh results/exp08_*.json

# Latest comparison analysis
cat results/comparison_analysis_*.md
```

📖 **Detailed Guide:** See [COMPARISON_EXPERIMENTS_GUIDE.md](COMPARISON_EXPERIMENTS_GUIDE.md) for complete instructions.

---

## Overview

This experiment extends Experiment 04 by adding **dynamic instance migration** between schedulers during phase transitions. The experiment demonstrates how instances can be safely migrated to maintain optimal resource distribution as workload characteristics change.

The experiment now supports **comparison testing** of 6 different configurations to evaluate the impact of scheduling strategies and migration modes on workflow performance.

### Key Features

1. **Three-Phase Workflow**:
   - **Phase 1** (n=3): Each A task triggers 3 B tasks
   - **Phase 2** (n=8): Each A task triggers 8 B tasks
   - **Phase 3** (n=1): Each A task triggers 1 B task

2. **Dynamic Instance Distribution**:
   - Instances are redistributed between schedulers to match task ratio
   - Phase 1 (1:3 ratio): 4 instances on A, 12 on B
   - Phase 2 (1:8 ratio): 2 instances on A, 14 on B
   - Phase 3 (1:1 ratio): 8 instances on A, 8 on B

3. **Safe Migration Process**:
   - Instances are drained before migration (no new tasks assigned)
   - Existing tasks complete normally
   - 50ms delay before re-registration
   - Continuous task submission during migration

4. **Comprehensive Metrics**:
   - Per-phase latency and throughput statistics
   - Migration timing and success rates
   - Instance utilization before/during/after migration

## Architecture

### Components

```
┌─────────────┐
│ Predictor   │ (port 8101)
│  Service    │
└─────────────┘
       │
       ├──────────┬──────────┐
       │          │          │
 ┌─────▼────┐ ┌──▼──────┐ ┌─▼─────────┐
 │Scheduler │ │Scheduler│ │  16       │
 │    A     │ │    B    │ │Instances  │
 │(port 8100)│ │(port 8200)│ │(8210-8225)│
 └──────────┘ └─────────┘ └───────────┘
     │            │             │
     └────────────┴─────────────┘
              Workflow
      A tasks → n × B tasks
```

### Phase Transitions with Migration

**Phase 1 → Phase 2 Transition:**
- Move 2 instances from Scheduler A to Scheduler B
- Instances `instance-000` and `instance-001` migrate

**Phase 2 → Phase 3 Transition:**
- Move 6 instances from Scheduler B to Scheduler A
- Instances `instance-004` through `instance-009` migrate

### Migration Workflow

```
1. Submit workflows until phase target reached
2. Trigger migration controller
3. For each instance to migrate:
   ┌──────────────────────────────────────┐
   │ POST /instance/drain                 │
   │   → Mark instance as DRAINING        │
   │   → Stop accepting new tasks         │
   └──────────────────────────────────────┘
                  │
   ┌──────────────▼───────────────────────┐
   │ Poll GET /instance/drain/status      │
   │   → Wait for pending_tasks = 0       │
   │   → Monitor drain progress           │
   └──────────────────────────────────────┘
                  │
   ┌──────────────▼──────────────────────────────────┐
   │ Wait 50ms delay (Simulation of startup overhead)│
   └─────────────────────────────────────────────────┘
                  │
   ┌──────────────▼───────────────────────┐
   │ POST /instance/register              │
   │   → Register to new scheduler        │
   └──────────────────────────────────────┘
4. Continue submitting next phase workflows
```

## Prerequisites

### 1. Scheduler with Safe Removal API

**Important:** This experiment requires the scheduler to have the safe instance removal API implemented. The following endpoints must be available:

- `POST /instance/drain` - Start draining an instance
- `GET /instance/drain/status` - Check drain status
- `POST /instance/remove` - Safely remove drained instance

These APIs have been added to `scheduler/src/api.py` as part of this experiment.

### 2. Services Running

Start all services before running the experiment:

```bash
./start_all_services.sh
```

This starts:
- Predictor service (port 8101)
- Scheduler A (port 8100)
- Scheduler B (port 8200)
- 16 instances (ports 8210-8225)

Initial distribution (Phase 1):
- 4 instances on Scheduler A (instance-000 to instance-003)
- 12 instances on Scheduler B (instance-004 to instance-015)

## Usage

### Comparison Experiments (Recommended)

**Run all 6 configurations:**

```bash
uv run python3 run_comparison_experiments.py \
    --num-workflows-per-phase 100 \
    --qps 2.0
```

**Analyze results:**

```bash
uv run python3 analyze_comparison_results.py \
    --results-dir results \
    --pattern "exp08_*.json"
```

### Single Experiment

**With dynamic migration (default):**

```bash
uv run python3 test_dynamic_migration.py \
    --strategy min_time \
    --num-workflows-per-phase 100 \
    --enable-migration
```

**Without migration (static distribution):**

```bash
uv run python3 test_dynamic_migration.py \
    --strategy probabilistic \
    --num-workflows-per-phase 100 \
    --disable-migration
```

### Command-Line Arguments

**test_dynamic_migration.py:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-workflows-per-phase` | 10 | Number of workflows per phase |
| `--strategy` | min_time | Strategy: min_time, probabilistic, round_robin |
| `--qps` | 2.0 | A task submission rate (queries per second) |
| `--enable-migration` | True | Enable dynamic instance migration |
| `--disable-migration` | - | Disable migration (static distribution) |
| `--total-instances` | 16 | Total number of instances |
| `--instance-start-port` | 8210 | Starting port for instances |

**run_comparison_experiments.py:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-workflows-per-phase` | 100 | Workflows per phase |
| `--qps` | 2.0 | QPS for task submission |
| `--output-dir` | results | Output directory |

**analyze_comparison_results.py:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | results | Results directory |
| `--pattern` | exp08_*.json | File pattern to match |
| `--output-prefix` | comparison_analysis | Output file prefix |

### Service Management

**Start services:**
```bash
./start_all_services.sh
```

**Stop services:**
```bash
./stop_all_services.sh
```

**Check service status:**
```bash
# Check if services are running
curl http://localhost:8100/health  # Scheduler A
curl http://localhost:8200/health  # Scheduler B
curl http://localhost:8101/health  # Predictor
```

## Output

### Comparison Experiments Output

**Individual experiment results:**
```
results/
├── exp08_migration_min_time_20250103_120000.json
├── exp08_static_min_time_20250103_120530.json
├── exp08_migration_probabilistic_20250103_121100.json
├── exp08_static_probabilistic_20250103_121630.json
├── exp08_migration_round_robin_20250103_122200.json
└── exp08_static_round_robin_20250103_122730.json
```

**Batch run summary:**
```
results/comparison_run_20250103_123000.json
```

**Analysis reports:**
```
results/
├── comparison_analysis_20250103_123500.json  # Structured data
└── comparison_analysis_20250103_123500.md    # Human-readable report
```

The Markdown report includes:
- Workflow latency comparison (all phases)
- Migration overhead analysis
- Throughput metrics
- Load balancing effectiveness (CV analysis)
- Migration impact comparison (migration vs static)

### Console Output

The script provides real-time progress updates:

```
========================================
Experiment 08: Dynamic Instance Migration
========================================

Configuration:
  - Strategy: min_time
  - Workflows per phase: 100
  - A task QPS: 2.0
  - Total instances: 16

Phase 1: n=3 (A:B ratio 1:3)
  - Scheduler A: 4 instances
  - Scheduler B: 12 instances
  - Submitting 100 workflows...

[Progress bar and stats]

Phase 1 → Phase 2 Migration:
  - Migrating 2 instances (A → B)
  - Draining instance-000...
  - Draining instance-001...
  - Waiting for tasks to complete...
  - Migration completed in 2345.6ms

Phase 2: n=8 (A:B ratio 1:8)
  - Scheduler A: 2 instances
  - Scheduler B: 14 instances
  - Submitting 100 workflows...

[...]

Results Summary:
================

Per-Phase Statistics:
  Phase 1 (n=3):
    - Workflows: 100
    - Avg latency: 5432.1ms
    - Throughput: 1.85 workflows/s

  Phase 2 (n=8):
    - Workflows: 100
    - Avg latency: 12345.6ms
    - Throughput: 0.81 workflows/s

  Phase 3 (n=1):
    - Workflows: 100
    - Avg latency: 3210.4ms
    - Throughput: 3.11 workflows/s

Migration Statistics:
  Phase 1 → 2:
    - Instances migrated: 2
    - Success rate: 100%
    - Avg drain time: 1234.5ms
    - Avg total time: 2345.6ms

  Phase 2 → 3:
    - Instances migrated: 6
    - Success rate: 100%
    - Avg drain time: 2345.6ms
    - Avg total time: 3456.7ms
```

### Results Directory

Results are saved to `results/` directory:

```
results/
└── results_YYYYMMDD_HHMMSS_min_time.json
```

JSON structure:
```json
{
  "experiment_config": {
    "strategy": "min_time",
    "num_workflows_per_phase": 100,
    "qps": 2.0,
    "total_instances": 16
  },
  "phases": [
    {
      "phase_id": 1,
      "fanout": 3,
      "num_workflows": 100,
      "workflows": [...],
      "statistics": {...}
    },
    ...
  ],
  "migrations": [
    {
      "from_phase": 1,
      "to_phase": 2,
      "migrations": [...],
      "statistics": {...}
    },
    ...
  ],
  "overall_statistics": {...}
}
```

## Key Differences from Experiment 04

| Aspect | Experiment 04 | Experiment 08 |
|--------|---------------|---------------|
| Phases | Single continuous run | 3 distinct phases (n=3, 8, 1) |
| Instance distribution | Static throughout | Dynamic, changes per phase |
| Migration | None | Safe migration between phases |
| Fanout | Random (3-8) | Fixed per phase |
| Instance count | 16 total (10+6) | 16 total (dynamic split) |
| Focus | Baseline performance | Migration impact analysis |

## Monitoring

### View Scheduler Logs

```bash
# Scheduler A
tail -f logs/scheduler_a.log

# Scheduler B
tail -f logs/scheduler_b.log
```

### View Instance Logs

```bash
# Specific instance
tail -f logs/instance_8210.log

# All instances
tail -f logs/instance_*.log
```

### Check Instance Registration

```bash
# Scheduler A
curl http://localhost:8100/instance/list | jq

# Scheduler B
curl http://localhost:8200/instance/list | jq
```

### Monitor Migration Status

During experiment execution, check drain status:

```bash
curl "http://localhost:8100/instance/drain/status?instance_id=instance-000" | jq
```

## Troubleshooting

### Migration Fails

**Symptom:** Instances fail to drain or re-register

**Solutions:**
1. Check instance logs for errors
2. Verify scheduler API is accessible
3. Ensure instances have no hanging tasks
4. Increase `max_drain_wait_ms` if timeouts occur

### Task Submission Errors

**Symptom:** "No available instance" errors during migration

**Cause:** Too many instances draining simultaneously

**Solution:** This is expected briefly during migration; the experiment retries automatically

### Inconsistent Results

**Symptom:** Wide variance in migration times

**Causes:**
- System load fluctuations
- Workload characteristics
- Instance queue lengths at migration time

**Mitigation:**
- Run multiple trials
- Control system load
- Use consistent QPS settings

## Cleanup

Stop all services:

```bash
./stop_all_services.sh
```

This kills:
- All instance processes
- Both scheduler processes
- Predictor service

## Files and Scripts

### Main Scripts

- **`test_dynamic_migration.py`** - Single experiment runner (supports both migration and static modes)
- **`run_comparison_experiments.py`** - Batch runner for all 6 configurations
- **`analyze_comparison_results.py`** - Result analyzer and report generator
- **`migration_controller.py`** - Instance migration orchestration
- **`workload_generator.py`** - Task execution time distributions

### Service Scripts

- **`start_all_services.sh`** - Start all services (predictor, schedulers, instances)
- **`stop_all_services.sh`** - Stop all services

### Documentation

- **`README.md`** - This file (overview and basic usage)
- **`COMPARISON_EXPERIMENTS_GUIDE.md`** - Complete guide for comparison experiments

### Testing

- **`test_scripts_functionality.py`** - Functionality tests for comparison scripts

## Related Experiments

- **Experiment 04**: Baseline multi-model workflow (no migration)
- **Experiment 05**: Parallel fanout workflow
- **Experiment 06**: Dynamic fanout with merge

## Key Improvements in This Experiment

### Comparison Testing Framework

This experiment adds a comprehensive comparison testing framework:

1. **6 Configuration Matrix**: 3 strategies × 2 modes (migration/static)
2. **Automated Batch Execution**: Run all configurations sequentially
3. **Multi-Dimensional Analysis**:
   - Workflow latency statistics
   - Migration overhead measurement
   - Throughput comparison
   - Load balancing effectiveness
   - Migration impact evaluation
4. **Comprehensive Reporting**: JSON + Markdown reports with detailed tables

### Safe Instance Migration

Implements production-ready instance migration:
- Drain-based approach (no task loss)
- Continuous task submission during migration
- Health checks and retry logic
- Detailed migration metrics

## References

- Scheduler safe removal API: `scheduler/src/api.py`
- Migration controller: `migration_controller.py`
- Workload generator: `workload_generator.py`
- Comparison framework: `run_comparison_experiments.py`, `analyze_comparison_results.py`
