# Experiment 09: Unified Multi-Model Workflow

This experiment serves as a unified entry point for experiments 04-07, allowing users to run different workflow patterns with a single command-line interface.

## Overview

This unified experiment provides a standardized way to execute and compare different multi-model workflow patterns:

| Experiment | Name | Workflow Pattern | Threads | Description |
|------------|------|------------------|---------|-------------|
| 04-ocr | OCR Pipeline | 1→n (parallel) | 4 | Each A task generates n B tasks that execute in parallel |
| 05-t2vid | T2Vid Pipeline | 1→n (sequential) | 4 | Each A task generates n B tasks that execute sequentially |
| 06-dr_simple | Data Reduction (Simple) | 1→n→1 (merge) | 6 | A task → n parallel B tasks → merge A task |
| 07-dr | Data Reduction (Pipeline) | 1→n→n→1 (B1/B2) | 7 | A task → n B1 tasks → n pipelined B2 tasks → merge |

## Architecture

### Common Components

- **common.py**: Shared data structures (WorkflowState, TaskRecord), rate limiter, utility functions, and metrics calculation
- **workload_generator.py**: Unified workload generation for all experiments
  - Bimodal distributions for A and B tasks (50% fast [1-3s], 50% slow [7-10s])
  - B1/B2 split distributions for experiment 07 (B1: slow only, B2: fast only)
  - Uniform fanout distribution (3-8 B tasks per A task)

### Experiment-Specific Workflows

Each experiment follows the standard from Experiment 04 with variations:

**Experiment 04 (04-ocr)**: 1→n Parallel
- Thread 1: PoissonTaskSubmitter - Submits A tasks with QPS control
- Thread 2: ATaskReceiver - Receives A completions, submits **all** B tasks immediately
- Thread 3: BTaskReceiver - Receives B completions, updates workflow state
- Thread 4: WorkflowMonitor - Monitors workflow completion

**Experiment 05 (05-t2vid)**: 1→n Sequential
- Same thread structure as 04
- Thread 2 submits only **first** B task
- Thread 3 chains B task submissions: each B completion triggers next B submission

**Experiment 06 (06-dr_simple)**: 1→n→1 Merge
- Extends experiment 04 with 2 additional daemon threads
- Thread 5: MergeTaskSubmitter - Submits merge task when all B tasks complete
- Thread 6: MergeTaskReceiver - Receives merge task completions
- Workflow completes when merge task completes (not when B tasks complete)

**Experiment 07 (07-dr)**: 1→n→n→1 B1/B2 Split
- B stage split into B1 (slow) and B2 (fast) substages
- Thread 3: B1TaskReceiver - Each B1 completion triggers paired B2 submission (pipelined)
- Thread 4: B2TaskReceiver - Receives B2 completions
- Threads 5-7: Same as experiment 06 for merge handling

## Installation

### Prerequisites

- Python 3.8+
- uv (Python package manager)
- Docker (for sleep_model instances)
- Predictor, Scheduler, and Instance services running

### Setup

```bash
# Install dependencies
uv sync

# Or using pip
pip install -r requirements.txt
```

## Usage

### Start Services

```bash
# Start all services with default configuration (10 Group A instances, 6 Group B instances)
./start_all_services.sh

# Or with custom instance counts
./start_all_services.sh 12 8  # 12 instances in Group A, 8 in Group B
```

### Run Experiments

#### Run a Specific Experiment

```bash
# Experiment 04-ocr (parallel workflow)
python unified_workflow.py --experiment 04-ocr --num-workflows 100 --qps 8.0

# Experiment 05-t2vid (sequential workflow)
python unified_workflow.py --experiment 05-t2vid --num-workflows 100 --qps 8.0

# Experiment 06-dr_simple (with merge)
python unified_workflow.py --experiment 06-dr_simple --num-workflows 100 --qps 8.0

# Experiment 07-dr (B1/B2 split with merge)
python unified_workflow.py --experiment 07-dr --num-workflows 100 --qps 8.0
```

#### Run All Experiments Sequentially

```bash
# Run all four experiments with the same configuration
python unified_workflow.py --experiment all --num-workflows 100 --qps 8.0
```

#### Advanced Options

```bash
# Run with specific scheduling strategies
python unified_workflow.py --experiment 04-ocr \
    --num-workflows 100 \
    --strategies min_time round_robin

# Run with global QPS limit (controls both A and B task submissions)
python unified_workflow.py --experiment 06-dr_simple \
    --num-workflows 100 \
    --gqps 10.0

# Run with warmup tasks (20% warmup, excluded from statistics)
python unified_workflow.py --experiment 05-t2vid \
    --num-workflows 100 \
    --warmup 0.2

# Run with custom random seed
python unified_workflow.py --experiment 07-dr \
    --num-workflows 100 \
    --seed 12345

# Run all experiments with custom configuration
python unified_workflow.py --experiment all \
    --num-workflows 50 \
    --qps 10.0 \
    --strategies probabilistic min_time \
    --seed 42
```

### Stop Services

```bash
# Stop all services and clean up
./stop_all_services.sh
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment` | str | **required** | Experiment to run: `04-ocr`, `05-t2vid`, `06-dr_simple`, `07-dr`, or `all` |
| `--num-workflows` | int | 100 | Number of workflows to execute per strategy |
| `--qps` | float | 8.0 | Target QPS for A task submission |
| `--gqps` | float | None | Global QPS limit for both A and B tasks (overrides `--qps`) |
| `--warmup` | float | 0.0 | Warmup task ratio (0.0-1.0), excluded from statistics |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--strategies` | list | all three | Scheduling strategies: `min_time`, `round_robin`, `probabilistic` |

## Output

### Console Output

Each experiment prints:
- Real-time progress updates
- Task submission and completion statistics
- Workflow completion metrics
- Per-strategy comparison table

### Results Files

Results are saved to `results/results_workflow_<experiment>_<timestamp>.json`:

```json
{
  "experiment": "09.multi_model_unified/04-ocr",
  "timestamp": "2025-01-15T10:30:00",
  "config": {
    "num_workflows": 100,
    "qps_a": 8.0,
    "seed": 42,
    "workload_a": {...},
    "workload_b": {...},
    "fanout": {...}
  },
  "results": [
    {
      "strategy": "min_time",
      "a_tasks": {...},
      "b_tasks": {...},
      "workflows": {...}
    },
    ...
  ]
}
```

## Workflow Timing Definitions

- **Experiment 04**: `workflow_time = A_submit → max(B_complete_times)`
- **Experiment 05**: `workflow_time = A_submit → last_B_complete` (sequential: sum of all B times)
- **Experiment 06**: `workflow_time = A_submit → merge_complete`
- **Experiment 07**: `workflow_time = A_submit → merge_complete` (includes B1→B2 pipeline)

## Expected Performance

Based on Experiment 04 standard configuration:

| Metric | Experiment 04 (Parallel) | Experiment 05 (Sequential) | Experiment 06 (Merge) | Experiment 07 (B1/B2) |
|--------|-------------------------|---------------------------|----------------------|----------------------|
| A task avg | ~2.5s | ~2.5s | ~2.5s | ~2.5s |
| B task avg | ~5.2s | ~5.2s | ~5.2s | B1: ~8.7s, B2: ~2.0s |
| Workflow avg | ~11.2s | ~19.3s | ~12.5s | ~13-14s |
| Total tasks | ~650 | ~650 | ~2250 | ~1700 |

## Troubleshooting

### Services Not Starting

```bash
# Check if ports are already in use
netstat -tuln | grep -E '8100|8101|8200|8210|8300'

# Check service logs
tail -f logs/predictor.log
tail -f logs/scheduler-a.log
tail -f logs/scheduler-b.log
```

### Experiment Fails to Import

Ensure the original experiment directories (04, 05, 06, 07) exist and contain their respective workflow files:
- `04.multi_model_workflow_dynamic/test_dynamic_workflow.py`
- `05.multi_model_workflow_dynamic_parallel/test_dynamic_workflow.py`
- `06.multi_model_workflow_dynamic_merge/test_dynamic_workflow_merge.py`
- `07.multi_model_workflow_dynamic_merge_2/test_dynamic_workflow_merge_2.py`

### Task Submission Failures

```bash
# Verify all instances are registered
curl http://localhost:8100/instance/query | python -m json.tool
curl http://localhost:8200/instance/query | python -m json.tool

# Check instance health
curl http://localhost:8210/health  # First Group A instance
curl http://localhost:8300/health  # First Group B instance
```

## File Structure

```
09.multi_model_unified/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── common.py                    # Shared utilities and data structures
├── workload_generator.py        # Unified workload generation
├── unified_workflow.py          # Main entry point
├── start_all_services.sh        # Start all services
├── stop_all_services.sh         # Stop all services
├── logs/                        # Service logs (created on startup)
└── results/                     # Experiment results (created on run)
```

## Key Differences from Individual Experiments

This unified experiment:
1. ✅ Provides a single command-line interface for all workflow patterns
2. ✅ Maintains exact compatibility with original experiment implementations
3. ✅ Uses shared utility functions and data structures
4. ✅ Follows Experiment 04 as the standard reference
5. ✅ Supports running all experiments sequentially with consistent parameters
6. ✅ Standardizes thread counts and workflow patterns across experiments

## References

- Experiment 04: `../04.multi_model_workflow_dynamic/`
- Experiment 05: `../05.multi_model_workflow_dynamic_parallel/`
- Experiment 06: `../06.multi_model_workflow_dynamic_merge/`
- Experiment 07: `../07.multi_model_workflow_dynamic_merge_2/`

## License

This experiment is part of the swarmpilot-refresh project.
