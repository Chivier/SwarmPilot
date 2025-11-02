# Experiment 03: Multi-Model 1-to-1 Workflow Dependencies

## Overview

This experiment extends experiment 02 by introducing **1-to-1 task dependencies** between two task types (A and B). Each A-type task, upon completion, triggers exactly one B-type task, forming a workflow chain.

The key difference from experiment 02:
- **Experiment 02**: Independent A and B tasks with separate QPS control
- **Experiment 03**: A tasks trigger B tasks (1-to-1 dependency), QPS control only on A tasks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Predictor (8101)                       │
│                    Shared by both schedulers                 │
└────────────────────┬───────────────────┬────────────────────┘
                     │                   │
         ┌───────────▼──────────┐   ┌───▼────────────────────┐
         │  Scheduler A (8100)  │   │  Scheduler B (8200)    │
         │  Handles A tasks     │   │  Handles B tasks       │
         └──────────┬───────────┘   └───┬────────────────────┘
                    │                   │
         ┌──────────▼──────────┐   ┌───▼────────────────────┐
         │ Group A Instances   │   │ Group B Instances      │
         │ (ports 8210-82xx)   │   │ (ports 8300-83xx)      │
         │ N1 instances        │   │ N2 instances           │
         └─────────────────────┘   └────────────────────────┘

Workflow Flow:
  1. A task submitted with Poisson QPS control
  2. A task executes on Group A instance
  3. A task completes → WebSocket notification
  4. B task immediately submitted to Scheduler B
  5. B task executes on Group B instance
  6. B task completes → Workflow complete
```

## Task Types and Workloads

### A Tasks (Bimodal Distribution)
- **Distribution**: Two distinct peaks in execution time
  - Left peak: 1-3 seconds (mean=2.0s, std=0.4s, 50% of tasks)
  - Right peak: 7-10 seconds (mean=8.5s, std=0.6s, 50% of tasks)
- **Submission**: Poisson process with configurable QPS (default: 8.0)
- **Scheduler**: Scheduler A (port 8100)
- **Instances**: Group A (N1 instances, default: 10)

### B Tasks (Pareto Long-Tail Distribution)
- **Distribution**: Power-law distribution with long tail
  - Range: 1-10 seconds
  - ~80% of tasks complete quickly
  - ~20% of tasks take significantly longer
  - Alpha parameter: 1.5
- **Submission**: Triggered by A task completion (no direct QPS control)
- **Scheduler**: Scheduler B (port 8200)
- **Instances**: Group B (N2 instances, default: 6)

## Workflow Tracking

Each workflow consists of one A task and one B task:

### Task ID Format
- **A task**: `task-A-{strategy}-workflow-{i:04d}-A`
- **B task**: `task-B-{strategy}-workflow-{i:04d}-B`
- **Workflow ID**: `wf-{strategy}-{i:04d}`

### Metrics Tracked

1. **A Task Metrics** (Submit → Complete):
   - Average, Median, P95, P99 completion times
   - Number completed/failed
   - Task distribution across instances

2. **B Task Metrics** (Submit → Complete):
   - Average, Median, P95, P99 completion times
   - Number completed/failed
   - Task distribution across instances

3. **Workflow Metrics** (A Submit → B Complete):
   - Average, Median, P50, P95, P99 workflow completion times
   - Number of completed workflows
   - End-to-end latency analysis

## Scheduling Strategies

Three strategies are tested:

1. **round_robin**: Evenly distributes tasks across instances in a round-robin fashion
2. **min_time**: Assigns tasks to the instance with minimum predicted completion time
3. **probabilistic**: Balances load using probabilistic selection based on predicted times

## Implementation Details

### Thread Architecture

The experiment uses a 4-threaded design:

1. **Thread 1**: WebSocket receiver for Scheduler A
   - Receives A task completion events
   - **Immediately submits corresponding B tasks** upon A completion
   - Tracks B task submissions

2. **Thread 2**: WebSocket receiver for Scheduler B
   - Receives B task completion events
   - Tracks workflow completion

3. **Thread 3**: Poisson task submitter for A tasks
   - Submits A tasks following Poisson process (configurable QPS)
   - Uses exponential inter-arrival times

4. **Thread 4**: Not used (B tasks submitted by Thread 1)

### Key Classes

#### WorkflowTaskData
Extends TaskData with workflow tracking:
```python
@dataclass
class WorkflowTaskData:
    task_id: str
    workflow_id: str  # Links A and B tasks
    task_type: str    # "A" or "B"
    sleep_time: float
    exp_runtime: float
```

#### TaskRecord
Tracks execution with workflow context:
```python
@dataclass
class TaskRecord:
    task_id: str
    workflow_id: str
    task_type: str
    sleep_time: float
    exp_runtime: float
    submit_time: Optional[float]
    complete_time: Optional[float]
    status: Optional[str]
    assigned_instance: Optional[str]
    result: Optional[Dict]
    error: Optional[str]
    execution_time_ms: Optional[float]
```

#### WebSocketResultReceiver (Enhanced)
Now includes B task submission capability:
```python
def __init__(self, name: str, ws_url: str, task_ids: List[str], result_queue: Queue,
             scheduler_b_url: Optional[str] = None, b_task_generator=None):
    # When scheduler_b_url and b_task_generator are provided,
    # this receiver will submit B tasks upon A task completion
```

## Usage

### Quick Start

1. **Start all services**:
```bash
cd experiments/03.multi_model_1_to_1
./start_all_services.sh
```

2. **Run experiment with default settings**:
```bash
python test_dual_scheduler.py
```

3. **Stop all services**:
```bash
./stop_all_services.sh
```

### Configuration Options

```bash
python test_dual_scheduler.py \
  --n1 10 \              # Number of Group A instances
  --n2 6 \               # Number of Group B instances
  --qps1 8.0 \           # QPS for A tasks (B tasks follow A completions)
  --num-workflows 100 \  # Number of workflows per strategy
  --strategies min_time round_robin probabilistic
```

### Example Output

```
=================================================================
03.multi_model_1_to_1 Experiment
=================================================================
Configuration:
  Group A: 10 instances (Scheduler A)
  Group B: 6 instances (Scheduler B)
  A Tasks QPS: 8.0
  Workflows per strategy: 100
  Strategies: min_time, round_robin, probabilistic
  Workflow: Each A task → triggers one B task
=================================================================

...

=================================================================
Results for MIN_TIME
=================================================================

[A Tasks - Scheduler A]
  Total tasks:              100
  Submitted:                100
  Completed:                100
  Failed:                   0
  Avg completion time:      2.456s
  Median completion time:   2.123s
  P95 completion time:      5.678s
  P99 completion time:      7.234s

[B Tasks - Scheduler B]
  Total tasks:              100
  Submitted:                100
  Completed:                100
  Failed:                   0
  Avg completion time:      3.234s
  Median completion time:   2.567s
  P95 completion time:      8.123s
  P99 completion time:      9.456s

[Workflows (A submit → B complete)]
  Total workflows:          100
  Completed workflows:      100
  Avg workflow time:        5.690s
  Median workflow time:     4.690s
  P50 workflow time:        4.690s
  P95 workflow time:        13.801s
  P99 workflow time:        16.690s

...

=================================================================
Strategy Comparison - Workflow Metrics
=================================================================
Strategy             A Avg        B Avg        Workflow Avg    Workflow P95
--------------------------------------------------------------------------------
min_time                  2.456s       3.234s          5.690s         13.801s
round_robin               3.123s       4.567s          7.690s         15.234s
probabilistic             2.789s       3.890s          6.679s         14.123s
================================================================================
```

## Results

Results are saved to `results/results_workflow_YYYYMMDD_HHMMSS.json`:

```json
{
  "experiment": "03.multi_model_1_to_1",
  "timestamp": "2025-01-15T10:30:45.123456",
  "config": {
    "n1": 10,
    "n2": 6,
    "qps1": 8.0,
    "num_workflows": 100,
    "workload_a": {
      "type": "Bimodal Distribution",
      "description": "Two distinct peaks...",
      "mean": 5.0,
      "std": 3.0,
      "min": 1.0,
      "max": 10.0
    },
    "workload_b": {
      "type": "Pareto Distribution",
      "description": "Power-law distribution...",
      "mean": 2.5,
      "std": 2.0,
      "min": 1.0,
      "max": 10.0
    }
  },
  "results": [
    {
      "strategy": "min_time",
      "qps": 8.0,
      "num_workflows": 100,
      "a_tasks": {
        "num_tasks": 100,
        "num_submitted": 100,
        "num_completed": 100,
        "num_failed": 0,
        "avg_completion_time": 2.456,
        "median_completion_time": 2.123,
        "p95_completion_time": 5.678,
        "p99_completion_time": 7.234
      },
      "b_tasks": {
        "num_tasks": 100,
        "num_submitted": 100,
        "num_completed": 100,
        "num_failed": 0,
        "avg_completion_time": 3.234,
        "median_completion_time": 2.567,
        "p95_completion_time": 8.123,
        "p99_completion_time": 9.456
      },
      "workflows": {
        "num_completed": 100,
        "avg_completion_time": 5.690,
        "median_completion_time": 4.690,
        "p50_completion_time": 4.690,
        "p95_completion_time": 13.801,
        "p99_completion_time": 16.690
      },
      "submission_time": 12.5,
      "actual_qps": 8.0
    }
  ]
}
```

## Key Differences from Experiment 02

| Aspect | Experiment 02 | Experiment 03 |
|--------|---------------|---------------|
| **Task Relationship** | Independent A and B tasks | A tasks trigger B tasks (1-to-1) |
| **QPS Control** | Both A and B have QPS control | Only A tasks have QPS control |
| **B Task Submission** | Separate Poisson submitter | Triggered by A completion |
| **Task ID Format** | `task-{scheduler}-{strategy}-{i}` | `task-{type}-{strategy}-workflow-{i}-{type}` |
| **Metrics** | Separate A and B metrics | A, B, and workflow metrics |
| **Thread 4** | B task submitter | Not used |
| **Results File** | `results_dual_*.json` | `results_workflow_*.json` |

## Analysis Goals

This experiment helps answer:

1. **Dependency Impact**: How do task dependencies affect overall workflow latency?
2. **Cascading Effects**: Do A task scheduling decisions impact B task performance?
3. **Strategy Comparison**: Which strategy minimizes end-to-end workflow time?
4. **Resource Utilization**: How efficiently are Group B instances utilized when driven by A completions?
5. **Tail Latency**: How do P95/P99 workflow times compare across strategies?

## Troubleshooting

### Common Issues

1. **WebSocket connection failed**
   - Ensure schedulers are running: `./start_all_services.sh`
   - Check scheduler health: `curl http://localhost:8100/health`

2. **B tasks not being submitted**
   - Check WebSocket receiver A logs for submission errors
   - Verify Scheduler B is accepting tasks: `curl http://localhost:8200/health`

3. **Incomplete workflows**
   - Check if A tasks are completing successfully
   - Verify B task submission in logs
   - Increase timeout in `collect_results()` if needed

4. **Port conflicts**
   - Verify ports 8100, 8101, 8200, 8210-82xx, 8300-83xx are available
   - Check `docker ps` for running containers

### Debug Mode

Add debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Dependencies

See `requirements.txt`:
- numpy>=1.21.0
- requests>=2.26.0
- websockets>=10.0
- scipy>=1.7.0

## References

- Experiment 01: [Quick Start Up](../01.quick-start-up/)
- Experiment 02: [Multi-Model No Dependencies](../02.multi_model_no_dep/)
- Scheduler API: See scheduler source code for API details
- Predictor API: See predictor source code for prediction interface
