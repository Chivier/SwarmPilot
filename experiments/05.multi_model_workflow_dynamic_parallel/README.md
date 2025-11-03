# Experiment 05: Multi-Model Workflow with Dynamic Fanout - Sequential B Tasks

## Overview

This experiment tests workflow dependencies where **each A task generates a variable number (n) of B tasks** that execute **sequentially** across two independent schedulers. Unlike Experiment 04 where B tasks run in parallel, here **each B task must complete before the next B task is submitted**. A workflow is complete only when **all B tasks** for that workflow have finished.

### Key Features

- **Dynamic fanout**: Each A task generates 3-8 B tasks (uniform distribution)
- **Sequential B task execution**: B tasks execute one at a time - next B submitted only after previous B completes
- **Four-thread architecture**: Specialized threads for submission, reception, and monitoring
- **Pre-calculated task IDs**: All task IDs generated upfront for WebSocket subscription
- **WebSocket-based result collection**: Real-time, push-based task completion events
- **Three scheduling strategies**: min_time, round_robin, probabilistic

### Differences from Experiment 04 (Parallel B Tasks)

| Aspect | Experiment 04 | Experiment 05 |
|--------|--------------|--------------|
| **B task execution** | All B tasks submitted immediately | B tasks submitted one at a time |
| **Submission trigger** | A task completion | Previous B task completion (after first) |
| **Workflow completion time** | A_time + max(B_times) | A_time + Σ(B_times) |
| **Concurrency** | High (all B tasks run in parallel) | Low (1 B task at a time per workflow) |
| **State tracking** | Simple counter | Counter + next_b_task_index |

---

## Architecture

### Four-Thread Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Experiment 05 Architecture                        │
│                     (Sequential B Task Execution)                    │
└─────────────────────────────────────────────────────────────────────┘

Thread 1: A Task Submitter (Poisson)
   │
   ├──[QPS=8]──> Scheduler A ──> Instance Group A (10 instances)
   │                 │
   │                 │ (WebSocket completion events)
   │                 ▼
   │            Thread 2: A Result Receiver
   │                 │
   │                 ├──[Extract n]──> Generate n B task data
   │                 │
   │                 └──[Submit B_0 ONLY]──> Scheduler B ──> Instance Group B (6 instances)
   │                                              │
   │                                              │ (WebSocket completion events)
   │                                              ▼
   │                                    Thread 3: B Result Receiver
   │                                         │
   │                                         ├──[B_i completes]
   │                                         │
   │                                         ├──[Submit B_{i+1}]──┐
   │                                         │                     │
   │                                         │   ┌────────────────┘
   │                                         │   │ (Sequential loop)
   │                                         │   └──> Scheduler B
   │                                         │
   │                                         └──[If all B done]──> Queue
   │                                                                │
   │                                                                ▼
   └────────────────────────────────────────────────> Thread 4: Workflow Monitor
                                                                   │
                                                                   └──> Calculate stats
```

### Thread Responsibilities

| Thread | Name | Input | Output | Purpose |
|--------|------|-------|--------|---------|
| 1 | A Task Submitter | A task data, QPS | Submitted A tasks | Submit A tasks with Poisson inter-arrival times |
| 2 | A Result Receiver | A completion events | First B task submission (B_0) | Receive A results, submit **first** B task only |
| 3 | B Result Receiver | B completion events | Next B task submission (B_{i+1}) | Track B completion, submit **next** B task sequentially |
| 4 | Workflow Monitor | Workflow completion events | Aggregated statistics | Calculate workflow metrics, detect experiment end |

---

## Workflow State Tracking

Each workflow maintains state to track completion:

```python
@dataclass
class WorkflowState:
    workflow_id: str                    # e.g., "wf-min_time-0042"
    a_task_id: str                      # e.g., "task-A-min_time-workflow-0042-A"
    b_task_ids: List[str]               # e.g., ["task-B-...-B-00", ..., "task-B-...-B-04"]
    total_b_tasks: int                  # e.g., 5
    completed_b_tasks: int              # Counter: 0 → 5
    next_b_task_index: int              # NEW: Track which B task to submit next (0 = first)

    # Timestamps
    a_submit_time: float
    a_complete_time: float
    b_complete_times: Dict[str, float]  # Map: task_id → complete_time
    workflow_complete_time: float       # max(b_complete_times)
```

### Sequential Workflow Logic

```python
# Thread 2: After A completes, submit ONLY first B task
def submit_first_b_task(workflow_id: str):
    """Submit B_0 after A completes."""
    b_tasks = b_tasks_by_workflow[workflow_id]
    submit_b_task(b_tasks[0])  # Submit only B_0
    workflow.next_b_task_index = 1  # Next submit will be B_1

# Thread 3: After each B completes, submit next B task
def mark_b_task_complete(b_task_id: str, complete_time: float):
    """Thread 3 calls this for each B task completion."""
    self.b_complete_times[b_task_id] = complete_time
    self.completed_b_tasks += 1

    # [SEQUENTIAL MODE] Submit next B task if there is one
    if self.next_b_task_index < self.total_b_tasks:
        submit_b_task(b_tasks[self.next_b_task_index])  # Submit B_{i+1}
        self.next_b_task_index += 1

    # Check if workflow is complete
    if self.completed_b_tasks >= self.total_b_tasks:
        self.workflow_complete_time = max(self.b_complete_times.values())
        # Push completion event to Thread 4
```

---

## Task ID Scheme

### Pre-calculation Strategy

To enable WebSocket subscription before task submission (avoiding race conditions), all task IDs are pre-calculated:

```
A tasks (100):  task-A-{strategy}-workflow-{i:04d}-A
B tasks (variable): task-B-{strategy}-workflow-{i:04d}-B-{j:02d}

Example for workflow 42 with 5 B tasks (strategy=min_time):
  A: task-A-min_time-workflow-0042-A
  B: task-B-min_time-workflow-0042-B-00
     task-B-min_time-workflow-0042-B-01
     task-B-min_time-workflow-0042-B-02
     task-B-min_time-workflow-0042-B-03
     task-B-min_time-workflow-0042-B-04
```

### Total Task Count

```
A tasks per strategy: 100
B tasks per strategy: Σ(fanout[i]) for i in [0, 99]
  - Fanout ~ Uniform(3, 8)
  - Expected total: 100 * (3+8)/2 = 550 B tasks
  - Actual varies per run

Total across 3 strategies: 300 A + ~1650 B = ~1950 tasks
```

---

## Workload Generation

### A Tasks: Bimodal Distribution

```python
Distribution: 50% fast, 50% slow
  - Fast: 1-3 seconds (mean=2.0s, std=0.4s)
  - Slow: 7-10 seconds (mean=8.5s, std=0.6s)
```

### B Tasks: Bimodal Distribution

```python
Distribution: Same as A tasks
  - Fast: 1-3 seconds
  - Slow: 7-10 seconds

Total B tasks: Σ(fanout_values) ≈ 550 per strategy
```

### Fanout: Uniform Distribution

```python
Distribution: Uniform(3, 8) - each value equally likely
  - Each A task generates 3, 4, 5, 6, 7, or 8 B tasks
  - Mean: 5.5 B tasks per A task
  - Std: ~1.71
```

---

## Metrics

### Task-Level Metrics

**A Tasks** (Submit → Complete time):
- `num_submitted`: Total A tasks submitted
- `num_completed`: A tasks that completed successfully
- `avg_completion_time`: Average A task execution time
- `p95_completion_time`: 95th percentile A task time

**B Tasks** (Submit → Complete time):
- `num_submitted`: Total B tasks submitted across all workflows
- `num_completed`: B tasks that completed successfully
- `avg_completion_time`: Average B task execution time
- `p95_completion_time`: 95th percentile B task time

### Workflow-Level Metrics (KEY)

**Workflow Time** = A submit → Last B complete time

```
Workflow time = max(b_complete_times) - a_submit_time

Example:
  A submitted: t=0.0s
  A completed: t=2.5s
  B tasks submitted: t=2.5s
  B completions: [t=4.2s, t=5.1s, t=6.3s, t=5.8s, t=7.0s]
  Workflow time: 7.0 - 0.0 = 7.0s
```

**Metrics:**
- `num_completed`: Workflows with all B tasks completed
- `avg_workflow_time`: Average workflow completion time
- `median_workflow_time`: Median workflow time
- `p95_workflow_time`: 95th percentile workflow time
- `p99_workflow_time`: 99th percentile workflow time
- `fanout_distribution`: Histogram of B task counts per workflow
- `avg_fanout`: Average number of B tasks per workflow

---

## Usage

### 1. Start Services

```bash
cd experiments/05.multi_model_workflow_dynamic_parallel

# Start all services (predictor, schedulers, instances)
./start_all_services.sh

# Optional: Customize instance counts
N1=10 N2=6 ./start_all_services.sh
```

This starts:
- Predictor service (port 8101)
- Scheduler A (port 8100) + 10 instances (ports 8210-8219)
- Scheduler B (port 8200) + 6 instances (ports 8300-8305)

### 2. Run Experiment

```bash
# Activate virtual environment
source .venv/bin/activate  # or: uv venv && source .venv/bin/activate

# Install dependencies (if not already done)
uv pip install -r requirements.txt

# Run full experiment (default: all 3 strategies, 100 workflows each, QPS=8.0)
uv run python3 test_dynamic_workflow.py

# Run with custom parameters
uv run python3 test_dynamic_workflow.py --num-workflows 50 --qps 10.0

# Run single strategy only
uv run python3 test_dynamic_workflow.py --strategies min_time

# Small test run (10 workflows, one strategy)
uv run python3 test_dynamic_workflow.py --num-workflows 10 --strategies min_time

# Show all available options
uv run python3 test_dynamic_workflow.py --help
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-workflows` | int | 100 | Number of workflows to generate and execute per strategy |
| `--qps` | float | 8.0 | Target queries per second (QPS) for A task submission |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--strategies` | list | all three | Scheduling strategies to test (min_time, round_robin, probabilistic) |

**Note**: The experiment will generate and submit **exactly** the number of tasks specified:
- A tasks: `num_workflows` tasks (e.g., 100)
- B tasks: `sum(fanout_values)` tasks (e.g., ~550 for 100 workflows with fanout 3-8)

### 3. Monitor Progress

The experiment provides detailed logging:

```
[INFO] Starting Experiment 05: Multi-Model Workflow - Sequential B Tasks
[INFO] Generating workloads...
[INFO] Testing strategy: min_time
[INFO] Step 1: Clearing tasks from schedulers
[INFO] Step 7: Starting Thread 3 (B Task Receiver)
[INFO] Subscribed to 547 B tasks
[INFO] Step 10: Starting Thread 1 (A Task Submitter)
[INFO] Submitted 20/100 A tasks
[INFO] Submitting first B task (0/5) for workflow wf-min_time-0001 [SEQUENTIAL MODE]
[INFO] [SEQUENTIAL] Submitted B task 1/5 for workflow wf-min_time-0001: task-B-...
[INFO] [SEQUENTIAL] Submitted B task 2/5 for workflow wf-min_time-0001: task-B-...
[INFO] Workflows completed: 10/100
...
[INFO] All workflows completed!
[INFO] Results saved to: results/results_workflow_dynamic_20251102_143022.json
```

### 4. View Results

Results are saved to `results/results_workflow_dynamic_<timestamp>.json` and printed to console:

```
================================================================================
Results Summary: min_time
================================================================================

A Tasks:
  Generated:  100
  Submitted:  100
  Completed:  100
  Failed:     0
  Avg time:   5.23s
  Median:     4.81s
  P95:        9.45s

B Tasks:
  Generated:  547
  Submitted:  547
  Completed:  547
  Failed:     0
  Avg time:   5.12s
  Median:     4.73s
  P95:        9.21s

Workflows:
  Completed:  100
  Avg fanout: 5.5 B tasks per A task
  Avg time:   12.34s
  Median:     11.87s
  P95:        18.56s
  P99:        21.23s

Fanout Distribution:
  3 B tasks: 18 workflows (18.0%)
  4 B tasks: 15 workflows (15.0%)
  5 B tasks: 19 workflows (19.0%)
  6 B tasks: 17 workflows (17.0%)
  7 B tasks: 16 workflows (16.0%)
  8 B tasks: 15 workflows (15.0%)
================================================================================
```

### 5. Stop Services

```bash
./stop_all_services.sh
```

---

## Configuration

### Experiment Parameters

**Via Command-Line** (recommended):

```bash
# Specify parameters when running
uv run python3 test_dynamic_workflow.py \
    --num-workflows 100 \
    --qps 8.0 \
    --seed 42 \
    --strategies min_time round_robin probabilistic
```

**Via Code** (for advanced customization):

Edit `workload_generator.py` to modify fanout distribution:

```python
# In workload_generator.py:
FANOUT_MIN = 3         # Minimum B tasks per A task
FANOUT_MAX = 8         # Maximum B tasks per A task
```

### Timeout

If workflows take longer than expected, increase the timeout:

```python
# In test_strategy_workflow():
timeout_minutes=10  # Change to 15 or 20
```

---

## Interpreting Results

### Strategy Comparison

The final comparison table shows:

```
Strategy Comparison
================================================================================
Strategy        A Avg (s)    B Avg (s)    WF Avg (s)   WF P95 (s)   Completed
--------------------------------------------------------------------------------
min_time        5.23         5.12         12.34        18.56        100
round_robin     6.45         6.32         15.67        23.12        100
probabilistic   5.89         5.74         13.98        20.45        100
================================================================================
```

**Key Insights:**

1. **Workflow time** reflects **both** A task execution **and** the longest B task in each workflow's fanout
2. **min_time** typically shows best performance (shortest workflow times)
3. **round_robin** may show higher variance due to load imbalance
4. **Fanout impact**: Higher fanout increases likelihood of encountering a slow B task, extending workflow time

### Fanout Impact Analysis

For workflows with higher fanout:
- **More B tasks** = higher probability of at least one slow task
- **Workflow time** bottlenecked by the **slowest B task**
- Expected workflow time ≈ A_time + max(B_times for that workflow)

---

## Troubleshooting

### Issue: Not all workflows complete

**Symptoms:** `Completed: 87/100` in output

**Causes:**
- B task submission failures
- WebSocket connection issues
- Scheduler overload

**Solutions:**
1. Check scheduler logs: `ls logs/`
2. Increase timeout: `timeout_minutes=15`
3. Reduce load: `NUM_WORKFLOWS=50` or `QPS_A=5.0`
4. Check instance health: `curl http://localhost:8210/health`

### Issue: WebSocket connection errors

**Symptoms:** `WebSocket connection error: ...` in logs

**Solutions:**
1. Verify schedulers are running: `curl http://localhost:8100/health`
2. Increase connection wait time: `time.sleep(3.0)` after starting receivers
3. Check network/firewall settings

### Issue: B tasks not being submitted

**Symptoms:** B task count much lower than expected

**Causes:**
- A tasks not completing
- Thread 2 (A Result Receiver) not receiving events

**Solutions:**
1. Check A task completion: Look for "A Tasks: Completed: X" in output
2. Verify WebSocket subscription: Look for "Subscribed to X A tasks" in logs
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

---

## Files

```
experiments/05.multi_model_workflow_dynamic_parallel/
├── test_dynamic_workflow.py     # Main experiment code with sequential B task logic
│   ├── Thread 1: PoissonTaskSubmitter
│   ├── Thread 2: ATaskReceiver
│   ├── Thread 3: BTaskReceiver
│   ├── Thread 4: WorkflowMonitor
│   └── Main orchestration and metrics
│
├── workload_generator.py         # Workload generation (268 lines)
│   ├── generate_bimodal_distribution()
│   ├── generate_pareto_distribution()
│   └── generate_fanout_distribution()
│
├── start_all_services.sh         # Service startup script
├── stop_all_services.sh          # Service cleanup script
├── requirements.txt              # Python dependencies
│
├── README.md                     # This file
├── QUICK_REFERENCE.md            # Quick command reference
│
├── results/                      # Experiment results (JSON)
└── logs/                         # Service logs
```

---

## Example Workflow Timeline

```
Workflow wf-min_time-0042 (fanout=5) - SEQUENTIAL MODE:

t=0.000s  │ A task submitted (task-A-min_time-workflow-0042-A)
          │
t=2.500s  │ A task completed
          │ └─> B-00 submitted (sleep=2.1s) ← FIRST B task only
          │
t=4.600s  │ B-00 completed (2.1s)
          │ └─> B-01 submitted (sleep=1.8s) ← Submit next after previous completes
          │
t=6.400s  │ B-01 completed (1.8s)
          │ └─> B-02 submitted (sleep=8.7s) ← Sequential submission
          │
t=15.100s │ B-02 completed (8.7s)
          │ └─> B-03 submitted (sleep=2.3s)
          │
t=17.400s │ B-03 completed (2.3s)
          │ └─> B-04 submitted (sleep=1.9s) ← Last B task
          │
t=19.300s │ B-04 completed (1.9s)
          │
          │ ✓ Workflow complete!
          │ Workflow time = 19.300 - 0.000 = 19.3s
          │ (vs. parallel mode: 11.2s - sequential takes longer!)
```

**Key Difference from Parallel Mode:**
- **Parallel (Exp 04)**: Workflow time = A_time + max(B_times) = 2.5 + 8.7 = 11.2s
- **Sequential (Exp 05)**: Workflow time = A_time + Σ(B_times) = 2.5 + (2.1+1.8+8.7+2.3+1.9) = 19.3s

---

## Related Experiments

- **Experiment 03 (multi_model_workflow_1_to_1)**: 1-to-1 workflow dependencies (baseline)
- **Experiment 04 (multi_model_workflow_dynamic)**: Dynamic fanout with **parallel** B task execution
- **Experiment 05 (multi_model_workflow_dynamic_parallel)**: Dynamic fanout with **sequential** B task execution (this experiment)

---

## References

- [Scheduler WebSocket API](../../scheduler/docs/7.WEBSOCKET_API.md)
- [Task Management API](../../scheduler/docs/4.TASK_API.md)
- [Data Models](../../scheduler/docs/2.DATA_MODELS.md)
