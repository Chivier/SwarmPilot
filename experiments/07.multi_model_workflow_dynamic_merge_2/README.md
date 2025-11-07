# Experiment 07: Multi-Model Workflow with B1/B2 Split and Merge (1-to-n-to-n-to-1)

## Overview

This experiment extends experiment 06 by **splitting the B stage into B1 and B2 substages**. Each A task generates n B1 tasks (slow, 7-10s), and each B1 task completion triggers one B2 task (fast, 1-3s). After all B2 tasks complete, a merge A task is triggered. This tests pipelined workflow execution with cascading dependencies.

### Workflow Pattern

```
A task → n B1 tasks (parallel, slow) → n B2 tasks (pipeline, fast) → Merge A task → Workflow Complete
```

### Key Features

- **Dynamic fanout**: Each A task generates 3-8 B1 tasks (uniform distribution)
- **B1/B2 pairing**: Each B1 task triggers exactly one B2 task (1:1 pairing via b_index)
- **Pipelined execution**: B2 tasks start as soon as their corresponding B1 completes (not batched)
- **Bimodal split**: B1 uses slow peak (7-10s), B2 uses fast peak (1-3s)
- **Seven-thread architecture**: Specialized threads for each stage transition
- **Merge task**: After all B2 tasks complete, a merge A task is submitted (0.5x execution time of original A)
- **Pre-calculated task IDs**: All task IDs (A, B1, B2, and merge) generated upfront for WebSocket subscription
- **WebSocket-based result collection**: Real-time, push-based task completion events
- **Three scheduling strategies**: min_time, round_robin, probabilistic

### Differences from Experiment 06 (1-to-n-to-1)

| Aspect | Experiment 06 | Experiment 07 |
|--------|--------------|--------------|
| **Workflow pattern** | A → n B → Merge | A → n B1 → n B2 → Merge |
| **B stage** | Single B stage (bimodal) | Split into B1 (slow) and B2 (fast) |
| **B1 distribution** | N/A | Slow peak only (7-10s) |
| **B2 distribution** | N/A | Fast peak only (1-3s) |
| **B1→B2 trigger** | N/A | Each B1 completion triggers paired B2 (pipeline) |
| **Threading model** | 6 threads | 7 threads (+ B1 receiver/B2 submitter) |
| **Scheduler usage** | B tasks → Scheduler B | B1 and B2 → same Scheduler B |

---

## Architecture

### Seven-Thread Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Experiment 07 Architecture                            │
│              (1-to-n-to-n-to-1 with B1/B2 Split and Merge)               │
└─────────────────────────────────────────────────────────────────────────┘

Thread 1: A Task Submitter (Poisson)
   │
   ├──[QPS=8]──> Scheduler A ──> Instance Group A (10 instances)
   │                 │
   │                 │ (WebSocket completion events for A tasks)
   │                 ▼
   │            Thread 2: A Result Receiver + B1 Submitter
   │                 │
   │                 ├──[Extract n]──> Generate n B1 task submissions
   │                 │
   │                 └──[Submit all B1]──> Scheduler B ──> Instance Group B (6 instances)
   │                                            │
   │                                            │ (WebSocket completion events for B1 tasks)
   │                                            ▼
   │                                       Thread 3: B1 Result Receiver + B2 Submitter
   │                                            │
   │                                            ├──[B1 complete]──> Submit paired B2 task
   │                                            │
   │                                            └──[Submit B2]──> Scheduler B (same)
   │                                                                    │
   │                                                                    │ (WebSocket for B2 tasks)
   │                                                                    ▼
   │                                                              Thread 4: B2 Result Receiver
   │                                                                    │
   │                                                                    ├──[Update workflow state]
   │                                                                    │
   │                                                                    └──[All B2 complete]──> merge_ready_queue
   │                                                                                                │
   │                                                                                                ▼
   │                                                                          Thread 5: Merge Task Submitter
   │                                                                                │
   │                                                                                └──[Submit merge A]──> Scheduler A
   │                                                                                                          │
   │                                                                                                          │ (WebSocket for merge)
   │                                                                                                          ▼
   │                                                                                        Thread 6: Merge Task Receiver
   │                                                                                                │
   │                                                                                                └──[Merge complete]──> completion_queue
   │                                                                                                                           │
   │                                                                                                                           ▼
   └───────────────────────────────────────────────────────────────────────────────────────────────────────> Thread 7: Workflow Monitor
                                                                                                                              │
                                                                                                                              └──> Calculate stats
```

### Thread Responsibilities

| Thread | Name | Input | Output | Purpose |
|--------|------|-------|--------|---------|
| 1 | A Task Submitter | A task data, QPS | Submitted A tasks | Submit initial A tasks with Poisson inter-arrival times |
| 2 | A Result Receiver + B1 Submitter | A completion events | n B1 task submissions | Receive A results, submit n B1 tasks (slow peak) per workflow |
| 3 | B1 Result Receiver + B2 Submitter | B1 completion events | B2 task submissions | Receive B1 results, submit paired B2 task (fast peak) immediately |
| 4 | B2 Result Receiver | B2 completion events | merge_ready_queue | Track B2 task completion, push workflow_id when all B2 tasks done |
| 5 | Merge Task Submitter | merge_ready_queue | Submitted merge A tasks | Submit merge A task (0.5x runtime) when all B2 tasks complete |
| 6 | Merge Task Receiver | Merge A completion events | completion_queue | Receive merge completions, push final workflow completion events |
| 7 | Workflow Monitor | completion_queue | Aggregated statistics | Calculate workflow metrics, detect experiment end |

---

## Workflow State Tracking

Each workflow maintains state to track completion:

```python
@dataclass
class WorkflowState:
    workflow_id: str                    # e.g., "wf-min_time-0042"
    a_task_id: str                      # e.g., "task-A-min_time-workflow-0042-A"
    b_task_ids: List[str]               # e.g., ["task-B-...-B-00", ..., "task-B-...-B-04"]
    merge_task_id: str                  # NEW: e.g., "task-A-min_time-workflow-0042-merge"
    total_b_tasks: int                  # e.g., 5
    completed_b_tasks: int              # Counter: 0 → 5

    # Timestamps
    a_submit_time: float
    a_complete_time: float
    b_complete_times: Dict[str, float]  # Map: task_id → complete_time
    all_b_complete_time: float          # NEW: When all B tasks finished
    merge_submit_time: float            # NEW: When merge task submitted
    merge_complete_time: float          # NEW: When merge task completed
    workflow_complete_time: float       # NEW: Same as merge_complete_time
```

### Workflow Completion Logic

```python
def mark_b_task_complete(b_task_id: str, complete_time: float):
    """Thread 3 calls this for each B task completion."""
    self.b_complete_times[b_task_id] = complete_time
    self.completed_b_tasks += 1

    # When all B tasks done, set all_b_complete_time
    if self.are_all_b_tasks_complete():
        self.all_b_complete_time = max(self.b_complete_times.values())

def mark_merge_task_complete(complete_time: float):
    """Thread 6 calls this when merge task completes."""
    self.merge_complete_time = complete_time
    self.workflow_complete_time = complete_time  # Workflow now complete!
```

---

## Task ID Scheme

### Pre-calculation Strategy

To enable WebSocket subscription before task submission (avoiding race conditions), all task IDs are pre-calculated:

```
A tasks (100):      task-A-{strategy}-workflow-{i:04d}-A
B tasks (variable): task-B-{strategy}-workflow-{i:04d}-B-{j:02d}
Merge tasks (100):  task-A-{strategy}-workflow-{i:04d}-merge

Example for workflow 42 with 5 B tasks (strategy=min_time):
  A:     task-A-min_time-workflow-0042-A
  B[0]:  task-B-min_time-workflow-0042-B-00
  B[1]:  task-B-min_time-workflow-0042-B-01
  B[2]:  task-B-min_time-workflow-0042-B-02
  B[3]:  task-B-min_time-workflow-0042-B-03
  B[4]:  task-B-min_time-workflow-0042-B-04
  Merge: task-A-min_time-workflow-0042-merge
```

### Total Task Count

```
A tasks per strategy:     100
B tasks per strategy:     Σ(fanout[i]) for i in [0, 99]
  - Fanout ~ Uniform(3, 8)
  - Expected total: 100 * (3+8)/2 = 550 B tasks
  - Actual varies per run
Merge tasks per strategy: 100

Total across 3 strategies: 300 A + ~1650 B + 300 merge = ~2250 tasks
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
cd experiments/06.multi_model_workflow_dynamic

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
| `--qps` | float | 8.0 | Target queries per second (QPS) for A task submission (Poisson arrival pattern) |
| `--gqps` | float | None | Global QPS limit for all task submissions (A, B1, B2, merge). Can be used together with --qps |
| `--warmup` | float | 0.0 | Warmup task ratio (0.0-1.0). E.g., 0.2 means 20% warmup tasks before actual workload |
| `--continuous` | bool | False | Enable continuous request mode (submit 2x workflows, track first num_workflows) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--strategies` | list | all three | Scheduling strategies to test (min_time, round_robin, probabilistic) |

**Note**: The experiment will generate and submit **exactly** the number of tasks specified:
- A tasks: `num_workflows` tasks (e.g., 100)
- B tasks: `sum(fanout_values)` tasks (e.g., ~550 for 100 workflows with fanout 3-8)

#### QPS Control: `--qps` vs `--gqps`

The experiment supports two independent QPS control mechanisms that can work together:

**1. `--qps` (A Task Arrival Rate)**
- Controls the **arrival pattern** of A tasks using a Poisson process
- Simulates realistic user request patterns with exponentially distributed inter-arrival times
- Only affects A task submission timing
- Default: 8.0 QPS

**2. `--gqps` (Global QPS Limit)**
- Applies a **global rate limit** to all task submissions (A, B1, B2, merge) using token bucket algorithm
- Prevents overloading the system by capping total submission rate
- Applies to all threads submitting tasks
- Default: None (no global limit)

**Using Both Parameters Together:**

When both `--qps` and `--gqps` are specified, they work in tandem:

```bash
# Example: A tasks arrive at 8 QPS (Poisson), but global submission rate is capped at 20 QPS
uv run python3 test_dynamic_workflow.py --num-workflows 200 --qps 8 --gqps 20
```

**Behavior:**
1. A tasks are submitted following Poisson process at 8 QPS average
2. Each task submission (A, B1, B2, merge) must acquire a token from the global rate limiter (20 QPS total)
3. If the total submission rate exceeds 20 QPS, tasks will be throttled

**Use Cases:**
- `--qps` only: Simulate specific arrival patterns without global constraint
- `--gqps` only: Hard cap on system load, A tasks submitted as fast as possible (within global limit)
- Both: Realistic arrival pattern with system protection (recommended for production-like scenarios)

**Example Scenarios:**

```bash
# Scenario 1: Low arrival rate, no global limit (baseline)
uv run python3 test_dynamic_workflow.py --num-workflows 100 --qps 5

# Scenario 2: High arrival rate with global protection
uv run python3 test_dynamic_workflow.py --num-workflows 200 --qps 10 --gqps 25

# Scenario 3: Stress test (high arrival, high global limit)
uv run python3 test_dynamic_workflow.py --num-workflows 500 --qps 20 --gqps 50
```

### 3. Monitor Progress

The experiment provides detailed logging:

```
[INFO] Starting Experiment 06: Multi-Model Workflow with Dynamic Fanout
[INFO] Generating workloads...
[INFO] Testing strategy: min_time
[INFO] Step 1: Clearing tasks from schedulers
[INFO] Step 7: Starting Thread 3 (B Task Receiver)
[INFO] Subscribed to 547 B tasks
[INFO] Step 10: Starting Thread 1 (A Task Submitter)
[INFO] Submitted 20/100 A tasks
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
experiments/06.multi_model_workflow_dynamic/
├── test_dynamic_workflow.py     # Main experiment code (1369 lines)
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
Workflow wf-min_time-0042 (fanout=5):

t=0.000s  │ A task submitted (task-A-min_time-workflow-0042-A)
          │
t=2.500s  │ A task completed
          │ ├─> B-00 submitted (sleep=2.1s)
          │ ├─> B-01 submitted (sleep=1.8s)
          │ ├─> B-02 submitted (sleep=8.7s)  ← slowest
          │ ├─> B-03 submitted (sleep=2.3s)
          │ └─> B-04 submitted (sleep=1.9s)
          │
t=4.300s  │ B-01 completed (1.8s)
t=4.600s  │ B-04 completed (1.9s)
t=4.800s  │ B-00 completed (2.1s)
t=4.800s  │ B-03 completed (2.3s)
t=11.200s │ B-02 completed (8.7s) ← last B task
          │
          │ ✓ Workflow complete!
          │ Workflow time = 11.200 - 0.000 = 11.2s
```

---

## Related Experiments

- **Experiment 03 (multi_model_1_to_1)**: 1-to-1 workflow dependencies (baseline)
- **Experiment 04 (multi_model_1_to_n_parallel)**: Static fanout, all B tasks independent
- **Experiment 05 (multi_model_1_to_n_serial)**: Chain dependencies (A → B1 → B2 → ...)

---

## References

- [Scheduler WebSocket API](../../scheduler/docs/7.WEBSOCKET_API.md)
- [Task Management API](../../scheduler/docs/4.TASK_API.md)
- [Data Models](../../scheduler/docs/2.DATA_MODELS.md)
