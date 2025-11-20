# Experiment 11: Multi-Model Workflow with Repeat Execution

## Quick Start

```bash
# 1. Start all services
cd experiments/11.Exp3.Repeat
./start_all_services.sh 48 80

# 3. Run experiment (100 workflows, each repeats 1-3 times)
python3 test_dynamic_workflow.py --num-workflows 200 --qps 10 --gqps 100

# 4. Stop services when done
./stop_all_services.sh
```

**Expected output**: ~200 A tasks, ~2000 B tasks across ~100 workflows with automatic repeat execution.

---

## Overview

This experiment extends the dynamic workflow pattern by adding **repeat execution support**, where each workflow can execute **1-3 times** (randomly assigned). When a workflow completes all its B tasks, if more iterations are needed, the system **automatically resubmits the A task** to start the next iteration.

### Key Features

- **Repeat execution**: Each workflow executes 1-3 times (uniform random distribution)
- **Automatic iteration triggering**: Thread 3 detects iteration completion and submits next A task
- **Dynamic fanout**: Each A task generates 5-15 B tasks (uniform distribution)
- **Per-iteration tracking**: Each iteration's state is tracked independently
- **Same parameters across iterations**: Task times and fanout remain constant across all iterations
- **Comprehensive ID scheme**: Task IDs include workflow repeat count and iteration number

### Differences from Experiment 04 (No Repeat)

| Aspect | Experiment 04 | Experiment 11 |
|--------|--------------|--------------|
| **Workflow execution** | Execute once | Execute 1-3 times (repeat_num) |
| **Iteration triggering** | Manual (initial submit only) | Automatic (Thread 3 submits next A) |
| **Workflow ID** | `wf-{strategy}-{idx}` | `wf-{strategy}-{idx}-{repeat_num}` |
| **Task ID** | No iteration info | Includes `iter-{iter:02d}` |
| **State tracking** | Simple counter | Per-iteration state dict |
| **Workflow completion** | All B tasks done | All iterations done |

---

## Architecture

### Enhanced Four-Thread Design with Repeat Support

```
┌─────────────────────────────────────────────────────────────────────┐
│           Experiment 11: Workflow with Repeat Execution              │
└─────────────────────────────────────────────────────────────────────┘

Thread 1: A Task Submitter (Poisson) - ONLY FIRST ITERATION
   │
   ├──[QPS=8]──> Scheduler A ──> Instance Group A (10 instances)
   │                 │
   │                 │ (WebSocket completion events - ALL ITERATIONS)
   │                 ▼
   │            Thread 2: A Result Receiver
   │                 │
   │                 ├──[Extract n & iteration]──> Generate n B task submissions
   │                 │
   │                 └──[Submit all for this iteration]──> Scheduler B
   │                                                           │
   │                                                           ▼
   │                                                   Instance Group B (6 instances)
   │                                                           │
   │                                                           │ (WebSocket completion events)
   │                                                           ▼
   │                                                   Thread 3: B Result Receiver
   │                                                           │
   │                                                           ├──[Update iteration state]
   │                                                           │
   │                                                           ├──[Check if iteration complete]
   │                                                           │
   │                                                           ├──[If iteration < repeat_num]
   │                                                           │        │
   │                                                           │        └──> Submit next A task
   │                                                           │             (back to Scheduler A)
   │                                                           │
   │                                                           └──[If final iteration]──> Queue
   │                                                                                       │
   │                                                                                       ▼
   └──────────────────────────────────────────────────────────────> Thread 4: Workflow Monitor
                                                                                          │
                                                                                          └──> Calculate stats
```

### Thread Responsibilities (Updated for Repeat)

| Thread | Name | Key Changes | Purpose |
|--------|------|-------------|---------|
| 1 | A Task Submitter | Only submits **first iteration** A tasks | Initial Poisson submission |
| 2 | A Result Receiver | Handles **all iteration** A completions | Receives A results from any iteration, submits corresponding B tasks |
| 3 | B Result Receiver | **Detects iteration completion & submits next A** | Core repeat logic: checks completion → starts next iteration |
| 4 | Workflow Monitor | Tracks **overall workflow** completion | Receives events only when final iteration completes |

---

## Repeat Workflow State Tracking

### Enhanced Data Structures

```python
@dataclass
class IterationState:
    """Tracks a single iteration of a workflow."""
    iteration: int                      # e.g., 1, 2, 3
    total_b_tasks: int                  # e.g., 5
    completed_b_tasks: int = 0          # Counter: 0 → 5
    a_submit_time: Optional[float]
    a_complete_time: Optional[float]
    b_complete_times: Dict[str, float]  # Map: task_id → complete_time
    iteration_complete_time: Optional[float]  # max(b_complete_times)

    def is_complete(self) -> bool:
        return self.completed_b_tasks >= self.total_b_tasks


@dataclass
class WorkflowState:
    """Tracks entire workflow with multiple iterations."""
    workflow_id: str                    # e.g., "wf-min_time-0042-3"
    strategy: str
    repeat_num: int                     # Total iterations: 1, 2, or 3
    current_iteration: int = 1          # Current iteration being executed

    # Task IDs organized by iteration
    a_task_ids: Dict[int, str]                      # {iteration: a_task_id}
    b_task_ids: Dict[int, List[str]]                # {iteration: [b_task_ids]}
    total_b_tasks: int                              # B tasks per iteration

    # State tracking per iteration
    iteration_states: Dict[int, IterationState]     # {iteration: state}

    # Overall workflow timestamps
    workflow_start_time: Optional[float]            # First A submit
    workflow_complete_time: Optional[float]         # Last iteration complete

    def is_workflow_complete(self) -> bool:
        """Check if ALL iterations are complete."""
        return (self.current_iteration >= self.repeat_num and
                self.iteration_states[self.current_iteration].is_complete())

    def is_iteration_complete(self, iteration: int = None) -> bool:
        """Check if specific iteration is complete."""
        iter_num = iteration if iteration else self.current_iteration
        return self.iteration_states[iter_num].is_complete()

    def start_next_iteration(self):
        """Move to next iteration."""
        if self.current_iteration < self.repeat_num:
            self.current_iteration += 1
            # Initialize new iteration state
            self.iteration_states[self.current_iteration] = IterationState(...)
```

### Repeat Execution Logic (Thread 3)

```python
async def _handle_b_result(self, data: Dict):
    """Handle B task completion with repeat support."""
    # Parse task_id to extract workflow_id and iteration
    workflow_id = "wf-min_time-0042-3"
    iteration = 2  # Extracted from task_id

    # Mark B task complete for this iteration
    workflow.mark_b_task_complete(iteration, task_id, complete_time)

    # Check if current iteration is complete
    if workflow.is_iteration_complete(iteration):

        # More iterations needed?
        if workflow.current_iteration < workflow.repeat_num:
            logger.info(f"Iteration {iteration}/{workflow.repeat_num} complete. "
                       f"Starting next iteration...")

            # Move to next iteration
            workflow.start_next_iteration()
            next_iteration = workflow.current_iteration

            # Submit next A task (repeat execution!)
            await self._submit_next_a_task(workflow_id, next_iteration)

        # Final iteration complete?
        elif workflow.is_workflow_complete():
            logger.info(f"Workflow {workflow_id} fully completed "
                       f"({workflow.repeat_num} iterations)")
            # Push workflow completion event to Thread 4
            self._push_workflow_completion(workflow)
```

---

## Task ID Scheme with Repeat Support

### Enhanced ID Format

All task IDs now include:
1. **Workflow repeat count** (part of workflow_id)
2. **Iteration number** (which execution cycle)

```
Workflow ID: wf-{strategy}-{idx:04d}-{repeat_num}
A task ID:   task-A-{strategy}-workflow-{idx:04d}-{repeat_num}-A-iter-{iter:02d}
B task ID:   task-B-{strategy}-workflow-{idx:04d}-{repeat_num}-B-{j:02d}-iter-{iter:02d}
```

### Example: Workflow 42 with repeat_num=3, fanout=5

**Workflow ID**: `wf-min_time-0042-3`

**Iteration 1**:
```
A: task-A-min_time-workflow-0042-3-A-iter-01
B: task-B-min_time-workflow-0042-3-B-00-iter-01
   task-B-min_time-workflow-0042-3-B-01-iter-01
   task-B-min_time-workflow-0042-3-B-02-iter-01
   task-B-min_time-workflow-0042-3-B-03-iter-01
   task-B-min_time-workflow-0042-3-B-04-iter-01
```

**Iteration 2**:
```
A: task-A-min_time-workflow-0042-3-A-iter-02
B: task-B-min_time-workflow-0042-3-B-00-iter-02
   task-B-min_time-workflow-0042-3-B-01-iter-02
   task-B-min_time-workflow-0042-3-B-02-iter-02
   task-B-min_time-workflow-0042-3-B-03-iter-02
   task-B-min_time-workflow-0042-3-B-04-iter-02
```

**Iteration 3**: (same pattern with iter-03)

### Task Parsing

Extract components from task IDs:

```python
# A task ID: "task-A-min_time-workflow-0042-3-A-iter-02"
parts = task_id.split("-")
# parts: [task, A, strategy, workflow, idx, repeat_num, A, iter, iter_num]

strategy = parts[2]      # "min_time"
wf_idx = parts[4]        # "0042"
repeat_num = parts[5]    # "3"
iteration = int(parts[8]) # 2

workflow_id = f"wf-{strategy}-{wf_idx}-{repeat_num}"  # "wf-min_time-0042-3"
```

### Total Task Count

```
Base workflows: 100
Repeat distribution: Uniform(1, 3)
Expected iterations: 100 * (1+2+3)/3 = 200 total iterations

A tasks per strategy: ~200 (across all iterations)
B tasks per strategy: ~200 * avg_fanout ≈ 200 * 10 = 2000 B tasks

Total across 3 strategies: ~600 A + ~6000 B = ~6600 tasks
```

---

## Workload Generation

### A Tasks: Bimodal Distribution (Same per Iteration)

```python
Distribution: 50% fast, 50% slow
  - Fast: 0.5-0.7 seconds (mean=1.0s, std=0.4s)
  - Slow: 10.0-15.0 seconds (mean=20s, std=0.6s)

Note: Same A task time used for ALL iterations of a workflow
```

### B Tasks: 4-Peak Distribution (Same per Iteration)

```python
Distribution: 25% each peak
  - Peak 1: 0.5-1.5s (mean=1.0s, std=0.2s)
  - Peak 2: 4.0-6.0s (mean=4.0s, std=0.4s)
  - Peak 3: 10-30s (mean=20s, std=1s)
  - Peak 4: 60-120s (mean=100s, std=0.4s)

Note: Same B task times used for ALL iterations of a workflow
```

### Fanout: Uniform Distribution (Same per Iteration)

```python
Distribution: Uniform(5, 15)
  - Each A task generates 5-15 B tasks
  - Mean: 10 B tasks per A task
  - Std: ~3.16

Note: Same fanout used for ALL iterations of a workflow
```

### Repeat: Uniform Distribution

```python
Distribution: Uniform(1, 3)
  - Each workflow executes 1, 2, or 3 times
  - Mean: 2 iterations per workflow
  - Std: ~0.82

Examples:
  - Workflow 0001: repeat_num=1 (execute once, no repeat)
  - Workflow 0002: repeat_num=2 (execute twice, repeat once)
  - Workflow 0003: repeat_num=3 (execute three times, repeat twice)
```

---

## Metrics

### Iteration-Level Metrics (Optional)

Each iteration can be tracked independently:
- **Iteration time**: A submit → last B complete for that iteration
- **Per-iteration A/B task metrics**

### Workflow-Level Metrics (Primary)

**Overall Workflow Time** = First A submit → Last iteration's last B complete

```
Workflow time = workflow_complete_time - workflow_start_time

Example (workflow with repeat_num=2, fanout=3):
  Iteration 1:
    A1 submitted: t=0.0s
    A1 completed: t=1.5s
    B1 completions: [t=3.0s, t=3.5s, t=4.0s]
    Iteration 1 complete: t=4.0s

  Iteration 2:
    A2 submitted: t=4.1s (auto-triggered by Thread 3)
    A2 completed: t=5.6s
    B2 completions: [t=7.0s, t=7.5s, t=8.5s]
    Iteration 2 complete: t=8.5s

  Workflow time: 8.5 - 0.0 = 8.5s
```

**Metrics:**
- `num_completed`: Workflows with all iterations completed
- `avg_workflow_time`: Average overall workflow time
- `p95_workflow_time`: 95th percentile workflow time
- `repeat_distribution`: Histogram of repeat_num values
- `avg_repeat`: Average number of iterations per workflow
- `total_iterations`: Sum of all iterations executed

---

## Usage

### 1. Start Services

```bash
cd experiments/11.multi_model_workflow_repeat

# Start all services (predictor, schedulers, instances)
./start_all_services.sh

# Optional: Customize instance counts
N1=10 N2=6 ./start_all_services.sh
```

This starts:
- Predictor service (port 8101)
- Scheduler A (port 8100, WebSocket 8001) + 10 instances (ports 8210-8219)
- Scheduler B (port 8200, WebSocket 8002) + 6 instances (ports 8300-8305)

### 2. Run Unit Tests (Optional but Recommended)

```bash
# Activate virtual environment
cd /path/to/swarmpilot-refresh
source .venv/bin/activate

# Run unit tests
cd experiments/11.multi_model_workflow_repeat
python3 test_repeat_workflow.py

# Expected output:
# Ran 19 tests in 0.006s
# OK
```

### 3. Run Experiment

```bash
# Run full experiment (default: all 3 strategies, 100 workflows each, QPS=8.0)
python3 test_dynamic_workflow.py

# Run with custom parameters
python3 test_dynamic_workflow.py --num-workflows 50 --qps 10.0

# Run single strategy only
python3 test_dynamic_workflow.py --strategies min_time

# Small test run (10 workflows, one strategy)
python3 test_dynamic_workflow.py --num-workflows 10 --strategies min_time

# Show all available options
python3 test_dynamic_workflow.py --help
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-workflows` | int | 100 | Number of base workflows (each may repeat 1-3 times) |
| `--qps` | float | 8.0 | Target QPS for initial A task submission |
| `--gqps` | float | None | Global QPS limit for all A+B submissions (overrides --qps) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--strategies` | list | all three | Scheduling strategies to test |
| `--warmup` | float | 0.0 | Warmup task ratio (0.0-1.0) |

**Note**: With repeat execution:
- **A tasks**: `sum(repeat_values)` tasks (e.g., ~200 for 100 workflows with avg repeat=2)
- **B tasks**: `sum(repeat_values) * avg_fanout` tasks (e.g., ~2000 for 200 A tasks with avg fanout=10)

### 4. Monitor Progress

```
[INFO] Starting Experiment 11: Multi-Model Workflow with Repeat Execution
[INFO] Generating workloads...
[INFO] Generated 100 repeat values (mean=2.0)
[INFO] Generated 217 A task IDs across 217 iterations (100 actual workflows)
[INFO] Step 7: Starting Thread 3 (B Task Receiver with repeat support)
[INFO] Step 10: Starting Thread 1 (A Task Submitter - first iteration only)
[INFO] Workflow wf-min_time-0042-2 iteration 1/2 completed. Starting next iteration...
[INFO] Submitted next A task task-A-min_time-workflow-0042-2-A-iter-02
[INFO] Workflow wf-min_time-0042-2 fully completed (2 iterations)
[INFO] Workflows completed: 50/100
...
[INFO] All workflows completed!
```

### 5. View Results

Results include repeat statistics:

```
================================================================================
Results Summary: min_time
================================================================================

A Tasks (across all iterations):
  Submitted:  217
  Completed:  217
  Avg time:   5.23s

B Tasks (across all iterations):
  Submitted:  2147
  Completed:  2147
  Avg time:   25.12s

Workflows:
  Completed:  100
  Avg repeat: 2.17 iterations per workflow
  Total iterations: 217
  Avg workflow time: 45.34s  (includes all iterations)
  P95 workflow time: 78.56s

Repeat Distribution:
  1 iteration:  32 workflows (32.0%)
  2 iterations: 35 workflows (35.0%)
  3 iterations: 33 workflows (33.0%)
================================================================================
```

### 6. Stop Services

```bash
./stop_all_services.sh
```

---

## Example Workflow Timeline (with Repeat)

```
Workflow wf-min_time-0042-2 (repeat_num=2, fanout=3):

━━━ ITERATION 1 ━━━
t=0.000s  │ A1 submitted (task-A-min_time-workflow-0042-2-A-iter-01)
t=1.500s  │ A1 completed
          │ ├─> B-00-iter-01 submitted (sleep=2.1s)
          │ ├─> B-01-iter-01 submitted (sleep=1.8s)
          │ └─> B-02-iter-01 submitted (sleep=2.5s)
t=3.300s  │ B-01-iter-01 completed
t=3.600s  │ B-00-iter-01 completed
t=4.000s  │ B-02-iter-01 completed ← Last B of iteration 1
          │
          │ ✓ Iteration 1 complete!
          │ Thread 3 detects completion → current_iteration < repeat_num
          │ Thread 3 automatically submits next A task...

━━━ ITERATION 2 ━━━
t=4.100s  │ A2 submitted (task-A-min_time-workflow-0042-2-A-iter-02)
          │ [Same fanout, same task times as iteration 1]
t=5.600s  │ A2 completed
          │ ├─> B-00-iter-02 submitted (sleep=2.1s)
          │ ├─> B-01-iter-02 submitted (sleep=1.8s)
          │ └─> B-02-iter-02 submitted (sleep=2.5s)
t=7.400s  │ B-01-iter-02 completed
t=7.700s  │ B-00-iter-02 completed
t=8.100s  │ B-02-iter-02 completed ← Last B of iteration 2
          │
          │ ✓ Iteration 2 complete!
          │ current_iteration == repeat_num → Workflow complete!
          │ Overall workflow time = 8.100 - 0.000 = 8.1s
```

---

## Key Design Decisions

### 1. Why Thread 3 Submits Next A Tasks?

**Thread 3 (B Result Receiver)** is the natural place for repeat logic because:
- It tracks B task completions
- It knows when an iteration is complete
- It has access to workflow state and next A task data
- It can immediately trigger the next iteration without delay

### 2. Why Keep Task Parameters Constant?

Each workflow uses the **same task times and fanout** across iterations to:
- Ensure consistent behavior for analysis
- Avoid confounding variables (is variation due to repeat or different workload?)
- Simplify workload generation
- Make results more interpretable

### 3. Why Include repeat_num in Workflow ID?

Including `repeat_num` in the workflow ID (`wf-{strategy}-{idx}-{repeat_num}`) enables:
- Immediate identification of workflow's total iterations from any task ID
- Better logging and debugging
- Clearer results interpretation

---

## Troubleshooting

### Issue: Workflows stuck at first iteration

**Symptoms:** All workflows complete 1 iteration but don't repeat

**Causes:**
- Thread 3 not detecting iteration completion
- Next A task submission failing

**Solutions:**
1. Check Thread 3 logs for "Iteration X/Y completed. Starting next iteration..."
2. Verify workflow state: `current_iteration` should increment
3. Check Scheduler A for next iteration A task submissions
4. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

### Issue: Duplicate iteration execution

**Symptoms:** Some iterations execute more than once

**Causes:**
- Race condition in iteration completion detection
- Duplicate B task completion events

**Solutions:**
1. Check `IterationState.mark_b_task_complete()` idempotency
2. Verify `completed_workflows` set prevents duplicates
3. Review B task ID parsing logic

### Issue: Wrong iteration count

**Symptoms:** Workflow reports different iteration count than expected

**Causes:**
- Incorrect repeat_num assignment
- Task ID parsing errors

**Solutions:**
1. Verify repeat_values generation: `print(repeat_values)`
2. Test ID parsing with unit tests: `python3 test_repeat_workflow.py`
3. Check workflow_id format in logs

---

## Files

```
experiments/11.multi_model_workflow_repeat/
├── test_dynamic_workflow.py     # Main experiment code (~2300 lines)
│   ├── WorkflowTaskData (with repeat_num, iteration)
│   ├── IterationState (per-iteration tracking)
│   ├── WorkflowState (multi-iteration support)
│   ├── Thread 1: PoissonTaskSubmitter (first iteration only)
│   ├── Thread 2: ATaskReceiver (all iterations)
│   ├── Thread 3: BTaskReceiver (repeat logic)
│   ├── Thread 4: WorkflowMonitor
│   └── Main orchestration and metrics
│
├── test_repeat_workflow.py      # Unit tests (19 tests, all passing)
│   ├── Test IterationState
│   ├── Test WorkflowState
│   ├── Test Task ID generation
│   └── Test Repeat distribution
│
├── workload_generator.py         # Workload generation
│   ├── generate_bimodal_distribution()
│   ├── generate_b_task_bimodal_distribution()
│   ├── generate_fanout_distribution()
│   └── generate_repeat_distribution()  ← NEW
│
├── start_all_services.sh         # Service startup script
├── stop_all_services.sh          # Service cleanup script
├── requirements.txt              # Python dependencies
│
├── README.md                     # This file
├── results/                      # Experiment results (JSON)
└── logs/                         # Service logs
```

---

## Testing

### Unit Tests

Run the comprehensive unit test suite:

```bash
python3 test_repeat_workflow.py -v
```

**Test Coverage:**
- ✅ WorkflowTaskData with repeat fields (2 tests)
- ✅ IterationState lifecycle (4 tests)
- ✅ WorkflowState with multiple iterations (6 tests)
- ✅ Task ID generation with repeat (3 tests)
- ✅ Repeat distribution generation (3 tests)
- ✅ Task ID parsing (2 tests)

**Total: 19 tests, all passing ✓**

---

## Related Experiments

- **Experiment 04 (multi_model_workflow_dynamic)**: Dynamic fanout without repeat (baseline)
- **Experiment 07 (multi_model_workflow_dynamic_merge_2)**: Dynamic workflow merging
- **Experiment 10 (multi_model_workflow_plan_change)**: Dynamic plan changes at runtime

---

## References

- [Scheduler WebSocket API](../../scheduler/docs/7.WEBSOCKET_API.md)
- [Task Management API](../../scheduler/docs/4.TASK_API.md)
- [Data Models](../../scheduler/docs/2.DATA_MODELS.md)
