# Testing Guide for 01.quick-start-up

## Overview

This guide explains how the updated testing framework works with task clearing and event-driven WebSocket architecture.

## Key Improvements

### 1. Task Clearing Before Each Test

**Why?** Ensures clean state between strategy tests, preventing interference from previous runs.

**Implementation:**
```python
# Before testing each strategy
def test_strategy_with_poisson(strategy_name, tasks, qps):
    # Step 1: Clear all tasks from scheduler
    clear_tasks()  # POST /task/clear

    # Step 2: Set the strategy
    set_strategy(strategy_name)  # POST /strategy/set

    # Step 3: Start WebSocket receiver (Thread 2)
    receiver = WebSocketResultReceiver(...)
    receiver.start()

    # Step 4: Start Poisson submitter (Thread 1)
    submitter = PoissonTaskSubmitter(...)
    submitter.start()
```

**API Endpoints Used:**
- `POST /task/clear`: Clears all tasks from scheduler's task registry
- `POST /strategy/set`: Sets strategy and reinitializes instance queues

### 2. Two-Threaded Event-Driven Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Main Thread                       │
│                                                     │
│  1. Generate pre-generated task data (shared)      │
│  2. For each strategy:                             │
│     ├─→ Clear tasks                                │
│     ├─→ Set strategy                               │
│     ├─→ Launch Thread 2 (WebSocket Receiver)      │
│     ├─→ Launch Thread 1 (Poisson Submitter)       │
│     └─→ Collect and analyze results                │
└─────────────────────────────────────────────────────┘
         │                            │
         ▼                            ▼
┌──────────────────────┐   ┌────────────────────────┐
│   Thread 2           │   │   Thread 1             │
│   (Receiver)         │   │   (Submitter)          │
│                      │   │                        │
│ [STARTS FIRST]       │   │ [STARTS SECOND]        │
│                      │   │                        │
│ 1. Connect WS        │   │ 1. Wait for WS ready   │
│ 2. Subscribe tasks   │   │ 2. Generate intervals  │
│ 3. Listen results    │   │ 3. Submit tasks        │
│ 4. Put to queue      │   │ 4. Track metrics       │
└──────────────────────┘   └────────────────────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
              ┌──────────────┐
              │ Result Queue │
              └──────────────┘
                      │
                      ▼
              Main Thread collects
```

### 3. Poisson Process Task Submission

**Formula:** Inter-arrival time ~ Exponential(λ = QPS)

```python
# Generate inter-arrival times
lambda_rate = qps  # e.g., 10 QPS
inter_arrival_times = np.random.exponential(1.0 / lambda_rate, num_tasks)

# Submit tasks with delays
for task, wait_time in zip(tasks, inter_arrival_times):
    time.sleep(wait_time)  # Wait for next arrival
    submit_task(task)
```

**Example:** QPS=10
- Average interval: 0.1 seconds
- Actual intervals: [0.05s, 0.15s, 0.08s, 0.12s, ...] (random but λ=10)

## Testing Workflow

### Quick Test (3 tasks, 1 QPS, single strategy)

```bash
uv run python test_scheduling_poisson.py \
  --qps 1 \
  --num-tasks 3 \
  --strategies round_robin
```

**Expected Output:**
```
============================================================
01.quick-start-up Experiment (Poisson Process)
============================================================
Testing 1 scheduling strategies
Tasks per strategy: 3
Target QPS: 1.0
...
✓ Scheduler is healthy
✓ Generated task times: min=1.523s, max=4.891s, mean=3.207s

============================================================
Testing strategy: ROUND_ROBIN
============================================================

Step 1: Clearing existing tasks...
✓ Cleared 0 tasks from scheduler

Step 2: Setting scheduling strategy...
✓ Strategy set to: round_robin
  Cleared 0 tasks during strategy switch
  Reinitialized 16 instances

✓ WebSocket connected to ws://localhost:8100/task/get_result
✓ Subscribed to 3 tasks

============================================================
Starting task submission (QPS=1.0)
============================================================
  Submitted 3/3 tasks
✓ Submitted 3/3 tasks
  Total submission time: 2.156s
  Actual QPS: 1.39

============================================================
Waiting for task results...
============================================================
  Completed 3/3 tasks
✓ Completed 3/3 tasks
✓ WebSocket connection closed

============================================================
Results for ROUND_ROBIN
============================================================
Total tasks:              3
Submitted tasks:          3
Completed tasks:          3
Target QPS:               1.00
Actual QPS:               1.39
Submission time:          2.156s
Avg completion time:      3.234s
Median completion time:   3.189s
P95 completion time:      4.912s
P99 completion time:      4.912s
Total execution time:     7.891s

Task distribution:
  instance-000: 1 tasks
  instance-001: 1 tasks
  instance-002: 1 tasks
```

### Full Test (100 tasks, 10 QPS, all strategies)

```bash
uv run python test_scheduling_poisson.py
```

**Runtime:** ~20-30 minutes for all three strategies

### Custom Test

```bash
# High throughput test
uv run python test_scheduling_poisson.py --qps 20 --num-tasks 200

# Compare min_time vs probabilistic
uv run python test_scheduling_poisson.py --strategies min_time probabilistic

# Low QPS for detailed observation
uv run python test_scheduling_poisson.py --qps 2 --num-tasks 50
```

## API Interactions

### 1. Clear Tasks (Before Each Strategy)

**Request:**
```bash
POST http://localhost:8100/task/clear
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully cleared 100 task(s)",
  "cleared_count": 100
}
```

### 2. Set Strategy

**Request:**
```bash
POST http://localhost:8100/strategy/set
Content-Type: application/json

{
  "strategy_name": "probabilistic",
  "target_quantile": 0.9
}
```

**Response:**
```json
{
  "success": true,
  "message": "Strategy switched to probabilistic",
  "cleared_tasks": 0,
  "reinitialized_instances": 16,
  "strategy_info": {
    "name": "probabilistic",
    "description": "Probabilistic scheduling with quantile-based predictions",
    "parameters": {
      "target_quantile": 0.9
    }
  }
}
```

### 3. WebSocket Subscribe

**Client → Server:**
```json
{
  "type": "subscribe",
  "task_ids": ["task-round_robin-0000", "task-round_robin-0001", ...]
}
```

**Server → Client (Acknowledgment):**
```json
{
  "type": "ack",
  "message": "Subscribed to 100 task(s)"
}
```

### 4. WebSocket Result

**Server → Client (On Task Completion):**
```json
{
  "type": "result",
  "task_id": "task-round_robin-0000",
  "status": "completed",
  "result": {
    "output": "Slept for 3.25 seconds"
  },
  "error": null,
  "execution_time_ms": 3250.12,
  "timestamp": "2025-11-02T10:30:00.000Z"
}
```

## Troubleshooting

### Tasks not cleared

**Symptom:** Old tasks appear in results

**Solution:**
```bash
# Manually clear tasks
curl -X POST http://localhost:8100/task/clear

# Check if clear endpoint is working
uv run python -c "
import requests
r = requests.post('http://localhost:8100/task/clear')
print(r.json())
"
```

### WebSocket connection failed

**Symptom:** `✗ WebSocket connection error`

**Solution:**
1. Check scheduler is running: `curl http://localhost:8100/health`
2. Check WebSocket endpoint: `wscat -c ws://localhost:8100/task/get_result`
3. Check firewall/network settings

### Strategy switch failed

**Symptom:** `✗ Failed to set strategy`

**Common Causes:**
- Tasks are still running (scheduler requires all idle)
- Invalid strategy name
- Predictor service not available (for min_time/probabilistic)

**Solution:**
```bash
# Check scheduler status
curl http://localhost:8100/health

# Check if tasks are still running
curl http://localhost:8100/task/list?status=running

# Wait for tasks to complete, then retry
```

## Comparison: Old vs New

### Old Implementation (test_scheduling.py)

```python
# Submit all tasks at once
for task in tasks:
    submit_task(task)

# Poll for completion
while not all_completed:
    for task in tasks:
        status = get_task_info(task)  # HTTP GET (polling)
        if status == "completed":
            # ...
    time.sleep(0.5)  # Polling interval
```

**Issues:**
- High polling overhead
- No task clearing between strategies
- All tasks submitted simultaneously (unrealistic)

### New Implementation (test_scheduling_poisson.py)

```python
# Clear tasks first
clear_tasks()

# Set strategy
set_strategy(strategy_name)

# WebSocket receiver (event-driven)
receiver.start()  # Subscribe to all tasks

# Poisson submitter
for task, interval in zip(tasks, exponential_intervals):
    time.sleep(interval)  # Poisson process
    submit_task(task)

# Results arrive via WebSocket (no polling)
```

**Improvements:**
- ✅ Clean state between tests
- ✅ Event-driven (no polling)
- ✅ Realistic workload (Poisson process)
- ✅ Lower overhead
- ✅ Shared task data across strategies

## References

- Scheduler API Documentation: `scheduler/README_FOR_LLM.md`
- WebSocket Protocol: Lines 610-698
- Task Clear Endpoint: Lines 542-563
- Strategy Set Endpoint: Lines 731-774
