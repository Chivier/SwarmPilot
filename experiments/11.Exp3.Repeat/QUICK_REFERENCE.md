# Quick Reference: Experiment 06

## One-Line Summary

**1-to-n dynamic fanout workflow**: Each A task generates 3-8 B tasks in parallel; workflow completes when all B tasks finish.

---

## Quick Start

```bash
# 1. Start services
./start_all_services.sh

# 2. Run experiment (default: 100 workflows, QPS=8.0)
uv run python3 test_dynamic_workflow.py

# 3. Stop services
./stop_all_services.sh
```

## Command-Line Parameters

```bash
# Full experiment (default parameters)
uv run python3 test_dynamic_workflow.py

# Custom workflows and QPS
uv run python3 test_dynamic_workflow.py --num-workflows 50 --qps 10.0

# Test single strategy only
uv run python3 test_dynamic_workflow.py --strategies min_time

# Test specific strategies
uv run python3 test_dynamic_workflow.py --strategies min_time round_robin

# Small test run
uv run python3 test_dynamic_workflow.py --num-workflows 10 --qps 5.0 --strategies min_time

# Show all options
uv run python3 test_dynamic_workflow.py --help
```

---

## Common Commands

### Service Management

```bash
# Start with default instances (10 A, 6 B)
./start_all_services.sh

# Start with custom instances
N1=15 N2=10 ./start_all_services.sh

# Stop all services
./stop_all_services.sh

# Check scheduler health
curl http://localhost:8100/health  # Scheduler A
curl http://localhost:8200/health  # Scheduler B
```

### Run Experiment

```bash
# Full experiment (all strategies, 100 workflows)
uv run python3 test_dynamic_workflow.py

# Test workload generator
uv run python3 workload_generator.py --num-workflows 100

# With logging
uv run python3 test_dynamic_workflow.py 2>&1 | tee experiment.log
```

### View Results

```bash
# List results
ls -lh results/

# View latest result
cat results/results_workflow_dynamic_*.json | jq '.results[] | {strategy, workflows}'

# Pretty print
cat results/results_workflow_dynamic_20251102_143022.json | jq '.'
```

---

## Key Parameters

### Command-Line Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-workflows` | 100 | Number of workflows to generate and execute per strategy |
| `--qps` | 8.0 | Target QPS for A task submission |
| `--seed` | 42 | Random seed for reproducibility |
| `--strategies` | all three | Strategies to test: min_time, round_robin, probabilistic |

### Configuration Constants

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `FANOUT_MIN` | 3 | `workload_generator.py:39` | Min B tasks per A task |
| `FANOUT_MAX` | 8 | `workload_generator.py:40` | Max B tasks per A task |
| `timeout_minutes` | 10 | `test_dynamic_workflow.py:1082` | Max wait time |
| `N1` | 10 | `start_all_services.sh:20` | Scheduler A instances |
| `N2` | 6 | `start_all_services.sh:21` | Scheduler B instances |

---

## Architecture Quick View

```
Thread 1 (Submitter)   →  Scheduler A  →  10 instances
                              ↓
Thread 2 (A Receiver)  ←──WebSocket
        ↓
    Submit n B tasks
        ↓
Thread 2 (B Submitter) →  Scheduler B  →  6 instances
                              ↓
Thread 3 (B Receiver)  ←──WebSocket
        ↓
    Update workflow state
        ↓
Thread 4 (Monitor)     ←──Completion queue
```

---

## Task ID Format

```
A task:  task-A-{strategy}-workflow-{i:04d}-A
B tasks: task-B-{strategy}-workflow-{i:04d}-B-{j:02d}

Example (workflow 42, fanout=5, strategy=min_time):
  task-A-min_time-workflow-0042-A
  task-B-min_time-workflow-0042-B-00
  task-B-min_time-workflow-0042-B-01
  task-B-min_time-workflow-0042-B-02
  task-B-min_time-workflow-0042-B-03
  task-B-min_time-workflow-0042-B-04
```

---

## Metrics Quick Reference

### A Tasks
- `avg_completion_time`: A submit → A complete (avg)
- `p95_completion_time`: A submit → A complete (P95)

### B Tasks
- `avg_completion_time`: B submit → B complete (avg)
- `p95_completion_time`: B submit → B complete (P95)

### Workflows (KEY)
- `avg_workflow_time`: A submit → last B complete (avg)
- `p95_workflow_time`: A submit → last B complete (P95)
- `avg_fanout`: Average B tasks per A task

---

## Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| Not all workflows complete | Increase timeout: `timeout_minutes=15` |
| WebSocket errors | Wait longer: `time.sleep(3.0)` after thread start |
| B tasks not submitted | Check A task completion count |
| Scheduler not responding | Restart services: `./stop_all_services.sh && ./start_all_services.sh` |
| Instance not healthy | Check logs: `tail -f logs/instance_*` |

---

## Expected Behavior

### Normal Run

```
[INFO] Starting Experiment 06
[INFO] Testing strategy: min_time
[INFO] Subscribed to 100 A tasks
[INFO] Subscribed to 547 B tasks
[INFO] Submitted 100/100 A tasks
[INFO] Workflows completed: 100/100
[INFO] All workflows completed!
```

### Performance Expectations

| Metric | Expected Range |
|--------|---------------|
| A task avg time | 4-7s |
| B task avg time | 4-7s |
| Workflow avg time | 10-15s |
| Workflow P95 | 18-25s |
| Total experiment time | 5-8 minutes |

---

## File Locations

```
.
├── test_dynamic_workflow.py    # Main code
├── workload_generator.py       # Workload generation
├── start_all_services.sh       # Start services
├── stop_all_services.sh        # Stop services
├── requirements.txt            # Dependencies
├── README.md                   # Full documentation
├── QUICK_REFERENCE.md          # This file
├── results/                    # JSON results
│   └── results_workflow_dynamic_<timestamp>.json
└── logs/                       # Service logs
    ├── predictor.log
    ├── scheduler_a.log
    ├── scheduler_b.log
    └── instance_*.log
```

---

## Useful Log Commands

```bash
# Follow all logs
tail -f logs/*.log

# Scheduler A activity
tail -f logs/scheduler_a.log | grep -E "Task|submitted|completed"

# Count completed tasks
grep "completed" logs/scheduler_a.log | wc -l
grep "completed" logs/scheduler_b.log | wc -l

# Check for errors
grep -i error logs/*.log

# Instance utilization
grep "assigned" logs/instance_*.log | awk '{print $1}' | sort | uniq -c
```

---

## Testing Tips

### Small Scale Test

```bash
# Quick test: single strategy, 10 workflows
uv run python3 test_dynamic_workflow.py --num-workflows 10 --strategies min_time

# Medium test: two strategies, 50 workflows
uv run python3 test_dynamic_workflow.py --num-workflows 50 --strategies min_time round_robin

# High QPS test: 20 QPS, 30 workflows
uv run python3 test_dynamic_workflow.py --num-workflows 30 --qps 20.0 --strategies min_time
```

### Debug Mode

```python
# Enable debug logging
logging.basicConfig(level=logging.DEBUG, ...)
```

### Dry Run

```bash
# Test workload generation only
uv run python3 workload_generator.py --num-workflows 10
```

---

## Comparison with Other Experiments

| Experiment | Pattern | B Tasks per A | Completion Criteria |
|------------|---------|--------------|---------------------|
| 03 | 1-to-1 | 1 | B completes |
| **06** | **1-to-n (dynamic)** | **3-8 (uniform)** | **All B complete** |
| 04 | 1-to-n (parallel) | Fixed n | All B complete |
| 05 | 1-to-n (serial) | Fixed n | Last B completes |

---

## Success Criteria

✅ **Experiment successful if:**
- All workflows complete (100/100 per strategy)
- No WebSocket errors
- Workflow times in expected range (10-15s avg)
- Results saved to JSON

❌ **Investigate if:**
- Completed workflows < 95%
- Workflow avg time > 20s
- Many "submit_failed" errors
- WebSocket connection drops

---

## Next Steps After Running

1. **Analyze results:**
   ```bash
   cat results/results_workflow_dynamic_*.json | jq '.results[] |
     {strategy, wf_avg: .workflows.avg_workflow_time}'
   ```

2. **Compare strategies:**
   - Check which strategy has lowest avg workflow time
   - Analyze fanout impact on workflow time

3. **Export for visualization:**
   ```bash
   cat results/results_workflow_dynamic_*.json | jq '.results' > analysis.json
   ```

4. **Run with different parameters:**
   - Test higher fanout: `FANOUT_MIN=5, FANOUT_MAX=12`
   - Test higher QPS: `QPS_A=15.0`
   - Test more instances: `N1=20 N2=15`

---

## Support

- Full documentation: `README.md`
- WebSocket API: `../../scheduler/docs/7.WEBSOCKET_API.md`
- Task API: `../../scheduler/docs/4.TASK_API.md`
