# Quick Reference: Experiment 03

## TL;DR

Experiment testing **1-to-1 workflow dependencies**: each A task triggers exactly one B task upon completion.

## Quick Start

```bash
# Start services
./start_all_services.sh

# Run experiment (default: 100 workflows, 3 strategies)
python test_dual_scheduler.py

# View results
ls -lh results/

# Stop services
./stop_all_services.sh
```

## Key Concept

```
Workflow:
  A task (Poisson QPS) → A completes → B task submitted → B completes
  └────────────────────── Workflow Time ─────────────────────┘
```

## Common Commands

### Basic Run
```bash
python test_dual_scheduler.py
```

### Custom Configuration
```bash
# 200 workflows, QPS=10, test only min_time
python test_dual_scheduler.py --num-workflows 200 --qps1 10.0 --strategies min_time

# Different instance counts
python test_dual_scheduler.py --n1 15 --n2 10

# High QPS test
python test_dual_scheduler.py --qps1 15.0 --num-workflows 150
```

### Test Specific Strategy
```bash
# Only round_robin
python test_dual_scheduler.py --strategies round_robin

# Multiple strategies
python test_dual_scheduler.py --strategies min_time probabilistic
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n1` | 10 | Number of Group A instances |
| `--n2` | 6 | Number of Group B instances |
| `--qps1` | 8.0 | QPS for A tasks (B follows A) |
| `--num-workflows` | 100 | Number of workflows per strategy |
| `--strategies` | all | Strategies to test |

## Strategies

- `round_robin`: Even distribution
- `min_time`: Assign to fastest instance
- `probabilistic`: Probabilistic load balancing

## Output Files

Results saved to `results/results_workflow_YYYYMMDD_HHMMSS.json`

## Service Ports

- **Predictor**: 8101
- **Scheduler A**: 8100 (WebSocket: ws://localhost:8100)
- **Scheduler B**: 8200 (WebSocket: ws://localhost:8200)
- **Group A Instances**: 8210-821N (N=n1-1)
- **Group B Instances**: 8300-830M (M=n2-1)

## Metrics Explained

### A Task Metrics
Time from A task submission to A task completion.

### B Task Metrics
Time from B task submission to B task completion.

### Workflow Metrics (KEY)
Time from A task submission to B task completion (end-to-end).

## Example Workflow

```
t=0.0s:   A task submitted (QPS control)
t=0.1s:   A task starts execution
t=2.5s:   A task completes
t=2.5s:   B task immediately submitted
t=2.6s:   B task starts execution
t=5.8s:   B task completes
─────────────────────────────
Workflow time: 5.8s
A completion: 2.5s
B completion: 3.3s (5.8 - 2.5)
```

## Quick Health Check

```bash
# Check Scheduler A
curl http://localhost:8100/health

# Check Scheduler B
curl http://localhost:8200/health

# Check Predictor
curl http://localhost:8101/health

# Check instances
docker ps | grep sleep_model
```

## Troubleshooting

### Services not starting
```bash
# Check Docker
docker ps

# Check logs
docker logs <container_id>

# Restart
./stop_all_services.sh
./start_all_services.sh
```

### WebSocket errors
```bash
# Test WebSocket manually
wscat -c ws://localhost:8100/task/get_result

# Send subscription
{"type": "subscribe", "task_ids": ["test-task-001"]}
```

### Results analysis
```bash
# Pretty print results
python -m json.tool results/results_workflow_20250115_103045.json

# Extract workflow metrics
jq '.results[].workflows' results/results_workflow_*.json
```

## Comparison with Experiment 02

| Feature | Exp 02 | Exp 03 |
|---------|--------|--------|
| A-B Relationship | Independent | 1-to-1 dependency |
| B Task QPS | Controlled | Follows A completion |
| Key Metric | Separate A/B times | Workflow time |
| Use Case | Independent workloads | Chained workflows |

## Analysis Examples

### Compare strategy workflow times
```bash
# Extract P95 workflow times
jq '.results[] | {strategy: .strategy, p95: .workflows.p95_completion_time}' results/results_workflow_*.json
```

### Count completed workflows
```bash
# Total workflows across all strategies
jq '[.results[].workflows.num_completed] | add' results/results_workflow_*.json
```

### Get best strategy
```bash
# Sort by average workflow time
jq '.results | sort_by(.workflows.avg_completion_time) | .[0].strategy' results/results_workflow_*.json
```

## Tips

1. **Start with default settings** to understand baseline performance
2. **Increase num-workflows** for more stable statistics
3. **Adjust QPS** to test under different load conditions
4. **Compare workflow P95** to understand tail latency impact
5. **Check A/B ratio** to understand where time is spent

## Common Patterns

### Load Testing
```bash
# Low load
python test_dual_scheduler.py --qps1 5.0 --num-workflows 50

# Medium load
python test_dual_scheduler.py --qps1 10.0 --num-workflows 100

# High load
python test_dual_scheduler.py --qps1 20.0 --num-workflows 200
```

### Strategy Comparison
```bash
# All strategies, large sample
python test_dual_scheduler.py --num-workflows 200 --strategies min_time round_robin probabilistic
```

### Resource Scaling
```bash
# Scale Group A
python test_dual_scheduler.py --n1 20

# Scale Group B
python test_dual_scheduler.py --n2 12

# Scale both
python test_dual_scheduler.py --n1 15 --n2 10
```

## Next Steps

1. Run experiment with default settings
2. Analyze workflow completion times
3. Compare strategies based on workflow P95
4. Adjust parameters to optimize for your use case
5. Consider scaling resources if needed

## Documentation

- Full details: See [README.md](README.md)
- Workload generation: See [workload_generator.py](workload_generator.py)
- Implementation: See [test_dual_scheduler.py](test_dual_scheduler.py)
