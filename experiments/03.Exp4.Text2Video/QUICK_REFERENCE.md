# Experiment 03: Quick Reference

## Service Management

### Start Services (Multiple Methods - Compatible with Exp07)

```bash
# Use defaults (4 Group A, 2 Group B, sleep models)
./start_all_services.sh

# Positional arguments (exp07 style)
./start_all_services.sh 10 6

# Positional with models
./start_all_services.sh 10 6 llm_service_small_model t2vid

# Environment variables (simulation)
N1=4 N2=2 ./start_all_services.sh

# Environment variables (real models)
N1=10 N2=6 MODEL_ID_A=llm_service_small_model MODEL_ID_B=t2vid ./start_all_services.sh

# Show help
./start_all_services.sh --help
```

### Stop Services
```bash
./stop_all_services.sh
```

### Check Service Health
```bash
curl http://localhost:8100/health  # Scheduler A
curl http://localhost:8200/health  # Scheduler B
curl http://localhost:8101/health  # Predictor
curl http://localhost:8202/health  # Planner
```

## Common Test Commands

### Basic Simulation Test
```bash
uv run python test_dynamic_workflow.py \
  --num-workflows 10 \
  --qps 5.0 \
  --strategies min_time \
  --mode simulation
```

### Real Model Test
```bash
uv run python test_dynamic_workflow.py \
  --num-workflows 24 \
  --qps 4.0 \
  --strategies min_time \
  --mode real
```

### Multiple Strategies
```bash
uv run python test_dynamic_workflow.py \
  --num-workflows 50 \
  --qps 6.0 \
  --strategies min_time round_robin probabilistic \
  --mode simulation
```

### With Global Rate Limiting
```bash
uv run python test_dynamic_workflow.py \
  --num-workflows 100 \
  --qps 8.0 \
  --gqps 20.0 \
  --strategies min_time
```

### With Warmup
```bash
uv run python test_dynamic_workflow.py \
  --num-workflows 100 \
  --qps 8.0 \
  --warmup 0.2 \
  --strategies min_time
```

### Continuous Mode
```bash
uv run python test_dynamic_workflow.py \
  --num-workflows 100 \
  --qps 8.0 \
  --continuous \
  --strategies min_time
```

## Workload Generation

### Generate Captions Cache
```bash
uv run python workload_generator.py \
  --num-captions 200 \
  --cache-path data/captions.jsonl \
  --stream-limit 5000
```

### Use Cached Captions
```bash
uv run python test_dynamic_workflow.py \
  --num-workflows 50 \
  --cache-path data/captions.jsonl
```

## Port Configuration

| Service | Default Port | Environment Variable |
|---------|-------------|---------------------|
| Scheduler A | 8100 | SCHEDULER_A_PORT |
| Scheduler B | 8200 | SCHEDULER_B_PORT |
| Predictor | 8101 | PREDICTOR_PORT |
| Planner | 8202 | PLANNER_PORT |
| Group A Instances | 8210+ | INSTANCE_GROUP_A_START_PORT |
| Group B Instances | 8300+ | INSTANCE_GROUP_B_START_PORT |

## Environment Variables

```bash
# Instance counts
N1=4              # Number of Group A instances
N2=2              # Number of Group B instances

# Model IDs
MODEL_ID_A=sleep_model_a                # For simulation
MODEL_ID_B=sleep_model_b                # For simulation
MODEL_ID_A=llm_service_small_model      # For real mode
MODEL_ID_B=t2vid                        # For real mode

# Planner
AUTO_OPTIMIZE_ENABLED=true              # Enable/disable auto optimization
AUTO_OPTIMIZE_INTERVAL=150              # Optimization interval (seconds)
```

## Results

### View Results
```bash
# Latest results
ls -lt results/*.json | head

# View specific result
cat results/text2video_min_time_20251120_123456.json | jq '.workflows'

# View all strategies
cat results/text2video_combined_20251120_123456.json | jq '.strategies | keys'
```

### Extract Metrics
```bash
# Workflow completion times
jq '.workflows.avg_workflow_time' results/text2video_min_time_*.json

# Task completion times
jq '.a1_tasks.p95_completion_time' results/text2video_min_time_*.json
jq '.a2_tasks.p95_completion_time' results/text2video_min_time_*.json
jq '.b_tasks.p95_completion_time' results/text2video_min_time_*.json

# Success rates
jq '.workflows.num_completed' results/text2video_min_time_*.json
```

## Troubleshooting

### Check Logs
```bash
# All logs
tail -f logs/*.log

# Specific service
tail -f logs/scheduler_a.log
tail -f logs/scheduler_b.log
tail -f logs/predictor.log
tail -f logs/planner.log
```

### Check Running Processes
```bash
# PIDs
cat logs/*.pid

# Processes
ps aux | grep -E "(scheduler|predictor|planner|instance)"
```

### Force Clean Restart
```bash
./stop_all_services.sh
pkill -f "scheduler|predictor|planner|instance"
rm -f logs/*.pid
sleep 5
./start_all_services.sh
```

## Scheduler API

### Submit Task
```bash
curl -X POST http://localhost:8100/task/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-task-001",
    "model_id": "llm_service_small_model",
    "task_input": {"sentence": "Generate a video of a cat"},
    "metadata": {"task_type": "A1"}
  }'
```

### Query Task
```bash
curl http://localhost:8100/task/query/test-task-001
```

### Clear Tasks
```bash
curl -X POST http://localhost:8100/task/clear
```

### Instance Status
```bash
curl http://localhost:8100/instance/status
```

## Quick Performance Test

```bash
# Small test (5 minutes)
uv run python test_dynamic_workflow.py \
  --num-workflows 20 \
  --qps 6.0 \
  --strategies min_time \
  --mode simulation \
  --timeout 5

# Medium test (15 minutes)
uv run python test_dynamic_workflow.py \
  --num-workflows 100 \
  --qps 8.0 \
  --strategies min_time round_robin \
  --mode simulation \
  --timeout 15

# Large test (30 minutes)
uv run python test_dynamic_workflow.py \
  --num-workflows 200 \
  --qps 10.0 \
  --strategies min_time round_robin probabilistic \
  --mode simulation \
  --gqps 25.0 \
  --timeout 30
```

## Development

### Run Syntax Check
```bash
python3 -m py_compile test_dynamic_workflow.py common.py workload_generator.py
```

### Test Workload Generation
```bash
uv run python workload_generator.py --num-captions 10
```

### Test Single Workflow
```bash
uv run python test_dynamic_workflow.py \
  --num-workflows 1 \
  --qps 1.0 \
  --strategies min_time \
  --mode simulation \
  --timeout 2
```
