# Quick Reference - Experiment 09

## Service Management

### Start Services
```bash
# Default (N1=10 Group A instances, N2=6 Group B instances)
./start_all_services.sh

# Custom instance counts
N1=15 N2=10 ./start_all_services.sh

# With specific configuration
MODEL_PORT=9999 CONTAINER_NAME=test_model N1=10 N2=6 ./start_all_services.sh
```

### Stop Services
```bash
./stop_all_services.sh

# Force kill if needed
pkill -9 -f "python.*scheduler"
pkill -9 -f "python.*instance"
```

### Check Service Status
```bash
# Check running processes
ps aux | grep -E "(scheduler|instance)" | grep -v grep

# Check scheduler A
curl http://localhost:8100/health

# Check scheduler B
curl http://localhost:8200/health

# List instances
curl http://localhost:8100/instance/list | python -m json.tool
curl http://localhost:8200/instance/list | python -m json.tool
```

## Running Experiments

### OCR Mode (Implemented ✅)
```bash
# Basic usage
python test_unified_workflow.py --mode ocr

# Custom workflows and QPS
python test_unified_workflow.py --mode ocr --num-workflows 200 --qps 10.0

# Global QPS mode (controls both A and B submissions)
python test_unified_workflow.py --mode ocr --gqps 20.0

# With warmup tasks
python test_unified_workflow.py --mode ocr --warmup 0.2  # 20% warmup

# Custom seed
python test_unified_workflow.py --mode ocr --seed 123

# Specific strategies only
python test_unified_workflow.py --mode ocr --strategies round_robin
python test_unified_workflow.py --mode ocr --strategies min_time probabilistic

# Full custom run
python test_unified_workflow.py \
  --mode ocr \
  --num-workflows 500 \
  --qps 15.0 \
  --warmup 0.1 \
  --seed 42 \
  --strategies round_robin min_time
```

### Other Modes (Not Yet Implemented 🚧)
```bash
# T2IMG mode - Use original experiment instead
cd ../05.multi_model_workflow_dynamic_parallel
python test_dynamic_workflow.py

# Merge mode - Use original experiment instead
cd ../06.multi_model_workflow_dynamic_merge
python test_dynamic_workflow.py

# DR mode - Use original experiment instead
cd ../07.multi_model_workflow_dynamic_merge_2
python test_dynamic_workflow.py
```

## Workload Generator

### Test Workload Generation
```bash
# Generate and view workload statistics
python workload_generator.py

# Custom parameters
python workload_generator.py --num-tasks 500 --num-workflows 200 --seed 42
```

## Results

### View Results
```bash
# List result files
ls -lh results/

# View latest result
cat results/results_workflow_ocr_*.json | python -m json.tool | less

# Extract key metrics
cat results/results_workflow_ocr_*.json | \
  python -c "import json, sys; data=json.load(sys.stdin); \
  print(f\"Workflows: {data['config']['num_workflows']}\"); \
  print(f\"QPS: {data['config']['qps_a']}\"); \
  [print(f\"{r['strategy']}: {r['workflows']['avg_workflow_time']:.2f}s avg\") \
   for r in data['results']]"
```

## Common Scenarios

### Quick Test (Small Scale)
```bash
./start_all_services.sh
python test_unified_workflow.py --mode ocr --num-workflows 50 --qps 5.0
./stop_all_services.sh
```

### Production-Like Test (Large Scale)
```bash
N1=20 N2=15 ./start_all_services.sh
python test_unified_workflow.py \
  --mode ocr \
  --num-workflows 1000 \
  --gqps 30.0 \
  --warmup 0.1 \
  --seed 42
./stop_all_services.sh
```

### Strategy Comparison
```bash
./start_all_services.sh
python test_unified_workflow.py \
  --mode ocr \
  --num-workflows 200 \
  --qps 8.0 \
  --strategies round_robin min_time probabilistic
./stop_all_services.sh
```

### Reproducibility Test
```bash
# Run 1
python test_unified_workflow.py --mode ocr --seed 42 --num-workflows 100

# Run 2 (should get identical workload)
python test_unified_workflow.py --mode ocr --seed 42 --num-workflows 100
```

## Debugging

### View Logs
```bash
# Tail scheduler A logs
tail -f logs/scheduler_a.log

# Tail scheduler B logs
tail -f logs/scheduler_b.log

# View instance logs
tail -f logs/instance_a_8210.log
tail -f logs/instance_b_8300.log

# Search for errors
grep -i error logs/*.log
grep -i exception logs/*.log
```

### Manual Task Submission (Testing)
```bash
# Submit a task to Scheduler A
curl -X POST http://localhost:8100/task/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "model_id": "sleep_model",
    "task_input": {"sleep_time": 2.0}
  }'

# Query task status
curl http://localhost:8100/task/query/test-001 | python -m json.tool
```

### Reset Environment
```bash
# Stop all services
./stop_all_services.sh

# Clean logs
rm -rf logs/*.log

# Clean results
rm -rf results/*.json

# Restart fresh
./start_all_services.sh
```

## Performance Tuning

### Adjust Instance Counts
```bash
# More A instances (better for A-heavy workloads)
N1=20 N2=6 ./start_all_services.sh

# More B instances (better for high-fanout workloads)
N1=10 N2=15 ./start_all_services.sh

# Balanced
N1=15 N2=15 ./start_all_services.sh
```

### Adjust QPS
```bash
# Low load
python test_unified_workflow.py --mode ocr --qps 2.0

# Medium load
python test_unified_workflow.py --mode ocr --qps 8.0

# High load
python test_unified_workflow.py --mode ocr --gqps 50.0
```

## Parameter Reference

| Parameter | OCR | T2IMG | Merge | DR | Description |
|-----------|-----|-------|-------|----|----|
| `--mode` | ✅ | 🚧 | 🚧 | 🚧 | Workflow execution mode |
| `--num-workflows` | ✅ | 🚧 | 🚧 | 🚧 | Workflows per strategy |
| `--qps` | ✅ | 🚧 | 🚧 | 🚧 | A task submission rate |
| `--gqps` | ✅ | 🚧 | 🚧 | 🚧 | Global QPS limit (A+B) |
| `--warmup` | ✅ | 🚧 | 🚧 | 🚧 | Warmup task ratio |
| `--seed` | ✅ | 🚧 | 🚧 | 🚧 | Random seed |
| `--strategies` | ✅ | 🚧 | 🚧 | 🚧 | Strategy selection |

Legend: ✅ Implemented | 🚧 Use original experiment

## Quick Tips

1. **Always start services before running experiments**
   ```bash
   ./start_all_services.sh
   ```

2. **Use warmup for realistic measurements**
   ```bash
   --warmup 0.2  # Exclude first 20% from stats
   ```

3. **Use consistent seeds for reproducibility**
   ```bash
   --seed 42
   ```

4. **Monitor logs during execution**
   ```bash
   tail -f logs/*.log
   ```

5. **Stop services when done**
   ```bash
   ./stop_all_services.sh
   ```

## File Locations

- **Scripts**: `test_unified_workflow.py`, `workload_generator.py`
- **Service Management**: `start_all_services.sh`, `stop_all_services.sh`
- **Logs**: `logs/scheduler_*.log`, `logs/instance_*.log`
- **Results**: `results/results_workflow_{mode}_{timestamp}.json`
- **Documentation**: `README.md`, `QUICK_REFERENCE.md`
