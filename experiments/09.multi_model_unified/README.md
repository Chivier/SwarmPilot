# Experiment 09: Unified Multi-Model Workflow Framework

## Overview

This experiment provides a unified framework integrating experiments 04-07 with mode selection for different workflow patterns. It consolidates common infrastructure while allowing users to switch between different execution modes via command-line parameters.

## Workflow Modes

### OCR Mode (Experiment 04) ✅ **IMPLEMENTED**
```
A → n B tasks (parallel) → Complete
```
- Each A task generates 3-8 B tasks (variable fanout)
- All B tasks execute in parallel
- Workflow completes when all B tasks finish
- **Status**: Fully implemented and tested

### T2IMG Mode (Experiment 05) 🚧 **TODO**
```
A → n B tasks (sequential) → Complete
```
- Each A task generates 3-8 B tasks (variable fanout)
- B tasks execute sequentially (one at a time)
- Next B task submitted only after previous completes
- Workflow time = A_time + Σ(B_times)
- **Status**: Use `experiments/05.multi_model_workflow_dynamic_parallel/` directly

### Merge Mode (Experiment 06) 🚧 **TODO**
```
A → n B tasks (parallel) → Merge A → Complete
```
- All B tasks execute in parallel
- After all B tasks complete, submit merge task
- Merge task execution time = 0.5× original A task time
- Workflow completes when merge task finishes
- **Status**: Use `experiments/06.multi_model_workflow_dynamic_merge/` directly

### DR Mode (Experiment 07) 🚧 **TODO**
```
A → n B1 tasks → n B2 tasks (pipelined) → Merge A → Complete
```
- B stage split into B1 (slow, 7-10s) and B2 (fast, 1-3s)
- Each B1 completion triggers paired B2 submission
- Pipelined B1→B2 execution
- After all B2 tasks complete, submit merge task
- Workflow completes when merge task finishes
- **Status**: Use `experiments/07.multi_model_workflow_dynamic_merge_2/` directly

## Quick Start

### 1. Start Services
```bash
# Start scheduler and instance services
./start_all_services.sh

# Default configuration:
# - N1=10 Group A instances (ports 8210-8219)
# - N2=6 Group B instances (ports 8300-8305)
# - Scheduler A on port 8100
# - Scheduler B on port 8200
```

### 2. Run Experiment (OCR Mode)
```bash
# Run with default settings
python test_unified_workflow.py --mode ocr

# Run with custom parameters
python test_unified_workflow.py --mode ocr --num-workflows 200 --qps 10.0 --seed 123

# Run with global QPS limit
python test_unified_workflow.py --mode ocr --gqps 15.0

# Run with warmup
python test_unified_workflow.py --mode ocr --warmup 0.2  # 20% warmup tasks

# Run specific strategies only
python test_unified_workflow.py --mode ocr --strategies round_robin min_time
```

### 3. Stop Services
```bash
./stop_all_services.sh
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | **required** | Workflow mode: `ocr`, `t2img`, `merge`, `dr` |
| `--num-workflows` | int | 100 | Number of workflows per strategy |
| `--qps` | float | 8.0 | Target QPS for A task submission |
| `--gqps` | float | None | Global QPS limit (A+B tasks) |
| `--warmup` | float | 0.0 | Warmup ratio (0.0-1.0) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--strategies` | list | all | Strategies to test: `round_robin`, `min_time`, `probabilistic` |

## Workload Characteristics

### A Tasks (All Modes)
- **Distribution**: Bimodal
  - 50% fast: 1-3 seconds (mean=2.0s)
  - 50% slow: 7-10 seconds (mean=8.5s)

### B Tasks

#### OCR/T2IMG/Merge Modes
- **Distribution**: Bimodal
  - 50% fast: 1-3 seconds (mean=2.0s)
  - 50% slow: 8-12 seconds (mean=10.0s)

#### DR Mode (B1/B2 Split)
- **B1 Tasks**: Slow peak only
  - Range: 7-10 seconds (mean=8.5s)
- **B2 Tasks**: Fast peak only
  - Range: 1-3 seconds (mean=2.0s)

### Fanout Distribution
- **Range**: 3-8 B tasks per A task
- **Distribution**: Uniform
- **Mean**: ~5.5 B tasks per workflow

### Merge Tasks (Merge/DR Modes)
- **Execution Time**: 0.5× original A task time
- Simulates aggregation/finalization work

## Architecture

### Thread Structure

#### OCR Mode (4 Threads)
1. **Thread 1**: A Task Submitter (Poisson process)
2. **Thread 2**: A Result Receiver + B Task Submitter
3. **Thread 3**: B Result Receiver
4. **Thread 4**: Workflow Monitor + Statistics

#### T2IMG Mode (4 Threads)
- Same as OCR, but Thread 3 implements sequential B submission logic

#### Merge Mode (6 Threads)
1-3. Same as OCR mode
4. **Thread 5**: Merge Task Submitter
5. **Thread 6**: Merge Result Receiver
6. **Thread 7**: Workflow Monitor + Statistics

#### DR Mode (7 Threads)
1-2. Same as OCR mode
3. **Thread 3**: B1 Result Receiver + B2 Task Submitter
4. **Thread 4**: B2 Result Receiver
5-7. Same threads 5-7 as Merge mode

## Results and Metrics

### Output Files
Results are saved to `results/results_workflow_{mode}_{timestamp}.json`

### Metrics Tracked
- **A Tasks**: Submission time, completion time, execution time
- **B Tasks**: Submission time, completion time, execution time, assigned instance
- **Workflows**: Total workflow time (A submit → final task complete)
- **Completion Rates**: Per-strategy workflow completion statistics

### Statistics Reported
- Average completion times
- P50, P80, P90, P95, P99 percentiles
- Min/Max times
- Standard deviation
- Completion counts

## Implementation Status

### ✅ Completed Components
- [x] Unified directory structure
- [x] Unified workload generator supporting all modes
- [x] OCR mode (fully functional)
- [x] Mode selection via command-line
- [x] Shared infrastructure (Rate Limiter, WebSocket handlers, etc.)
- [x] Documentation (README, QUICK_REFERENCE)

### 🚧 TODO: Future Work
- [ ] T2IMG mode integration (sequential B submission)
- [ ] Merge mode integration (add merge task threads)
- [ ] DR mode integration (B1/B2 split + merge)
- [ ] Mode-specific statistics and visualization
- [ ] Unified result comparison across modes

## File Structure

```
experiments/09.multi_model_unified/
├── test_unified_workflow.py      # Main experiment script (OCR mode implemented)
├── workload_generator.py          # Unified workload generation
├── start_all_services.sh          # Start scheduler + instances
├── stop_all_services.sh           # Stop all services
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── QUICK_REFERENCE.md            # Quick command reference
├── .gitignore                    # Git ignore rules
├── logs/                         # Service logs (generated)
└── results/                      # Experiment results (generated)
```

## Comparison with Original Experiments

| Feature | Exp 04 | Exp 05 | Exp 06 | Exp 07 | Exp 09 |
|---------|--------|--------|--------|--------|--------|
| Parallel B | ✅ | ❌ | ✅ | ✅ | ✅ (OCR) |
| Sequential B | ❌ | ✅ | ❌ | ❌ | 🚧 (T2IMG) |
| Merge Task | ❌ | ❌ | ✅ | ✅ | 🚧 (Merge) |
| B1/B2 Split | ❌ | ❌ | ❌ | ✅ | 🚧 (DR) |
| Mode Selection | ❌ | ❌ | ❌ | ❌ | ✅ |
| Unified Code | ❌ | ❌ | ❌ | ❌ | ✅ |

## Troubleshooting

### Services Won't Start
```bash
# Check if ports are already in use
lsof -i :8100 -i :8200

# Kill existing processes
./stop_all_services.sh
pkill -f "python.*scheduler"
pkill -f "python.*instance"
```

### WebSocket Connection Errors
- Ensure schedulers are running (`ps aux | grep scheduler`)
- Check scheduler logs in `logs/` directory
- Verify firewall settings allow local connections

### Task Submission Failures
- Check instance availability: `curl http://localhost:8100/instance/list`
- Verify model registration
- Review instance logs for errors

## References

- **Original Experiments**:
  - Experiment 04: `../04.multi_model_workflow_dynamic/`
  - Experiment 05: `../05.multi_model_workflow_dynamic_parallel/`
  - Experiment 06: `../06.multi_model_workflow_dynamic_merge/`
  - Experiment 07: `../07.multi_model_workflow_dynamic_merge_2/`

- **Related Documentation**:
  - `QUICK_REFERENCE.md` - Quick command examples
  - Each original experiment has its own README with detailed explanations

## Contact

For questions or issues, please refer to the individual experiment READMEs or consult the project documentation.
