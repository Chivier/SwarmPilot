# Experiment 03: Text2Video Workflow (A1→A2→B)

## Overview

This experiment tests a linear workflow for text-to-video generation using two models:
- **Model A** (llm_service_small_model): Generates video prompts
  - **A1 task**: Generate positive prompt from caption
  - **A2 task**: Generate negative prompt from positive prompt
- **Model B** (t2video): Generates video from both prompts
  - **B task**: Generate video frames using positive and negative prompts

**Workflow Pattern**: A1 → A2 → B (linear chain, no fanout, no merge)

## Architecture

### Thread Model (4 threads)
1. **Thread 1**: A1 Task Submitter (Poisson process, QPS-controlled)
2. **Thread 2**: A1 Result Receiver → A2 Task Submitter
3. **Thread 3**: A2 Result Receiver → B Task Submitter  
4. **Thread 4**: B Result Receiver → Workflow Completion

### Services
- **Predictor** (Port 8101): Predicts task execution times
- **Scheduler A** (Port 8100): Manages A1 and A2 tasks (LLM service)
- **Scheduler B** (Port 8200): Manages B tasks (video generation)
- **Planner** (Port 8202): Optional AI-based optimization
- **Instances**:
  - Group A (Ports 8210-82xx): llm_service_small_model instances
  - Group B (Ports 8300-83xx): t2video instances

## Quick Start

### 1. Start Services

**Interface compatible with Experiment 07:**

```bash
# Method 1: Use defaults (4 Group A, 2 Group B, sleep models)
./start_all_services.sh

# Method 2: Positional arguments
./start_all_services.sh 10 6                              # 10 Group A, 6 Group B

# Method 3: Positional with model IDs
./start_all_services.sh 10 6 llm_service_small_model t2vid

# Method 4: Environment variables (simulation)
N1=4 N2=2 ./start_all_services.sh

# Method 5: Environment variables (real models)
N1=10 N2=6 MODEL_ID_A=llm_service_small_model MODEL_ID_B=t2vid ./start_all_services.sh

# Show help
./start_all_services.sh --help
```

### 2. Run Tests

```bash
# Simulation test (quick validation)
uv run python test_dynamic_workflow.py \
  --num-workflows 10 \
  --qps 5.0 \
  --strategies min_time \
  --mode simulation

# Real model test
uv run python test_dynamic_workflow.py \
  --num-workflows 24 \
  --qps 4.0 \
  --strategies min_time round_robin \
  --mode real

# With global rate limiting
uv run python test_dynamic_workflow.py \
  --num-workflows 50 \
  --qps 8.0 \
  --gqps 20.0 \
  --strategies min_time

# With warmup and continuous mode
uv run python test_dynamic_workflow.py \
  --num-workflows 100 \
  --qps 8.0 \
  --warmup 0.2 \
  --continuous \
  --strategies min_time
```

### 3. Stop Services

```bash
./stop_all_services.sh
```

## Command-Line Options

```
--num-workflows    Number of workflows to execute per strategy (default: 24)
--qps              Target QPS for A1 task submission (default: 6.0)
--strategies       Scheduling strategies to test (default: min_time)
                   Choices: min_time, round_robin, probabilistic, random, po2
--mode             Test mode: simulation or real (default: simulation)
--model-a-id       Model ID for A1/A2 tasks (default: llm_service_small_model)
--model-b-id       Model ID for B tasks (default: t2vid)
--gqps             Global QPS limit for all task submissions (optional)
--warmup           Warmup task ratio 0.0-1.0 (default: 0.0)
--continuous       Enable continuous request mode
--timeout          Test timeout in minutes (default: 30)
--output-dir       Output directory for results (default: results)
```

## Rate Limiting

### QPS (A1 Arrival Rate)
- Controls A1 task submission using Poisson process
- Simulates realistic user request patterns
- Example: `--qps 8.0` → 8 A1 tasks per second

### Global QPS (All Submissions)
- Applies token bucket rate limiting to ALL submissions (A1, A2, B)
- Prevents system overload
- Example: `--gqps 20.0` → maximum 20 total task submissions per second

### Combined Example
```bash
# A1 arrives at 8 QPS, but total submission capped at 20 QPS
uv run python test_dynamic_workflow.py \
  --num-workflows 100 \
  --qps 8.0 \
  --gqps 20.0
```

## Workload Generation

Captions are sampled from the OpenVid-1M dataset:
- **Default**: Stream from `nkp37/OpenVid-1M` (requires internet)
- **Cached**: Use local JSONL file for faster iteration

Frame counts follow a four-peak distribution:
- **16 frames**: 40% (fast generation)
- **24 frames**: 30% (standard)
- **32 frames**: 20% (high quality)
- **48 frames**: 10% (maximum quality)

## Results

Results are saved in JSON format to the `results/` directory:
- **Per-strategy files**: `text2video_{strategy}_{timestamp}.json`
- **Combined file**: `text2video_combined_{timestamp}.json`

### Metrics Collected

#### Task-Level Metrics (A1, A2, B)
- Completion times (avg, median, P95, P99)
- Success/failure counts
- Task execution statistics

#### Workflow-Level Metrics
- End-to-end workflow time (A1 submit → B complete)
- Workflow completion rates
- Percentile distributions

#### Continuous Mode Metrics
- **Makespan**: Time from first target workflow to last target completion
- Workflow counts: warmup, target, overflow
- Throughput analysis

## File Structure

```
03.Exp4.Text2Video/
├── test_dynamic_workflow.py    # Main test runner (replaces old test_*.py)
├── common.py                   # Shared utilities (makespan, continuous mode)
├── workload_generator.py       # Caption sampling and frame generation
├── start_all_services.sh       # Service orchestration (with deploy)
├── start_all_services_no_deploy.sh  # Without auto redeploy
├── stop_all_services.sh        # Service cleanup
├── manual_deploy_planner.sh    # Manual planner deployment
├── deploy_models.sh            # Model deployment script
├── redeploy.py                 # Planner migration helper
├── README.md                   # This file
├── QUICK_REFERENCE.md          # Command reference
├── requirements.txt            # Python dependencies
├── data/                       # Dataset directory
│   ├── captions.jsonl          # Cached captions (optional)
│   └── frame_distribution.json # Frame statistics
└── results/                    # Test results (generated)
```

## Implementation Details

### Data Structures

**WorkflowTaskData**: Pre-generated task data
```python
@dataclass
class WorkflowTaskData:
    task_id: str
    workflow_id: str
    task_type: str  # "A1", "A2", or "B"
    caption: str    # For A1
    frame_count: int  # For B
    is_warmup: bool
```

**WorkflowState**: Real-time state tracking
```python
@dataclass
class WorkflowState:
    workflow_id: str
    a1_task_id: str
    a2_task_id: str
    b_task_id: str
    a1_submit_time: float
    a1_complete_time: float
    a2_submit_time: float
    a2_complete_time: float
    b_submit_time: float
    b_complete_time: float
```

### Threading Architecture

All threads use async/await for WebSocket communication:
- **Thread 1**: Synchronous Poisson submission
- **Threads 2-4**: Async WebSocket receivers with task submission

### Task Dependencies

```
A1 (Caption)
  ↓ (extracts positive prompt)
A2 (Positive Prompt)
  ↓ (extracts negative prompt)
B (Positive + Negative → Video)
  ↓
Workflow Complete
```

## Comparison with Experiment 07

| Aspect | Exp07 (Deep Research) | Exp03 (Text2Video) |
|--------|----------------------|-------------------|
| **Workflow** | A → {B1} → {B2} → Merge | A1 → A2 → B |
| **Fanout** | Variable (3-8 tasks) | None (1:1:1) |
| **Threads** | 7 threads | 4 threads |
| **Models** | 2 models (A and B) | 2 models (A and B) |
| **Complexity** | Fan-out/fan-in | Linear chain |
| **Use Case** | Parallel search & analysis | Sequential prompt → video |

## Troubleshooting

### Services won't start
```bash
# Check logs
tail -f logs/*.log

# Check ports
lsof -i :8100 -i :8200 -i :8101 -i :8202

# Force stop and restart
./stop_all_services.sh
sleep 5
./start_all_services.sh
```

### WebSocket connection errors
- Ensure schedulers are healthy: `curl http://localhost:8100/health`
- Check firewall settings
- Increase connection timeout in code

### Caption sampling fails
```bash
# Use local cache instead
python workload_generator.py --num-captions 100 --cache-path data/captions.jsonl

# Then use cache in tests
python test_dynamic_workflow.py --cache-path data/captions.jsonl
```

## References

- [OpenVid-1M Dataset](https://huggingface.co/datasets/nkp37/OpenVid-1M)
- [Experiment 07 (Pattern Reference)](../07.Exp2.Deep_Research_Real/)
- [SwarmPilot Documentation](../../README.md)
