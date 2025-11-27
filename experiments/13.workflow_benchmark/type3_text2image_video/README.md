# Type3: Text2Image+Video Workflow

> **Workflow Pattern**: `LLM (A) → FLUX (C) → T2VID (B loops)`

A three-stage workflow that generates videos from text captions through image generation.

## Quick Start (5 minutes)

### 1. Start Services (Simulation Mode)

```bash
cd experiments/13.workflow_benchmark

# Start all services (default: 4 LLM + 2 FLUX + 2 T2VID instances)
./type3_text2image_video/scripts/start_type3_sim_services.sh

# Or customize instance counts
N1=6 N2=3 N3=3 ./type3_text2image_video/scripts/start_type3_sim_services.sh
```

### 2. Run Experiment

```bash
# Basic run
uv run python tools/cli.py run-text2image-video-sim \
    --num-workflows 50 --qps 2.0

# With specific strategies
uv run python tools/cli.py run-text2image-video-sim \
    --num-workflows 100 --qps 2.0 --strategies probabilistic,min_time

# With resolution distribution (70% 512x512, 30% 1024x1024)
uv run python tools/cli.py run-text2image-video-sim \
    --num-workflows 50 --resolution-config type3_text2image_video/configs/resolution_70_30.json
```

### 3. Stop Services

```bash
./type3_text2image_video/scripts/stop_type3_services.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Type3 Workflow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Caption ──► [A: LLM] ──► [C: FLUX] ──► [B: T2VID] ──► Video  │
│              (prompt)     (image)       (1-N loops)             │
│                 │            │              │                   │
│                 ▼            ▼              ▼                   │
│           Scheduler A   Scheduler C   Scheduler B               │
│           (port 8100)   (port 8300)   (port 8200)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Workflow Steps:**
1. **A (LLM)**: Generates positive prompt from caption
2. **C (FLUX)**: Generates image from prompt (512×512 or 1024×1024)
3. **B (T2VID)**: Generates video frames (1-N iterations, negative_prompt="blur")

## CLI Reference

```bash
uv run python tools/cli.py run-text2image-video-sim [OPTIONS]
uv run python tools/cli.py run-text2image-video-real [OPTIONS]
```

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-workflows` | 10 | Number of workflows to run |
| `--qps` | 2.0 | Queries per second |
| `--duration` | 120 | Max duration (seconds) |
| `--strategies` | default | Strategies: `probabilistic,min_time,round_robin,random,po2` |
| `--warmup` | 0.2 | Warmup ratio (0.0-1.0) |
| `--portion-stats` | 1.0 | Portion for statistics |

### Type3-Specific Options

| Option | Default | Description |
|--------|---------|-------------|
| `--resolution` | 512 | FLUX resolution (512 or 1024) |
| `--resolution-config` | - | JSON config for resolution distribution |
| `--max-b-loops` | 3 | Max T2VID iterations per workflow |
| `--max-b-loops-config` | - | JSON config for loop distribution |
| `--frame-count` | 16 | Frames per video |
| `--frame-count-config` | - | JSON config for frame distribution |

## Configuration Files

Pre-built configs in `type3_text2image_video/configs/`:

```bash
# Resolution distributions
resolution_static_512.json   # All 512×512
resolution_static_1024.json  # All 1024×1024
resolution_70_30.json        # 70% 512, 30% 1024
resolution_50_50.json        # 50/50 mix

# Frame count distributions
frame_count_static.json      # Fixed 16 frames
frame_count_uniform.json     # Uniform 8-24
frame_count_two_peak.json    # Peaks at 12 and 20
frame_count_four_peak.json   # Multi-modal

# B-loop distributions
max_b_loops_static.json      # Fixed 3 loops
max_b_loops_uniform.json     # Uniform 1-4
max_b_loops_two_peak.json    # Peaks at 2 and 4
```

## Examples

```bash
# Quick test (10 workflows)
uv run python tools/cli.py run-text2image-video-sim

# Production benchmark
uv run python tools/cli.py run-text2image-video-sim \
    --num-workflows 200 \
    --qps 3.0 \
    --duration 600 \
    --strategies all \
    --resolution-config type3_text2image_video/configs/resolution_70_30.json \
    --max-b-loops-config type3_text2image_video/configs/max_b_loops_two_peak.json

# Compare strategies
for strategy in probabilistic min_time round_robin; do
    uv run python tools/cli.py run-text2image-video-sim \
        --num-workflows 100 --strategies $strategy
done
```

## Output

Metrics are saved to `output/metrics_<strategy>.json`:

```json
{
  "workflow_metrics": [...],
  "task_metrics": [...],
  "summary": {
    "total_workflows": 100,
    "completed_workflows": 98,
    "avg_latency_ms": 12500,
    "p50_latency_ms": 11200,
    "p99_latency_ms": 18900
  }
}
```

## Real Cluster Mode

For production deployment on multi-node clusters:

```bash
# 1. Start services on each node
bash type3_text2image_video/scripts/start_type3_real_services.sh --auto-optimize

# 2. Deploy models (from client node)
bash type3_text2image_video/scripts/manual_deploy_type3.sh

# 3. Run experiment
uv run python tools/cli.py run-text2image-video-real \
    --num-workflows 100 --qps 2.0
```

## Troubleshooting

**Services won't start?**
```bash
# Check logs
tail -f type3_text2image_video/logs/*.log

# Kill orphan processes
./type3_text2image_video/scripts/stop_type3_services.sh
pkill -f "scheduler.*start"
```

**Low throughput?**
- Increase instance counts: `N1=8 N2=4 N3=4 ./start_type3_sim_services.sh`
- Check scheduler queues: `curl http://localhost:8100/tasks`

**Experiment timeout?**
- Increase `--duration`
- Reduce `--num-workflows`
- Lower `--qps`
