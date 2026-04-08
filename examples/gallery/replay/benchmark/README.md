# Real-Model Replay Benchmark

Replay real LLM agent conversations through a SwarmPilot cluster with
real vLLM model instances for latency benchmarking.

## Prerequisites

- SwarmPilot installed (`uv sync` in project root)
- PyLet installed (`uv pip install pylet`)
- GPU(s) available for vLLM model serving
- Prepared replay data (output of `replay prepare`)

## Quick Start

```bash
# 1. Prepare replay data from a HuggingFace dataset
uv run python -m replay.cli prepare \
    --dataset mcp-atlas \
    --limit 50 \
    --output data/mcp-atlas.jsonl

# 2. Start the SwarmPilot cluster (Planner + 2 Schedulers)
bash examples/gallery/replay/benchmark/start_cluster.sh

# 3. Deploy real vLLM model instances
bash examples/gallery/replay/benchmark/deploy_models.sh

# 4. Run the benchmark
uv run python examples/gallery/replay/benchmark/benchmark_runner.py \
    --data data/mcp-atlas.jsonl \
    --config examples/gallery/replay/benchmark/config.yaml \
    --output results/mcp-atlas \
    --warmup 5 \
    --limit 50

# 5. Stop the cluster
bash examples/gallery/replay/benchmark/stop_cluster.sh
```

## Configuration

Edit `config.yaml` to configure:

- **Model IDs**: HuggingFace model identifiers for large/small models
- **GPU allocation**: Tensor-parallel size and replica count per model
- **Scheduling**: Algorithm (probabilistic, round_robin, min_time, etc.)
- **QPS control**: Independent token bucket rates for large/small models
- **Timing**: Poisson arrival rate, inter-step delays, request timeout

## Architecture

```
ReplayGroups (real prompts)
    │
    ▼
benchmark_runner.py
    ├── Poisson arrival scheduling
    ├── Cumulative message history per group
    ├── Burst-mode for consecutive small-model steps
    ├── Dual TokenBucket QPS limiting (large/small)
    └── exp_runtime estimation from token count
    │
    ▼
SwarmPilot Schedulers (transparent proxy)
    ├── Scheduler-Large (:8010) → large model instances
    └── Scheduler-Small (:8020) → small model instances
    │
    ▼
Real vLLM Instances (deployed via PyLet)
```

## Output

Results are streamed to three JSONL files:
- `{output}-large.jsonl` — per-request latencies for large model
- `{output}-small.jsonl` — per-request latencies for small model
- `{output}-e2e.jsonl` — per-group end-to-end latencies

Each file ends with a summary line containing percentile statistics.

## Differences from Mock Benchmark

| Aspect | Mock Benchmark | This Benchmark |
|--------|---------------|----------------|
| Model instances | Sleep servers (simulated) | Real vLLM (GPU inference) |
| Request content | `sleep_time_ms:523.0` | Real conversation history |
| Latency source | Controlled sleep duration | Actual model inference |
| Deployment | `sleep_server.py` processes | `splanner serve` via PyLet |
| Runtime prediction | Exact (from mock data) | Estimated from token count |
