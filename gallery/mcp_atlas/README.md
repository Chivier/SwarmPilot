# MCP-Atlas Replay Benchmark

Replay [ScaleAI/MCP-Atlas](https://huggingface.co/datasets/ScaleAI/MCP-Atlas) agent conversations through a SwarmPilot cluster to benchmark scheduling latency with real model inference.

MCP-Atlas conversations follow a **large + small model** pattern: the assistant (reasoning) routes to a large model, and tool calls route to a small model.

## Architecture

```
HuggingFace Dataset
    |
    v
[Reshaper] --> ReplayGroups (JSONL)
    |
    v
[ReplayScheduler]  (Poisson arrival + token-bucket QPS)
    |
    +---> [AsyncOpenAI] ---> SwarmPilot Scheduler (large) :8000/v1/chat/completions
    |                              |
    |                              v
    |                         vLLM / Model Backend (80B)
    |
    +---> [AsyncOpenAI] ---> SwarmPilot Scheduler (small) :8010/v1/chat/completions
                                   |
                                   v
                              vLLM / Model Backend (8B)
```

## Quick Start

All commands run from within this directory (`gallery/mcp_atlas/`).

### 1. Start SwarmPilot Cluster

```bash
bash scripts/start_cluster.sh
```

### 2. Prepare Dataset

```bash
python -m replay.cli prepare \
  --limit 100 \
  --output data/mcp-atlas.jsonl
```

### 3. Run Replay

Models are deployed automatically when the `planner` section is present in the config. No manual deployment needed.

```bash
python -m replay.cli run \
  --data data/mcp-atlas.jsonl \
  --config config.example.yaml \
  --output results/mcp-atlas.json
```

### 4. Stop Cluster

```bash
bash scripts/stop_cluster.sh
```

## Configuration

See [`config.example.yaml`](config.example.yaml) for all parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `large_model.base_url` | SwarmPilot scheduler URL for large model | — |
| `small_model.base_url` | SwarmPilot scheduler URL for small model | — |
| `poisson_qps` | Poisson arrival rate (groups/sec) | 0.1 |
| `global_qps` | Global QPS ceiling across all requests | 5.0 |
| `agent_delay_ms` | Inter-step delay after agent response | 100 |
| `user_delay_ms` | Inter-step delay after user response | 5000 |
| `timeout_s` | Per-request timeout | 120.0 |
| `max_tokens` | Max tokens per request (1 = latency-only) | 1 |

## How Requests Flow Through SwarmPilot

1. The replay client sends OpenAI-compatible `POST /v1/chat/completions` to the scheduler
2. The scheduler selects an instance using its scheduling strategy (e.g., `adaptive_bootstrap`)
3. The scheduler proxies the request to the selected vLLM backend
4. The response is returned through the scheduler to the replay client
5. Latency metrics are recorded per-request and per-group

## Output

Results JSON contains per-request latency metrics and aggregate statistics:

```
REPLAY SUMMARY
============================================================
Groups: 100
Requests: 1523 total, 1520 success, 3 failed

  All requests: p50=142ms p90=312ms p99=891ms mean=198ms
  Large model:  p50=203ms p90=445ms p99=1102ms mean=281ms
  Small model:  p50=89ms  p90=167ms p99=423ms mean=112ms
  Group e2e:    p50=4231ms p90=8912ms p99=15234ms mean=5102ms
```
