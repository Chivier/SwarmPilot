# Dataclaw Replay Benchmark

Replay [peteromallet/dataclaw-peteromallet](https://huggingface.co/datasets/peteromallet/dataclaw-peteromallet) coding agent sessions through a SwarmPilot cluster to benchmark scheduling latency with real model inference.

Dataclaw conversations contain user/assistant turns with model size assignment:
- First and last assistant response per user turn -> **large model**
- Intermediate assistant responses -> **small model**
- User follow-ups -> **large model**

## Quick Start

All commands run from within this directory (`gallery/dataclaw/`).

### 1. Start SwarmPilot Cluster

```bash
bash scripts/start_cluster.sh
```

### 2. Prepare Dataset

```bash
python -m replay.cli prepare \
  --limit 100 \
  --output data/dataclaw.jsonl
```

### 3. Run Replay

Models are deployed automatically when the `planner` section is present in the config. No manual deployment needed.

```bash
python -m replay.cli run \
  --data data/dataclaw.jsonl \
  --config config.example.yaml \
  --output results/dataclaw.json
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

## Dataset Characteristics

Compared to MCP-Atlas:
- **Longer conversations**: Dataclaw sessions tend to have more turns (user follow-ups)
- **More small-model steps**: Multiple intermediate tool-use turns between reasoning steps
- **User-initiated delays**: `user_delay_ms` impacts end-to-end latency more significantly
