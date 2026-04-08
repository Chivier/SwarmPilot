# Dataclaw Replay Benchmark

Replay [peteromallet/dataclaw-peteromallet](https://huggingface.co/datasets/peteromallet/dataclaw-peteromallet) coding agent sessions through a SwarmPilot cluster to benchmark scheduling latency with real model inference.

Dataclaw conversations contain user/assistant turns with model size assignment:
- First and last assistant response per user turn -> **large model**
- Intermediate assistant responses -> **small model**
- User follow-ups -> **large model**

## Deployment

Deploy models via the Planner SDK (see [MCP-Atlas README](../mcp_atlas/README.md) for full instructions):

```bash
# Start cluster
bash examples/gallery/replay/benchmark/start_cluster.sh

# Deploy models via Planner
splanner serve Qwen/Qwen3-Next-80B-A3B-Instruct --gpu 4 --replicas 2
splanner serve Qwen/Qwen3-VL-8B-Instruct --gpu 1 --replicas 2
```

## Usage

```bash
cd examples/gallery

# 1. Prepare dataset
python -m replay.cli prepare \
  --dataset dataclaw \
  --limit 100 \
  --output data/dataclaw.jsonl

# 2. Run replay
python -m replay.cli run \
  --data data/dataclaw.jsonl \
  --config dataclaw/config.example.yaml \
  --output results/dataclaw.json
```

## Configuration

See [`config.example.yaml`](config.example.yaml). Parameters are identical to MCP-Atlas; see [MCP-Atlas README](../mcp_atlas/README.md#configuration) for the full table.

## Dataset Characteristics

Compared to MCP-Atlas:
- **Longer conversations**: Dataclaw sessions tend to have more turns (user follow-ups)
- **More small-model steps**: Multiple intermediate tool-use turns between reasoning steps
- **User-initiated delays**: `user_delay_ms` impacts end-to-end latency more significantly
