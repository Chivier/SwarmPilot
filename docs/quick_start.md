# Quick Start

## Install

```bash
git clone <repo-url> swarmpilot-refresh && cd swarmpilot-refresh
uv sync    # installs sscheduler, spredictor, splanner CLIs
```

## Option A: Local GPU Cluster (Planner + PyLet + vLLM)

```bash
# 1. Start Planner with local PyLet cluster
PYLET_ENABLED="true" PYLET_LOCAL_MODE="true" PYLET_BACKEND="vllm" PYLET_GPU_COUNT="4" \
  splanner start --port 8002

# 2. Start Scheduler (registers with Planner, model assigned on first serve)
PLANNER_REGISTRATION_URL="http://localhost:8002" SCHEDULER_SELF_URL="http://localhost:8000" \
  sscheduler start --port 8000

# 3. Deploy a model
splanner serve "Qwen/Qwen3-8B" --gpu 4 --replicas 1

# 4. Send requests (OpenAI-compatible)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'

# 5. Stop
splanner terminate --all    # then Ctrl-C the Planner/Scheduler processes
```

## Option B: Online API (Cloud Providers, No GPU)

```bash
# 1. Create online_endpoints.yaml (see docs/cluster_deployment.md for format)
# 2. Start Scheduler with online endpoints
export TOGETHER_API_KEY="your-key"
SCHEDULER_MODEL_ID="Qwen/Qwen3-8B" ONLINE_ENDPOINTS_CONFIG="./online_endpoints.yaml" \
  sscheduler start --port 8000

# 3. Send requests — same curl as Option A
```

`concurrency_limit` in the YAML controls max concurrent requests per provider. See `examples/7.online_api/` for a working multi-provider example.

## Next Steps

[Cluster Deployment](cluster_deployment.md) | [SDK Usage](sdk_usage.md) | [Predictor](predictor.md)
