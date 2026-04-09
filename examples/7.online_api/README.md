# Online API Multi-Provider Scheduling

Load-balance the same model across multiple cloud inference providers (Together AI, Fireworks AI, Lepton AI). No local GPUs needed.

## Quick Start

```bash
./examples/7.online_api/start_cluster.sh        # mock servers + 3 schedulers
python examples/7.online_api/api_example.py      # send 9 requests (3 models x 3 prompts)
./examples/7.online_api/stop_cluster.sh          # tear down
```

## YAML Config

Each scheduler loads a YAML file listing its providers:

```yaml
endpoints:
  - name: "together-qwen3-8b"
    base_url: "https://api.together.xyz"
    api_key_env: "TOGETHER_API_KEY"        # env var holding the API key
    model_id: "Qwen/Qwen3-8B"
    concurrency_limit: 15                  # max concurrent requests to this provider
```

`concurrency_limit` creates N **virtual instances** per provider, each with a serial queue. Total concurrent requests = N. Traffic is proportional: a provider with limit 15 gets 3x the traffic of one with limit 5.

## Using Real Providers

```bash
export TOGETHER_API_KEY="your-key"
export FIREWORKS_API_KEY="your-key"

SCHEDULER_MODEL_ID="Qwen/Qwen3-8B" \
  ONLINE_ENDPOINTS_CONFIG="./online_endpoints_qwen3_8b.yaml" \
  sscheduler start --port 8010
```

## How It Works

1. At startup, the scheduler reads the YAML, creates N virtual instances per provider, each with a `WorkerQueueThread` that injects `Authorization: Bearer` headers
2. The scheduling strategy routes across all virtual instances; the Predictor learns per-provider latency
3. Instances are grouped by `endpoint_group` (visible in `/v1/instance/list`)

See [Cluster Deployment: Online API Endpoints](../../docs/cluster_deployment.md#online-api-endpoints-cloud-providers) for full reference. See `examples/8.virtual_concurrency/` for concurrency stress testing.
