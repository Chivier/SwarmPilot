# Online API Multi-Provider Scheduling

Three models, each load-balanced across three cloud inference providers.

## Overview

This example demonstrates **multi-cloud inference redundancy**: the same
open-source model is hosted by multiple inference providers (Together AI,
Fireworks AI, Lepton AI), and SwarmPilot load-balances across them. The
Predictor learns each provider's latency distribution independently,
allowing the adaptive scheduler to prefer faster providers over time.

- **Scheduler A** (:8010) — `Qwen/Qwen3-8B` (3 providers)
- **Scheduler B** (:8020) — `Qwen/Qwen3-Next-80B-A3B` (3 providers)
- **Scheduler C** (:8030) — `google/gemma-4-41b-it` (3 providers)

No Planner — each scheduler independently manages its 3 provider endpoints.
Online endpoints are auto-registered at scheduler startup via YAML config
(no manual `POST /v1/instance/register` needed).

## Architecture

```
                  ┌──────────────────────────────┐
                  │           Client              │
                  │      (api_example.py)         │
                  └──┬──────────┬──────────┬──────┘
                     │          │          │
                     ▼          ▼          ▼
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │ Scheduler A  │ │ Scheduler B  │ │ Scheduler C  │
          │   :8010      │ │   :8020      │ │   :8030      │
          │  Qwen3-8B    │ │ Qwen3-Next   │ │  Gemma4-41B  │
          └─┬─────┬────┬─┘ └─┬─────┬────┬─┘ └─┬─────┬────┬─┘
            │     │    │     │     │    │     │     │    │
            ▼     ▼    ▼     ▼     ▼    ▼     ▼     ▼    ▼
          ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
          │TGR│ │FWK│ │LPT│ │TGR│ │FWK│ │LPT│ │TGR│ │FWK│ │LPT│
          │:91│ │:91│ │:91│ │:91│ │:91│ │:91│ │:91│ │:91│ │:91│
          │00 │ │01 │ │02 │ │10 │ │11 │ │12 │ │20 │ │21 │ │22 │
          └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
           Together  Fireworks  Lepton   (x 3 models)
```

## Quick Start

```bash
# 1. Start mock API servers + 3 schedulers (auto-registers all 9 endpoints)
./examples/7.online_api/start_cluster.sh

# 2. Send requests (9 tasks: 3 models x 3 prompts)
python examples/7.online_api/api_example.py

# 3. Tear down
./examples/7.online_api/stop_cluster.sh
```

No `deploy_model.sh` step — online endpoints are auto-registered at
scheduler startup from the YAML config.

## Scripts

| Script | Purpose |
|--------|---------|
| `start_cluster.sh` | Start 9 mock servers + 3 schedulers with auto-registration |
| `stop_cluster.sh` | Stop all services |
| `api_example.py` | Send chat requests to all 3 models via their schedulers |
| `mock_online_api.py` | 9 mock OpenAI-compatible API servers with per-provider latency |

## Configuration

### Online Endpoint YAML Files

Each scheduler loads a YAML config listing its providers:

| Config File | Model | Providers |
|-------------|-------|-----------|
| `online_endpoints_qwen3_8b.yaml` | `Qwen/Qwen3-8B` | Together / Fireworks / Lepton |
| `online_endpoints_qwen3_next.yaml` | `Qwen/Qwen3-Next-80B-A3B` | Together / Fireworks / Lepton |
| `online_endpoints_gemma4.yaml` | `google/gemma-4-41b-it` | Together / Fireworks / Lepton |

YAML format:

```yaml
endpoints:
  - name: "together-qwen3-8b"             # -> instance_id: "online-together-qwen3-8b"
    base_url: "https://api.together.xyz"
    api_key_env: "TOGETHER_API_KEY"        # reads key from this env var
    model_id: "Qwen/Qwen3-8B"
    concurrency_limit: 15
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOGETHER_API_KEY` | `mock-together-key` | Together AI API key |
| `FIREWORKS_API_KEY` | `mock-fireworks-key` | Fireworks AI API key |
| `LEPTON_API_KEY` | `mock-lepton-key` | Lepton AI API key |
| `SCHEDULER_QWEN3_8B_PORT` | `8010` | Scheduler A listen port |
| `SCHEDULER_QWEN3_NEXT_PORT` | `8020` | Scheduler B listen port |
| `SCHEDULER_GEMMA4_PORT` | `8030` | Scheduler C listen port |
| `ONLINE_ENDPOINTS_CONFIG` | (per scheduler) | Path to YAML config file |

### Port Map

```
Schedulers:          8010 (Qwen3-8B)   8020 (Qwen3-Next)  8030 (Gemma4)
Mock API servers:    9100-9102          9110-9112           9120-9122
Logs/PIDs:           /tmp/7.online_api/
```

## Using Real Providers

To switch from mock servers to real inference providers:

1. **Set real API keys:**
   ```bash
   export TOGETHER_API_KEY="your-together-api-key"
   export FIREWORKS_API_KEY="your-fireworks-api-key"
   export LEPTON_API_KEY="your-lepton-api-key"
   ```

2. **Edit YAML files** — replace `base_url` with real provider URLs:
   ```yaml
   # Together AI
   base_url: "https://api.together.xyz"

   # Fireworks AI
   base_url: "https://api.fireworks.ai/inference"

   # Lepton AI
   base_url: "https://lepton.ai/api"
   ```

3. **Start schedulers only** (skip mock servers):
   ```bash
   # Comment out the mock server section in start_cluster.sh,
   # or start schedulers manually:
   SCHEDULER_MODEL_ID="Qwen/Qwen3-8B" \
       ONLINE_ENDPOINTS_CONFIG="./examples/7.online_api/online_endpoints_qwen3_8b.yaml" \
       uv run sscheduler start --port 8010
   ```

## How It Works

1. **Startup:** Each scheduler reads its `online_endpoints_*.yaml`, resolves
   API keys from env vars, and registers 3 `Instance` objects — one per
   provider. Each instance gets a unique `PlatformInfo` (SHA-256 hash of
   base_url + api_key), so the Predictor trains separate models per provider.

2. **Request routing:** Client sends a chat request to a Scheduler. The
   scheduling strategy (default: `adaptive_bootstrap`) picks one of the 3
   provider instances. The request is forwarded via `WorkerQueueThread`,
   which injects the `Authorization: Bearer <key>` header automatically.

3. **Learning:** After each request completes, the actual runtime is recorded
   as a training sample. Over time, the QuantilePredictor learns that (e.g.)
   Together AI responds in 80-200ms while Lepton AI takes 150-450ms, and
   the scheduler shifts traffic toward faster providers.

## Troubleshooting

**"Port already in use"** — Run `./examples/7.online_api/stop_cluster.sh`
first, or check for stale processes: `lsof -i:8010`

**"Skipping online endpoint: KEY not set"** — The API key env var is empty.
For mock mode, `start_cluster.sh` sets dummy keys automatically.

**Scheduler starts but no instances registered** — Check that the
`ONLINE_ENDPOINTS_CONFIG` path is correct (absolute or relative to cwd)
and that the YAML file is valid: `python -c "import yaml; print(yaml.safe_load(open('...')))"`.
