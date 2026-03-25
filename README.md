# SwarmPilot

Distributed task scheduling with ML-based runtime prediction and automatic instance management.

## What Is SwarmPilot?

SwarmPilot orchestrates compute instances to run tasks efficiently. It predicts how long tasks will take, routes them to the best instance, and scales the cluster up or down via [PyLet](https://github.com/your-org/pylet).

| Service | Default Port | Role |
|---------|-------------|------|
| **Scheduler** | 8000 | Accepts tasks, picks an instance, tracks results. Embeds the **Predictor** (MLP-based runtime prediction) as a library — no separate service needed. |
| **Planner** | 8002 | Orchestrates multi-model deployments via PyLet. Manages scheduler discovery and instance lifecycle. |

> **Note:** The Predictor can also run as a standalone HTTP service (`spredictor start`) for external use, but the Scheduler uses it as an embedded library by default.

## Quick Start

Each Scheduler process serves **one model at a time**. The model is assigned dynamically by the Planner when the first deployment request arrives — you do not need to set `SCHEDULER_MODEL_ID` at startup. When all instances of a model are removed, the Scheduler becomes idle and the Planner can reassign it to a different model.

```bash
# Install
git clone <repo-url> swarmpilot-refresh
cd swarmpilot-refresh
uv sync

# Node 1: Start Planner (orchestration layer)
uv run splanner start --port 8002

# Node 2: Start Scheduler (auto-registers with Planner, model assigned on first deploy)
PLANNER_REGISTRATION_URL="http://localhost:8002" \
  uv run sscheduler start --port 8000
```

Deploy models and interact via the SDK:

```python
import asyncio
from swarmpilot.sdk import SwarmPilotClient

async def main():
    async with SwarmPilotClient("http://localhost:8002") as sp:
        # Deploy 2 replicas — Planner auto-assigns an idle Scheduler
        group = await sp.serve("Qwen/Qwen3-VL-8B-Instruct", gpu=1, replicas=2)
        print(f"Deployed: {group.name}")

        # Scale up
        scaled = await sp.scale("Qwen/Qwen3-VL-8B-Instruct", replicas=3)

        # Terminate — Scheduler becomes idle, can serve a different model next
        await sp.terminate(all=True)

        # Deploy a different model on the same Scheduler
        group = await sp.serve("meta-llama/Llama-3.1-8B", gpu=1, replicas=1)

asyncio.run(main())
```

See [examples/](examples/) for full runnable scripts with mock instances covering single-model, multi-model, and planner-managed deployments.

## Architecture

```
              ┌──────────────────┐
              │     Client       │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │     Planner      │ ◄── orchestration (optional)
              │     (8002)       │
              └────┬────────┬────┘
                   │        │
         ┌─────────▼──┐  ┌──▼─────────┐
         │ Scheduler  │  │ Scheduler  │ ◄── one per model
         │ ┌────────┐ │  │ ┌────────┐ │
         │ │Predictor│ │  │ │Predictor│ │ ◄── embedded (library mode)
         │ └────────┘ │  │ └────────┘ │
         └─────┬──────┘  └──────┬─────┘
               │                │
          ┌────▼────┐      ┌───▼─────┐
          │Instances│      │Instances│
          └─────────┘      └─────────┘
                                ▲ deploys/scales
                          PyLet Cluster
```

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/QUICK_START.md) | Local cluster walkthrough |
| [Architecture](docs/ARCHITECTURE.md) | System design and data flows |
| [API Reference](docs/API_REFERENCE.md) | All endpoints for all services |
| [Configuration](docs/CONFIGURATION.md) | Environment variables and CLI flags |
| [Deployment](docs/DEPLOYMENT.md) | Production deployment with PyLet |
| [Development](docs/DEVELOPMENT.md) | Dev setup, testing, contributing |
| [LLM Docs](docs/llm/) | Single-file references for AI assistants |

## Project Structure

```
swarmpilot-refresh/
├── swarmpilot/              # Python package
│   ├── scheduler/           # Task scheduling + embedded predictor
│   ├── predictor/           # Runtime prediction (library + HTTP)
│   ├── planner/             # Deployment optimization (PyLet)
│   ├── sdk/                 # Async Python SDK (SwarmPilotClient)
│   ├── graph/               # Client library
│   └── scripts/             # Deployment utilities
├── examples/                         # Runnable cluster examples
│   ├── single_model/                 # Single model, scheduler-only
│   ├── multi_model_direct/           # Multi-model, no planner
│   ├── multi_model_planner/          # Multi-model with planner + SDK
│   ├── predictor/                    # ML prediction (library + HTTP API)
│   └── predictor_training_playground/# Runtime collection + MLP training
├── tests/                   # Test suites
├── docs/                    # Documentation
└── pyproject.toml           # Package configuration
```

## Installation

```bash
# Development (recommended)
git clone <repo-url> swarmpilot-refresh
cd swarmpilot-refresh
uv sync                        # editable install
uv sync --extra pylet          # include PyLet

# Using pip
pip install swarmpilot

# With PyLet support
pip install swarmpilot[pylet]
```
## Pre-Trained Model

Predictor models for Qwen3-8B-VL

> https://www.dropbox.com/scl/fo/wvwm5cklrfvjqtofcsgfk/ANRYs9NnKR-sFnvpPfFV_vM?rlkey=gryoqk9sa6c65lky3csfhvsvp&st=2biytvst&dl=0

## License

TBD
