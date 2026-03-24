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

Each Scheduler process serves exactly **one model**. For multi-model deployments, run one Scheduler per model and use the Planner for coordination.

```bash
# Install
git clone <repo-url> swarmpilot-refresh
cd swarmpilot-refresh
uv sync

# Terminal 1: Start Planner (orchestration layer)
uv run splanner start --port 8002

# Terminal 2: Start Scheduler for model A (auto-registers with Planner)
SCHEDULER_MODEL_ID="Qwen/Qwen3-8B-VL" \
  PLANNER_REGISTRATION_URL="http://localhost:8002" \
  uv run sscheduler start --port 8010

# Terminal 3: Start Scheduler for model B (auto-registers with Planner)
SCHEDULER_MODEL_ID="meta-llama/Llama-3.1-8B" \
  PLANNER_REGISTRATION_URL="http://localhost:8002" \
  uv run sscheduler start --port 8020
```

Deploy models and interact via the SDK:

```python
import asyncio
from swarmpilot.sdk import SwarmPilotClient

async def main():
    async with SwarmPilotClient("http://localhost:8002") as sp:
        # Check registered schedulers
        schedulers = await sp.schedulers()
        print(schedulers)

        # Deploy 2 replicas of Qwen
        group = await sp.serve("Qwen/Qwen3-8B-VL", gpu=1, replicas=2)
        print(f"Deployed: {group.name}")

        # Scale up
        scaled = await sp.scale("Qwen/Qwen3-8B-VL", replicas=3)

        # Clean up
        await sp.terminate(all=True)

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
├── examples/                # Runnable cluster examples
│   ├── single_model/        # Single model, scheduler-only
│   ├── multi_model_direct/  # Multi-model, no planner
│   ├── multi_model_planner/ # Multi-model with planner + SDK
│   └── predictor/           # ML prediction (library + HTTP API)
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

## License

TBD
