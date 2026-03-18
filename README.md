# SwarmPilot

Distributed task scheduling with ML-based runtime prediction and automatic instance management.

## What Is SwarmPilot?

SwarmPilot orchestrates compute instances to run tasks efficiently. It predicts how long tasks will take, routes them to the best instance, and scales the cluster up or down via [PyLet](https://github.com/your-org/pylet).

| Service | Default Port | Role |
|---------|-------------|------|
| **Scheduler** | 8000 | Accepts tasks, picks an instance, tracks results |
| **Predictor** | 8001 | MLP-based runtime prediction (expect/error and quantile) |
| **Planner** | 8002 | Optimization-driven deployment via PyLet |

## Quick Start

```bash
# Install
pip install swarmpilot          # or: uv add swarmpilot

# Start services (3 terminals)
spredictor start --port 8001
sscheduler start --port 8000
splanner start --port 8002      # optional, needed for PyLet
```

See [docs/QUICK_START.md](docs/QUICK_START.md) for a full walkthrough with a local test cluster.

## Architecture

```
              ┌─────────────┐
              │   Client    │
              └──────┬──────┘
                     │ POST /v1/task/submit
                     ▼
              ┌─────────────┐         ┌─────────────┐
              │  Scheduler  │────────▶│  Predictor   │
              │   (8000)    │◀────────│   (8001)     │
              └──────┬──────┘         └──────────────┘
                     │ dispatches
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Instance │ │Instance │ │Instance │
   │    A    │ │    B    │ │    C    │
   └─────────┘ └─────────┘ └─────────┘
                     ▲ deploys/scales
              ┌──────┴──────┐
              │   Planner   │──── PyLet Cluster
              │   (8002)    │
              └─────────────┘
```

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/QUICK_START.md) | Local cluster in 5 minutes |
| [Architecture](docs/ARCHITECTURE.md) | System design and data flows |
| [API Reference](docs/API_REFERENCE.md) | All endpoints for all services |
| [Configuration](docs/CONFIGURATION.md) | Environment variables and CLI flags |
| [Deployment](docs/DEPLOYMENT.md) | Production deployment with PyLet |
| [Development](docs/DEVELOPMENT.md) | Dev setup, testing, contributing |
| [LLM Docs](docs/llm/) | Single-file references for AI assistants |

## Project Structure

```
swarmpilot-refresh/
├── swarmpilot/             # Python package
│   ├── scheduler/          # Task scheduling service
│   ├── predictor/          # Runtime prediction service
│   ├── planner/            # Deployment optimization (PyLet)
│   ├── graph/              # Client library
│   └── scripts/            # Deployment utilities
├── examples/               # Example cluster configurations
│   ├── single_model/       # Single model, direct registration
│   ├── multi_model_direct/ # Multi-model, direct registration
│   └── multi_model_planner/# Multi-model with Planner optimizer
├── tests/                  # Test suites
├── scripts/                # Startup scripts
├── docs/                   # Documentation
└── pyproject.toml          # Package configuration
```

## Installation

```bash
# Using pip
pip install swarmpilot

# Using uv (recommended)
uv add swarmpilot

# With PyLet support
pip install swarmpilot[pylet]

# Development
git clone <repo-url> swarmpilot-refresh
cd swarmpilot-refresh
uv sync                        # editable install
uv sync --extra pylet          # include PyLet
```

## License

TBD
