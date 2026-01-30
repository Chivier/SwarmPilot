# SwarmPilot

A distributed task scheduling and execution system with dynamic load balancing, intelligent task allocation, and runtime prediction.

## Architecture

SwarmPilot consists of three core services, shipped as a single Python package:

- **Scheduler**: Task orchestration and instance management
- **Predictor**: MLP-based runtime prediction service
- **Planner**: Deployment optimization using linear programming

Task execution nodes are managed via **PyLet** (see [Quick Start](docs/QUICK_START.md)).

## Quick Start

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Using pip
pip install swarmpilot

# Using uv (recommended)
uv add swarmpilot

# With PyLet support (for production deployments)
pip install swarmpilot[pylet]
```

### Development Installation

```bash
git clone <repo-url> swarmpilot-refresh
cd swarmpilot-refresh
uv sync              # install in editable mode
uv sync --extra pylet  # include PyLet for planner
```

### Usage

Installing `swarmpilot` provides three CLI tools:

```bash
# Start scheduler on default port
sscheduler start

# Start predictor service
spredictor start

# Start planner service
splanner start
```

For help with any command:

```bash
sscheduler --help
spredictor --help
splanner --help
```

### Library Usage

```python
from swarmpilot.scheduler.config import config
from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.planner.core.swarm_optimizer import SimulatedAnnealingOptimizer
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific service
uv run pytest scheduler/tests/
uv run pytest predictor/tests/
uv run pytest planner/tests/
```

## Project Structure

```
swarmpilot-refresh/
├── swarmpilot/             # Distributable Python package
│   ├── scheduler/          # Scheduling service
│   ├── predictor/          # Prediction service
│   ├── planner/            # Planning service (with PyLet integration)
│   ├── graph/              # Client library
│   └── scripts/            # Deployment utilities
├── scheduler/tests/        # Scheduler tests
├── predictor/tests/        # Predictor tests
├── planner/tests/          # Planner tests
├── examples/               # Example cluster configurations
└── pyproject.toml          # Package configuration
```

## Documentation

See [docs/](docs/) for detailed guides:

- [Quick Start](docs/QUICK_START.md) — get a local cluster running
- [Scheduler Architecture](docs/scheduler_architecture_report.md)
- [PyLet Migration](docs/pylet_migration.md)

## Experiments

### Basic Experiments: Verify Installation

Exp. 01, 02, 03

### Scheduler Experiments

Exp. 09 (Universial Entrance for 04~07)

Cluster Config: static, A = m, B = n, m << n
Example Config: m = 8, n = 120
```
Exp. 04:   -> B
         A -> B
           -> B


Exp. 05: A -> B -> B -> B


Exp. 06:   -> B ->
         A -> B -> A
           -> B ->


Exp. 07:   -> B1 -> B2 ->
         A -> B1 -> B2 -> A
           -> B1 -> B2 ->
```


## License

TBD

## Contributing

TBD
