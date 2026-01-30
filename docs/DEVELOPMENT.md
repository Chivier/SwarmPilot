# Development

Developer guide for contributing to SwarmPilot.

## Setup

```bash
git clone <repo-url> swarmpilot-refresh
cd swarmpilot-refresh
uv sync                        # install all dependencies
uv sync --extra pylet          # include PyLet for planner work
```

This installs the package in editable mode with three CLI entry points:
- `sscheduler` -> `swarmpilot.scheduler.cli:app`
- `spredictor` -> `swarmpilot.predictor.cli:main`
- `splanner` -> `swarmpilot.planner.cli:app`

---

## Running Tests

```bash
# All tests
uv run pytest

# By service
uv run pytest scheduler/tests/
uv run pytest predictor/tests/
uv run pytest planner/tests/

# Verbose with output
uv run pytest -v -s
```

---

## Code Style

SwarmPilot follows [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### Formatting and Linting

```bash
uv run black .             # auto-format
uv run ruff check --fix .  # lint + auto-fix
uv run mypy src/           # type checking
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Functions/variables | `snake_case` | `submit_task` |
| Classes | `PascalCase` | `TaskRegistry` |
| Constants | `UPPER_SNAKE` | `DEFAULT_PORT` |
| Type hints | Required on function signatures | `def submit(task_id: str) -> bool:` |

### Line Length

80 characters maximum.

---

## Library Usage

SwarmPilot components can be imported directly:

```python
# Scheduler config
from swarmpilot.scheduler.config import config

# Predictor models
from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor

# Planner optimizer
from swarmpilot.planner.core.swarm_optimizer import SimulatedAnnealingOptimizer
from swarmpilot.planner.core.swarm_optimizer import IntegerProgrammingOptimizer
```

---

## Adding a Scheduling Strategy

1. Create a new file in `swarmpilot/scheduler/algorithms/`:

```python
# swarmpilot/scheduler/algorithms/my_strategy.py
from swarmpilot.scheduler.algorithms.base import SchedulingStrategy

class MyStrategy(SchedulingStrategy):
    def select_instance(self, task, instances, predictions):
        # Your selection logic here
        ...
```

2. Register it in `swarmpilot/scheduler/algorithms/factory.py`:

```python
from swarmpilot.scheduler.algorithms.my_strategy import MyStrategy

# In get_strategy():
elif strategy_name == "my_strategy":
    return MyStrategy(predictor_client, instance_registry)
```

3. Use it:

```bash
# Via environment variable
SCHEDULING_STRATEGY=my_strategy sscheduler start

# Via API at runtime
curl -X POST http://localhost:8000/v1/strategy/set \
  -H "Content-Type: application/json" \
  -d '{"strategy_name": "my_strategy"}'
```

### Existing Strategies

| Key | Class | File |
|-----|-------|------|
| `adaptive_bootstrap` | `AdaptiveBootstrapStrategy` | `adaptive_bootstrap.py` |
| `min_time` | `MinimumExpectedTimeStrategy` | `min_expected_time.py` |
| `probabilistic` | `ProbabilisticSchedulingStrategy` | `probabilistic.py` |
| `round_robin` | `RoundRobinStrategy` | `round_robin.py` |
| `random` | `RandomStrategy` | `random.py` |
| `po2` | `PowerOfTwoStrategy` | `power_of_two.py` |
| `severless` | `MinimumExpectedTimeServerlessStrategy` | `serverless.py` |

---

## Project Structure

```
swarmpilot-refresh/
├── swarmpilot/                # Python package (pip install swarmpilot)
│   ├── scheduler/             # Task scheduling service
│   │   ├── api.py             # FastAPI endpoints
│   │   ├── cli.py             # sscheduler CLI
│   │   ├── config.py          # Configuration
│   │   ├── models.py          # Pydantic models
│   │   ├── algorithms/        # 7 scheduling strategies
│   │   ├── registry/          # Instance + Task registries
│   │   ├── services/          # WorkerQueue, WebSocket, etc.
│   │   ├── clients/           # Predictor + Training clients
│   │   └── utils/
│   ├── predictor/             # Runtime prediction service
│   │   ├── api/               # FastAPI app + routes
│   │   ├── cli.py             # spredictor CLI
│   │   ├── config.py          # Pydantic settings
│   │   ├── predictor/         # MLP models (ExpectError, Quantile)
│   │   ├── preprocessor/      # Feature pipelines
│   │   └── storage/           # Model persistence
│   ├── planner/               # Deployment optimization
│   │   ├── api.py             # FastAPI endpoints
│   │   ├── pylet_api.py       # PyLet router
│   │   ├── cli.py             # splanner CLI
│   │   ├── config.py          # Configuration
│   │   ├── core/              # SA + IP optimizers
│   │   ├── pylet/             # PyLet integration
│   │   └── scheduler_registry.py
│   ├── graph/                 # Client library
│   └── scripts/               # Deployment utilities
├── examples/                  # Example cluster configs
│   ├── mock_llm_cluster/
│   ├── llm_cluster/
│   ├── multi_scheduler/
│   └── pylet_benchmark/
├── scripts/                   # Startup scripts
├── tests/                     # Test suites
├── docs/                      # Documentation
└── pyproject.toml             # Package config (hatchling)
```

---

## Example Clusters

| Directory | Purpose |
|-----------|---------|
| `examples/mock_llm_cluster/` | Local test with mock predictor + sleep models |
| `examples/llm_cluster/` | Real LLM cluster with vLLM/SGLang |
| `examples/multi_scheduler/` | Multi-scheduler setup with planner registration |
| `examples/pylet_benchmark/` | PyLet benchmarking tools |
