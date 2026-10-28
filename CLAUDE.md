# CLAUDE.md

## Project Overview

SwarmPilot is a distributed task scheduling and execution system with ML-based runtime prediction. It consists of three microservices:

| Service | Port | CLI | Role |
|---------|------|-----|------|
| **Scheduler** | 8000 | `sscheduler` | Task routing, instance management, queue dispatch |
| **Predictor** | 8001 | `spredictor` | MLP-based runtime prediction (ExpectError & Quantile) |
| **Planner** | 8002 | `splanner` | Optimization-driven deployment via PyLet |

## Quick Reference

```bash
# Install dependencies
uv sync                      # production deps
uv sync --dev                # with dev tools

# Run tests
uv run pytest                         # all tests
uv run pytest tests/scheduler/ -v     # scheduler only
uv run pytest tests/predictor/ -v     # predictor only
uv run pytest tests/planner/ -v       # planner only
uv run pytest tests/sdk/ -v           # SDK only
uv run pytest -k "test_name"          # single test
cargo test --release --manifest-path rust/Cargo.toml   # simulator + core (Rust)
uv sync --dev --extra core && SCHEDULER_CORE=rust uv run pytest tests/core/  # Rust core parity

# Lint & format
uv run ruff check --fix .             # lint + auto-fix
uv run black .                        # auto-format

# Start services
spredictor start --port 8001
sscheduler start --port 8000
splanner start --port 8002

# Online API mode (cloud providers, no local GPU)
SCHEDULER_MODEL_ID="Qwen/Qwen3-8B" \
  ONLINE_ENDPOINTS_CONFIG="./online_endpoints.yaml" \
  sscheduler start --port 8000

# SDK usage (async)
from swarmpilot.sdk import SwarmPilotClient
async with SwarmPilotClient("http://localhost:8002") as sp:
    group = await sp.serve("Qwen/Qwen2.5-7B", gpu=1)

# CLI deployment commands
splanner serve Qwen/Qwen2.5-7B --gpu 1 --replicas 2
splanner ps                              # list instances
splanner scale Qwen/Qwen2.5-7B --replicas 3
splanner terminate --all
splanner register my-model --gpu 2 --replicas 3
splanner deploy
splanner schedulers
splanner run "python train.py" --name my-job
```

## Project Structure

```
swarmpilot-refresh/
├── swarmpilot/              # Main Python package
│   ├── scheduler/           # Scheduler service
│   │   ├── api.py           # FastAPI endpoints (/v1/ prefix)
│   │   ├── cli.py           # sscheduler CLI (typer)
│   │   ├── config.py        # Dataclass-based config (env vars)
│   │   ├── algorithms/      # 7 scheduling strategies
│   │   ├── registry/        # InstanceRegistry, TaskRegistry
│   │   ├── services/        # WorkerQueue, WebSocket, PlannerRegistrar
│   │   ├── clients/         # PredictorClient, TrainingClient
│   │   ├── models/          # Pydantic request/response models
│   │   ├── proxy/           # Transparent proxy router
│   │   └── routing/         # Route definitions
│   ├── predictor/           # Predictor service
│   │   ├── api/             # FastAPI app + routes (no prefix)
│   │   ├── cli.py           # spredictor CLI
│   │   ├── config.py        # PydanticSettings (PREDICTOR_* env prefix)
│   │   ├── predictor/       # MLP models (ExpectError, Quantile)
│   │   ├── preprocessor/    # Feature pipelines
│   │   ├── storage/         # On-disk model persistence (.json)
│   │   └── models.py        # Pydantic models
│   ├── planner/             # Planner service
│   │   ├── api.py           # FastAPI endpoints (/v1/ prefix)
│   │   ├── pylet_api.py     # PyLet router (mounted at /v1/pylet)
│   │   ├── cli.py           # splanner CLI (typer) — serve, run, scale, etc.
│   │   ├── config.py        # Plain class config (env vars)
│   │   ├── core/            # SwarmOptimizer (SA + IP)
│   │   ├── pylet/           # PyLet SDK integration
│   │   ├── routes/          # Modular API routes
│   │   │   └── sdk_api.py   # SDK deployment endpoints (serve/run/scale/terminate)
│   │   └── scaling/         # Scaling logic
│   ├── scheduler/           # Scheduler service (continued)
│   │   └── routes/          # Modular API routes
│   │       └── predictor.py # Predictor management (train/predict/status)
│   ├── api/                 # API styles + translation (OpenAI Chat/Responses, Anthropic Messages)
│   ├── sdk/                 # Python SDK (async httpx client)
│   │   ├── __init__.py      # Public exports: SwarmPilotClient, models
│   │   ├── client.py        # SwarmPilotClient (async context manager)
│   │   └── models.py        # Dataclasses: Instance, InstanceGroup, Process, etc.
│   ├── sim/                 # Cluster simulator: Python face + data (catalogs, specs)
│   ├── errors.py            # Centralized error hierarchy (SwarmPilotError, etc.)
│   ├── graph/               # Client library
│   └── scripts/             # Deployment utilities
├── tests/                   # Test suites (mirrors swarmpilot/ layout)
│   ├── scheduler/           # Scheduler tests + conftest.py with fixtures
│   ├── predictor/           # Predictor tests
│   ├── planner/             # Planner tests (unit/ + integration/)
│   ├── sdk/                 # SDK client + model unit tests
│   ├── integration/         # End-to-end SDK integration tests
│   └── conftest.py          # Shared: --run-integration flag, PyLet fixtures
├── rust/                    # Cargo workspace: core + core-py (scheduling core, PyO3), gateway (swarmpilot-gateway), sim-core, sim-cluster, sim-cli (swarmpilot-sim)
├── proto/                   # gRPC protos (gateway control channel)
├── examples/                # Example cluster configurations
├── docs/                    # Documentation
├── pyproject.toml           # Package config (hatchling build, uv tooling)
└── uv.lock                  # Locked dependencies
```

## Code Conventions

### Style
- **Google Python Style Guide** enforced via Ruff
- **Line length:** 80 characters
- **Formatter:** Black (double quotes, space indent)
- **Import sorting:** isort via Ruff, `swarmpilot` is first-party
- **Docstrings:** Google convention (enforced by ruff D rules, except D100/D104/D105/D107)
- **Type hints:** Required on function signatures

### Naming
| Element | Convention | Example |
|---------|-----------|---------|
| Functions/variables | `snake_case` | `submit_task` |
| Classes | `PascalCase` | `TaskRegistry` |
| Constants | `UPPER_SNAKE` | `DEFAULT_PORT` |

### Imports
```python
# Standard library
# Third-party
# First-party (swarmpilot.*)
# Local-folder (relative)
```

## Testing Conventions

- **Framework:** pytest with pytest-asyncio (`asyncio_mode = "auto"`)
- **Import mode:** `importlib` (not prepend)
- **Markers:** `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- **Integration tests** require `--run-integration` flag and a PyLet cluster
- **Mocking:** `unittest.mock` (MagicMock, AsyncMock), `respx` for httpx
- **Test files:** `test_*.py`, classes `Test*`, functions `test_*`
- **Fixtures:** Service-specific conftest.py in each test subdirectory (e.g., `tests/scheduler/conftest.py` has instance/task/prediction fixtures)

## Architecture Notes

### Scheduler <-> Predictor
The Scheduler embeds the Predictor as a library by default (direct Python calls, no HTTP). Standalone Predictor uses WebSocket for batch predictions.

### Config Patterns
Each service uses a different config approach:
- **Scheduler:** `@dataclass` classes with `os.getenv()`, global `config = Config.load()`
- **Predictor:** `pydantic_settings.BaseSettings` with `PREDICTOR_*` env prefix
- **Planner:** Plain class with `os.getenv()` in `__init__`, global `config = PlannerConfig()`
  - Supports `PYLET_LOCAL_MODE=true` to start a local PyLet cluster as subprocesses (no external cluster needed)
  - Supports `PYLET_SIM_MODE=true` to start a real PyLet head with simulated GPU nodes and virtual engines (`swarmpilot-sim up`, see `docs/simulator.md`)
  - Mixed hardware: `PLANNER_HARDWARE_MAP` maps PyLet worker id/host patterns to hardware names; instances register with their worker's hardware, `serve --hardware 'H100*'` pins a model (`swarmpilot/planner/pylet/hardware.py`)
  - Auto-optimize loop (`AUTO_OPTIMIZE_ENABLED`, `swarmpilot/planner/services/auto_optimizer.py`) re-divides instances between models; `AUTO_OPTIMIZE_ELASTIC=true` also grows the pool into free GPUs and shrinks it when idle (`AUTO_OPTIMIZE_TARGET_CONCURRENCY` tasks in flight per instance); `AUTO_OPTIMIZE_HARDWARE_AWARE=true` weighs machines by hardware speed learned from execution times (`services/hardware_speed.py`) and names which instances move and where; it holds each model's peak demand for `AUTO_OPTIMIZE_DEMAND_HOLD` (default two intervals) before planning for less
  - Instance watcher (`PLANNER_INSTANCE_WATCH_INTERVAL`, default 2 s) removes instances PyLet reports FAILED and replaces them (`PLANNER_REPLACE_FAILED`)
  - Whole nodes (`PLANNER_WHOLE_NODES`, default true; `false` shares workers by GPU): one instance per PyLet worker, holding all its GPUs; a model's `gpu_count` is the minimum worker size (`instance_manager.py` `free_nodes`/`_place_nodes`)

### API Prefixes
- Scheduler endpoints: `/v1/...`
- Predictor endpoints: root (no prefix, e.g., `POST /predict`)
- Planner endpoints: `/v1/...`
- Planner SDK endpoints: `/v1/serve`, `/v1/run`, `/v1/scale`, `/v1/terminate`, `/v1/instances`, `/v1/schedulers`, `/v1/register`, `/v1/deploy`, `/v1/registered`
- Scheduler predictor endpoints: `/v1/predictor/train`, `/v1/predictor/predict`, `/v1/predictor/status/{model_id}`, `/v1/predictor/models`

### SDK & CLI
The `swarmpilot.sdk` package provides `SwarmPilotClient`, an async httpx-based client for Planner and Scheduler APIs. The `splanner` CLI exposes matching commands: `serve`, `run`, `register`, `deploy`, `ps`, `scale`, `terminate`, `schedulers`. Error hierarchy in `swarmpilot.errors`: `SwarmPilotError` > `DeployError`, `RegistrationError`, `SchedulerNotFound`, `ModelNotDeployed`, `OptimizationError`, `SwarmPilotTimeoutError`.

### One Scheduler Per Model (Dynamic Assignment)
Each Scheduler process serves one model at a time. The model can be assigned dynamically — `SCHEDULER_MODEL_ID` is optional at startup. When the Planner receives a `serve()` request for a model with no registered scheduler, it finds an idle scheduler (zero instances) and calls `POST /v1/model/reassign` to bind it to the new model. When all instances of a model are terminated, the scheduler becomes idle and can be reassigned.

| Deployment | Schedulers | Planner |
|------------|-----------|---------|
| 1 model | 1 Scheduler | Optional |
| N models | N Schedulers (one per model) | Recommended |
| Model switch | Same Scheduler (after teardown) | Required |

### API Styles (Translation Layer)
Clients may call the Scheduler proxy in any registered API style (`swarmpilot/api/styles/`: `openai_chat`, `anthropic_messages`, `openai_responses`; plugins via the `swarmpilot.api_styles` entry-point group). If the engine's adapter lists the client's style in `native_styles()` the body passes through untouched; otherwise it is decoded into `swarmpilot.api.canonical` and re-encoded in the engine's style, answers and SSE streams included. Answer-changing fields that cannot be translated give a 400 (set `SWARMPILOT_API_TRANSLATION=lenient` to drop them); dropped fields are listed in the `x-swarmpilot-dropped` header. Regenerate `swarmpilot/api/canonical.schema.json` with `python -m swarmpilot.api.canonical`.

### Unreachable Instances
A refused connection marks a Scheduler instance `UNHEALTHY` (no new tasks; its unsent tasks are re-routed at once). The health loop restores it when it answers (`SCHEDULER_HEALTH_CHECK_INTERVAL`) or removes it after `SCHEDULER_UNHEALTHY_REMOVE_AFTER` seconds. See "Engine Failures" in `docs/cluster_deployment.md`.

### Rust Core
`SCHEDULER_CORE=rust` runs queue sketches, strategy selection, feature extractors and predictor inference (MLP forward pass, within 1e-5 of torch) in Rust (`rust/crates/core`, module `swarmpilot_core`, `uv sync --extra core`). Results must match Python bit for bit: change both, regenerate `tests/golden/`, and keep `tests/core/` green. See `docs/rust_core.md`.

### Rust Gateway
`swarmpilot-gateway` (`rust/crates/gateway`) can serve the Scheduler's proxy port in Rust: start the Scheduler with `SCHEDULER_GATEWAY_SOCKET=<path>` and the gateway with `--control <path> --listen <addr>`. The Scheduler streams routing state and receives completions over gRPC on that Unix socket (`proto/swarmpilot/gateway/v1/control.proto`, Python side `swarmpilot/scheduler/gateway_control.py`; regenerate `swarmpilot/gateway/v1/*_pb2*` as in `swarmpilot/gateway/__init__.py`). It translates API styles itself with `swarmpilot_core::api` (a port of `swarmpilot/api`): change both and keep `tests/core/test_translation_parity.py` green. Every test in `tests/conformance/` and `tests/api/test_conformance_matrix.py` runs against both data planes when the binary is built. Gaps (Python-only API style plugins and feature adapters → 501 / no features) in `docs/rust_core.md`. Shadow mode: `SCHEDULER_GATEWAY_SHADOW_URL` + `swarmpilot-gateway --shadow` mirrors every Python routing decision (with the strategy's random draws, `swarmpilot/scheduler/shadow.py`) to the gateway and compares (`GET /v1/gateway/shadow`); `tests/perf/ab_cluster.py` is the 256-instance simulated-cluster A/B.

### Scheduling Strategies
7 strategies available, selectable at runtime via `POST /v1/strategy/set` or `SCHEDULING_STRATEGY` env var:
`adaptive_bootstrap` (default), `min_time`, `probabilistic`, `round_robin`, `random`, `po2`, `severless`

## Commit Message Convention

Uses conventional commits with optional scope and ticket reference:
```
type(scope): description [TICKET-NNN]
```
Examples from history:
- `fix(ci): remove hardcoded pylet local path and guard imports [PYLET-041]`
- `refactor: consolidate project to swarmpilot/ tests/ docs/ examples/ [PYLET-040]`
- `feat(predictor): add library API examples and get_model_info method`

## Key Dependencies

- **Web:** FastAPI + Uvicorn + Pydantic v2
- **ML:** PyTorch, NumPy, SciPy, Transformers
- **Optimization:** PuLP (linear/integer programming)
- **HTTP client:** httpx
- **CLI:** Typer
- **Logging:** Loguru
- **Build:** Hatchling (via uv)
- **Python:** >=3.11, developed on 3.13

## CI

GitHub Actions (`.github/workflows/ci.yml`):
- **test** job: Python 3.11 + 3.12 matrix, runs `uv run pytest tests/{scheduler,predictor,planner}/` separately
- **lint** job: Python 3.11, syntax check via `py_compile` on all .py files (excludes `.venv/`, `experiments/`)
