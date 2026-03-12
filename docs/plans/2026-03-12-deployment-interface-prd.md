# SwarmPilot Deployment Interface Redesign — PRD

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Provide a PyLet-inspired Python SDK (`import swarmpilot`) and CLI extension (`splanner`) that allows users to deploy models, manage custom workloads, and configure predictors through three clear paths — optimized deployment, manual deployment, and custom workload — without touching HTTP APIs directly.

**Architecture:** The SDK connects to Planner as the sole entry point, delegates deployment to Planner's existing PyLet integration, and direct-connects to Schedulers for predictor management. Planner gains new REST endpoints (`/v1/serve`, `/v1/run`, `/v1/register`, `/v1/deploy`, etc.) while old endpoints are preserved as legacy. Scheduler gains new `/v1/predictor/*` endpoints for preprocessor upload and MLP training management.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, httpx, Typer, PyLet SDK, loguru

**Design Document:** `docs/plans/2026-03-12-deployment-interface-design.md`

---

## Existing Codebase Reference

Understanding the existing code is essential for every task. Key files:

| Component | Path | Lines | Role |
|-----------|------|-------|------|
| Package init | `swarmpilot/__init__.py` | 3 | Only exports `__version__` — SDK entry point will live here |
| Planner API | `swarmpilot/planner/api.py` | 632 | FastAPI app, scheduler registry endpoints, PyLet router mount |
| Planner CLI | `swarmpilot/planner/cli.py` | 85 | Typer CLI: `start`, `version` — add `serve`, `run`, `register`, `deploy`, `ps`, `scale`, `terminate`, `info`, `predictor` |
| Planner models | `swarmpilot/planner/models/__init__.py` | 80 | Pydantic models re-exports — add new request/response models |
| Planner models (pylet) | `swarmpilot/planner/models/pylet.py` | ~200 | PyLet-specific Pydantic models |
| PyLet client | `swarmpilot/planner/pylet/client.py` | 518 | `deploy_model()`, `cancel_instance()`, `list_model_instances()` |
| Instance manager | `swarmpilot/planner/pylet/instance_manager.py` | 757 | `deploy_instances()`, `wait_instance_ready()`, `terminate_instance()` |
| Deployment service | `swarmpilot/planner/pylet/deployment_service.py` | 503 | `apply_deployment()`, `scale_model()`, `terminate_all()` |
| Scheduler registry | `swarmpilot/planner/scheduler_registry.py` | 156 | `register()`, `get_scheduler_url()`, `list_all()` |
| Scheduler API | `swarmpilot/scheduler/api.py` | 3062 | Full scheduler with proxy routing, strategy management |
| Scheduler CLI | `swarmpilot/scheduler/cli.py` | 213 | Typer CLI: `start`, `version` |
| Predictor routes | `swarmpilot/predictor/api/routes/` | 7 files | `/train`, `/predict`, `/health`, `/list`, cache, websocket |
| Preprocessor V2 | `swarmpilot/predictor/preprocessor/` | 9 files | BasePreprocessorV2, chain, registry, adapters |
| Predictor core | `swarmpilot/predictor/api/core.py` | ~400 | `PredictorLowLevel`, `PredictorCore` |
| Scheduler predictor client | `swarmpilot/scheduler/clients/predictor_library_client.py` | ~300 | Library-mode predictor calls from Scheduler |
| Scheduler training client | `swarmpilot/scheduler/clients/training_library_client.py` | ~250 | Buffer-based MLP training |
| Planner config | `swarmpilot/planner/config.py` | ~60 | `PlannerConfig` with env vars |
| Scheduler config | `swarmpilot/scheduler/config.py` | ~180 | `Config` dataclass, `PlannerRegistrationConfig` |
| Planner test fixtures | `tests/planner/conftest.py` | 90 | `sample_planner_input`, `client` (TestClient) |
| Scheduler test fixtures | `tests/scheduler/conftest.py` | 398 | Comprehensive: registries, predictor mocks, API client |

**No `swarmpilot/errors.py` exists yet** — custom exceptions are scattered inline. We create a centralized module.

---

## Task 1: Error Types Module

Create centralized error hierarchy used by SDK, Planner, and Scheduler.

**Files:**
- Create: `swarmpilot/errors.py`
- Create: `tests/test_errors.py`

**Step 1: Write failing tests**

```python
# tests/test_errors.py
"""Tests for swarmpilot.errors module."""
import pytest

from swarmpilot.errors import (
    SwarmPilotError,
    DeployError,
    RegistrationError,
    SchedulerNotFound,
    OptimizationError,
    ModelNotDeployed,
    TimeoutError as SPTimeoutError,
)


class TestErrorHierarchy:
    """All errors inherit from SwarmPilotError."""

    def test_base_error(self):
        err = SwarmPilotError("base")
        assert isinstance(err, Exception)
        assert str(err) == "base"

    def test_deploy_error_partial_success(self):
        succeeded = [{"name": "inst-1", "endpoint": "http://w1:30001"}]
        failed = [{"replica": 2, "error": "GPU exhausted"}]
        err = DeployError(
            "2 of 3 replicas failed",
            succeeded=succeeded,
            failed=failed,
        )
        assert err.succeeded == succeeded
        assert err.failed == failed
        assert isinstance(err, SwarmPilotError)

    def test_registration_error(self):
        err = RegistrationError(
            "Scheduler unreachable",
            scheduler="http://sched:8000",
            model="Qwen/Qwen2.5-7B",
        )
        assert err.scheduler == "http://sched:8000"
        assert err.model == "Qwen/Qwen2.5-7B"

    def test_scheduler_not_found(self):
        err = SchedulerNotFound(model="unknown-model")
        assert err.model == "unknown-model"
        assert "scheduler" in err.hint.lower()

    def test_optimization_error(self):
        err = OptimizationError(
            reason="Insufficient GPU: need 8, available 5"
        )
        assert err.reason == "Insufficient GPU: need 8, available 5"

    def test_model_not_deployed(self):
        err = ModelNotDeployed(model="Qwen/Qwen2.5-7B")
        assert err.model == "Qwen/Qwen2.5-7B"
        assert "deploy" in err.hint.lower()

    def test_timeout_error(self):
        err = SPTimeoutError(
            "Instance not ready",
            timeout=300,
            name="qwen-7b-abc12",
        )
        assert err.timeout == 300
        assert err.name == "qwen-7b-abc12"
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_errors.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'swarmpilot.errors'`

**Step 3: Implement errors module**

```python
# swarmpilot/errors.py
"""Centralized error types for SwarmPilot SDK, Planner, and Scheduler."""


class SwarmPilotError(Exception):
    """Base exception for all SwarmPilot errors."""


class DeployError(SwarmPilotError):
    """Deployment failed (partially or fully).

    Attributes:
        succeeded: Instances that deployed successfully.
        failed: Details of failed replicas.
    """

    def __init__(
        self,
        message: str,
        succeeded: list | None = None,
        failed: list | None = None,
    ):
        super().__init__(message)
        self.succeeded = succeeded or []
        self.failed = failed or []


class RegistrationError(SwarmPilotError):
    """Scheduler registration failed.

    Attributes:
        scheduler: Scheduler URL that was unreachable.
        model: Model ID that failed registration.
    """

    def __init__(
        self,
        message: str,
        scheduler: str = "",
        model: str = "",
    ):
        super().__init__(message)
        self.scheduler = scheduler
        self.model = model


class SchedulerNotFound(SwarmPilotError):
    """No Scheduler registered for the given model.

    Attributes:
        model: Model ID with no Scheduler mapping.
        hint: Actionable suggestion for the user.
    """

    def __init__(self, model: str):
        self.model = model
        self.hint = (
            f"No scheduler registered for '{model}'. "
            "Use scheduler='http://...' to specify manually, "
            "or scheduler=None to skip registration."
        )
        super().__init__(self.hint)


class OptimizationError(SwarmPilotError):
    """Planner optimizer found no feasible solution.

    Attributes:
        reason: Human-readable explanation.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class ModelNotDeployed(SwarmPilotError):
    """Predictor operation attempted on a model with no Scheduler mapping.

    Attributes:
        model: Model ID that has no Scheduler.
        hint: Actionable suggestion.
    """

    def __init__(self, model: str):
        self.model = model
        self.hint = (
            f"Model '{model}' has no scheduler mapping. "
            "Deploy the model first with swarmpilot.serve() "
            "or swarmpilot.deploy()."
        )
        super().__init__(self.hint)


class TimeoutError(SwarmPilotError):
    """Instance did not become ready within timeout.

    Attributes:
        timeout: Timeout in seconds.
        name: Instance name that timed out.
    """

    def __init__(
        self,
        message: str,
        timeout: int = 0,
        name: str = "",
    ):
        super().__init__(message)
        self.timeout = timeout
        self.name = name
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_errors.py -v
```

Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add swarmpilot/errors.py tests/test_errors.py
git commit -m "feat: add centralized error types module [SDK-001]"
```

---

## Task 2: SDK Data Models

Create SDK return types: `Instance`, `InstanceGroup`, `Process`, `DeploymentResult`, `ClusterState`, and predictor data models.

**Files:**
- Create: `swarmpilot/sdk/models.py`
- Create: `swarmpilot/sdk/__init__.py`
- Create: `tests/sdk/__init__.py`
- Create: `tests/sdk/test_models.py`

**Step 1: Write failing tests**

```python
# tests/sdk/test_models.py
"""Tests for SDK data models."""
import pytest

from swarmpilot.sdk.models import (
    Instance,
    InstanceGroup,
    Process,
    DeploymentResult,
    ClusterState,
    PreprocessorInfo,
    ModelStatus,
    TrainResult,
    PredictResult,
)


class TestInstance:
    def test_create_instance(self):
        inst = Instance(
            name="qwen-7b-abc12",
            model="Qwen/Qwen2.5-7B",
            command="vllm serve Qwen/Qwen2.5-7B --port 30001",
            endpoint="http://w1:30001",
            scheduler="http://sched:8000",
            status="running",
            gpu=1,
        )
        assert inst.name == "qwen-7b-abc12"
        assert inst.scheduler == "http://sched:8000"

    def test_instance_pending_has_no_endpoint(self):
        inst = Instance(
            name="qwen-7b-abc12",
            model="Qwen/Qwen2.5-7B",
            command="vllm serve Qwen/Qwen2.5-7B --port $PORT",
            endpoint=None,
            scheduler="http://sched:8000",
            status="pending",
            gpu=1,
        )
        assert inst.endpoint is None

    def test_instance_no_scheduler(self):
        inst = Instance(
            name="qwen-nosched",
            model="Qwen/Qwen2.5-7B",
            command="vllm serve Qwen/Qwen2.5-7B --port $PORT",
            endpoint="http://w1:30001",
            scheduler=None,
            status="running",
            gpu=1,
        )
        assert inst.scheduler is None

    def test_instance_custom_command_no_model(self):
        inst = Instance(
            name="custom-inference",
            model=None,
            command="python my_server.py --port $PORT",
            endpoint="http://w1:30001",
            scheduler="http://sched:8000",
            status="running",
            gpu=2,
        )
        assert inst.model is None


class TestInstanceGroup:
    def test_endpoints_property(self):
        instances = [
            Instance(
                name="q-1", model="Q", command="cmd",
                endpoint=f"http://w{i}:3000{i}",
                scheduler="http://s:8000", status="running", gpu=1,
            )
            for i in range(1, 4)
        ]
        group = InstanceGroup(
            name="Q", model="Q", command="cmd",
            instances=instances, scheduler="http://s:8000",
        )
        assert group.endpoints == [
            "http://w1:30001", "http://w2:30002", "http://w3:30003"
        ]

    def test_endpoints_filters_none(self):
        instances = [
            Instance(
                name="q-1", model="Q", command="cmd",
                endpoint=None,
                scheduler="http://s:8000", status="pending", gpu=1,
            ),
            Instance(
                name="q-2", model="Q", command="cmd",
                endpoint="http://w2:30002",
                scheduler="http://s:8000", status="running", gpu=1,
            ),
        ]
        group = InstanceGroup(
            name="Q", model="Q", command="cmd",
            instances=instances, scheduler="http://s:8000",
        )
        assert group.endpoints == ["http://w2:30002"]


class TestProcess:
    def test_process_scheduler_always_none(self):
        proc = Process(
            name="crawler",
            command="python crawler.py --port $PORT",
            endpoint="http://w2:34521",
            status="running",
            gpu=0,
        )
        assert proc.scheduler is None


class TestDeploymentResult:
    def test_getitem(self):
        group = InstanceGroup(
            name="Q", model="Qwen/Qwen2.5-7B", command="cmd",
            instances=[], scheduler="http://s:8000",
        )
        result = DeploymentResult(
            plan={"Qwen/Qwen2.5-7B": 3},
            groups={"Qwen/Qwen2.5-7B": group},
            status="ready",
        )
        assert result["Qwen/Qwen2.5-7B"] is group

    def test_getitem_missing_raises_keyerror(self):
        result = DeploymentResult(plan={}, groups={}, status="ready")
        with pytest.raises(KeyError):
            result["nonexistent"]


class TestPreprocessorInfo:
    def test_create(self):
        info = PreprocessorInfo(
            name="llm-output-predictor",
            feature="input_text",
            path="/storage/preprocessors/llm-output-predictor",
        )
        assert info.name == "llm-output-predictor"
        assert info.feature == "input_text"


class TestModelStatus:
    def test_create(self):
        status = ModelStatus(
            model="Qwen/Qwen2.5-7B",
            samples_collected=87,
            last_trained="2026-03-12T10:30:00",
            prediction_types=["expect_error", "quantile"],
            metrics={"mse": 0.03, "mae": 12.5},
            strategy="round_robin",
            preprocessors=[
                PreprocessorInfo(
                    name="llm-out", feature="input_text", path="/p"
                )
            ],
        )
        assert status.strategy == "round_robin"
        assert len(status.preprocessors) == 1


class TestTrainResult:
    def test_create(self):
        result = TrainResult(
            model="Qwen/Qwen2.5-7B",
            samples_trained=87,
            metrics={"mse": 0.02},
            strategy="probabilistic",
        )
        assert result.strategy == "probabilistic"


class TestPredictResult:
    def test_create_expect_error(self):
        result = PredictResult(
            model="Qwen/Qwen2.5-7B",
            expected_runtime_ms=1234.5,
            error_margin_ms=50.0,
            quantiles=None,
        )
        assert result.quantiles is None

    def test_create_quantile(self):
        result = PredictResult(
            model="Qwen/Qwen2.5-7B",
            expected_runtime_ms=None,
            error_margin_ms=None,
            quantiles={0.5: 1200, 0.9: 1500, 0.95: 1700},
        )
        assert result.expected_runtime_ms is None
        assert result.quantiles[0.9] == 1500
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/sdk/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'swarmpilot.sdk'`

**Step 3: Implement SDK models**

```python
# swarmpilot/sdk/__init__.py
"""SwarmPilot Python SDK."""

# swarmpilot/sdk/models.py
"""SDK data models for deployment results and predictor management."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Instance:
    """A single deployed instance."""

    name: str
    model: str | None
    command: str
    endpoint: str | None
    scheduler: str | None
    status: str
    gpu: int


@dataclass
class InstanceGroup:
    """A group of replicas for the same model/command."""

    name: str
    model: str | None
    command: str
    instances: list[Instance]
    scheduler: str | None

    @property
    def endpoints(self) -> list[str]:
        """Return endpoints of all instances that have one."""
        return [
            i.endpoint for i in self.instances if i.endpoint is not None
        ]


@dataclass
class Process:
    """A custom workload (no scheduler)."""

    name: str
    command: str
    endpoint: str | None
    status: str
    gpu: int
    scheduler: None = field(default=None, init=False)


@dataclass
class DeploymentResult:
    """Result of an optimized deploy() call."""

    plan: dict
    groups: dict[str, InstanceGroup]
    status: str

    def __getitem__(self, model: str) -> InstanceGroup:
        return self.groups[model]


@dataclass
class ClusterState:
    """Current state of all instances and processes."""

    instances: list[Instance]
    processes: list[Process]
    groups: list[InstanceGroup]


@dataclass
class PreprocessorInfo:
    """Metadata about a registered preprocessor."""

    name: str
    feature: str
    path: str


@dataclass
class ModelStatus:
    """Predictor training status for a model."""

    model: str
    samples_collected: int
    last_trained: str | None
    prediction_types: list[str]
    metrics: dict
    strategy: str
    preprocessors: list[PreprocessorInfo]


@dataclass
class TrainResult:
    """Result of a training or retrain operation."""

    model: str
    samples_trained: int
    metrics: dict
    strategy: str


@dataclass
class PredictResult:
    """Result of a manual prediction."""

    model: str
    expected_runtime_ms: float | None
    error_margin_ms: float | None
    quantiles: dict[float, float] | None
```

Also create `tests/sdk/__init__.py` (empty).

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/sdk/test_models.py -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add swarmpilot/sdk/ tests/sdk/
git commit -m "feat(sdk): add data models for Instance, InstanceGroup, Process, DeploymentResult, predictor types [SDK-002]"
```

---

## Task 3: Planner — New Pydantic Request/Response Models

Add Pydantic models for the new Planner endpoints: `/v1/serve`, `/v1/run`, `/v1/register`, `/v1/deploy`, `/v1/instances`, `/v1/scale`, `/v1/terminate`, `/v1/schedulers`.

**Files:**
- Create: `swarmpilot/planner/models/sdk_api.py`
- Modify: `swarmpilot/planner/models/__init__.py` — add re-exports
- Create: `tests/planner/test_sdk_api_models.py`

**Step 1: Write failing tests**

Test each request/response model validates correctly — model names are sanitized for auto-generated names, scheduler values accept `"auto"`, `null`, and explicit URLs, etc.

Key models to test:
- `ServeRequest` — `model_or_command: str`, `name: str | None`, `replicas: int = 1`, `gpu: int`, `scheduler: str | None = "auto"`, `memory: int | None = None`, `env: dict = {}`
- `ServeResponse` — `name`, `model`, `command`, `replicas: list[ReplicaStatus]`, `scheduler`, `status`
- `RunRequest` — `command: str`, `name: str`, `gpu: int = 0`, `memory: int | None = None`, `env: dict = {}`
- `RunResponse` — `name`, `command`, `endpoint`, `scheduler: None`, `status`
- `RegisterRequest` — `model: str`, `gpu: int`, `target_qps: float`
- `DeployResponse` — `plan: dict`, `groups: dict[str, ServeResponse]`, `status`
- `ScaleRequest` — `model: str`, `replicas: int`
- `TerminateRequest` — `name: str | None = None`, `model: str | None = None`, `all: bool = False`
- `InstanceDetailResponse` — full instance detail with `endpoint`, `scheduler`, `status`, `gpu`, `command`
- `SchedulerMapResponse` — `schedulers: dict[str, str]` (model_id → scheduler_url)

**Step 2:** Run tests, confirm FAIL.

**Step 3:** Implement in `swarmpilot/planner/models/sdk_api.py` using Pydantic v2 `BaseModel`. Add re-exports to `swarmpilot/planner/models/__init__.py`.

**Step 4:** Run tests, confirm PASS.

**Step 5: Commit**

```bash
git commit -m "feat(planner): add Pydantic models for new SDK API endpoints [SDK-003]"
```

---

## Task 4: Planner — New REST Endpoints (Deployment)

Implement the 8 new Planner endpoints for deployment management. These endpoints wire into the existing `PyLetDeploymentService`, `InstanceManager`, and `SchedulerRegistry`.

**Files:**
- Create: `swarmpilot/planner/routes/sdk_api.py` — new FastAPI APIRouter
- Modify: `swarmpilot/planner/api.py` — mount the new router at `/v1`
- Create: `tests/planner/test_sdk_api_endpoints.py`

**Endpoints to implement:**

### 4a. `POST /v1/serve`

Logic:
1. Parse `model_or_command` — if it looks like a model name (contains `/`), auto-generate `vllm serve {model} --port $PORT` command. Otherwise, treat as raw command.
2. `name` defaults to sanitized model name (replace `/` with `-`).
3. Call `InstanceManager.deploy_instances(model_id, count=replicas, gpu_count=gpu)`.
4. Call `InstanceManager.wait_instances_ready(pylet_ids, register=True/False)` based on `scheduler` param.
5. Scheduler resolution: `"auto"` → `SchedulerRegistry.get_scheduler_url(model)`, `None` → skip, explicit URL → use directly.
6. If `scheduler="auto"` and model not in registry → raise HTTP 404 with `SchedulerNotFound` hint.
7. Return `ServeResponse` with all replica statuses.

References:
- `swarmpilot/planner/pylet/instance_manager.py:deploy_instances()` (line ~180)
- `swarmpilot/planner/pylet/instance_manager.py:wait_instances_ready()` (line ~280)
- `swarmpilot/planner/scheduler_registry.py:get_scheduler_url()` (line ~60)

### 4b. `POST /v1/run`

Logic:
1. Call `PyLetClient.deploy_model()` with `command` directly, `replicas=1`.
2. No scheduler registration.
3. Return `RunResponse`.

References:
- `swarmpilot/planner/pylet/client.py:deploy_model()` (line ~170)

### 4c. `POST /v1/register`

Logic:
1. Store model requirements in a new in-memory `dict[str, RegisterRequest]` on the Planner (thread-safe).
2. If model already registered, update (overwrite).
3. No deployment triggered.

New state needed: `_registered_models: dict[str, RegisterRequest]` — similar to `SchedulerRegistry` pattern.

### 4d. `POST /v1/deploy`

Logic:
1. Collect all registered models from `_registered_models`.
2. Build capacity matrix from cluster state (`PyLetClient` or pylet cluster info).
3. Call `SwarmOptimizer.optimize()` (existing: `swarmpilot/planner/core/`).
4. Execute deployment plan via `DeploymentService.apply_deployment()`.
5. Return `DeployResponse` with plan and per-model instance groups.

References:
- `swarmpilot/planner/api.py:plan()` (line ~140) — existing optimizer invocation pattern
- `swarmpilot/planner/pylet/deployment_service.py:apply_deployment()` (line ~200)

### 4e. `GET /v1/registered`

Return the current `_registered_models` dict.

### 4f. `GET /v1/instances` and `GET /v1/instances/{name}`

Logic:
1. Query `InstanceManager.get_active_instances()` for all models.
2. Aggregate into `ClusterState` or single `InstanceDetailResponse`.

References:
- `swarmpilot/planner/pylet/instance_manager.py:get_active_instances()` (line ~400)

### 4g. `POST /v1/scale`

Logic:
1. Call `DeploymentService.scale_model(model_id, target_count)`.
2. Return updated instance group.

References:
- `swarmpilot/planner/pylet/deployment_service.py:scale_model()` (line ~300)

### 4h. `POST /v1/terminate`

Logic:
1. If `name` provided — find and terminate by name.
2. If `model` provided — terminate all instances for that model.
3. If `all=True` — `DeploymentService.terminate_all()`.

References:
- `swarmpilot/planner/pylet/deployment_service.py:terminate_all()` (line ~400)
- `swarmpilot/planner/pylet/instance_manager.py:terminate_instance()` (line ~330)

### 4i. `GET /v1/schedulers`

Logic:
1. Query `SchedulerRegistry.list_all()`.
2. Return `SchedulerMapResponse` — `{model_id: scheduler_url}`.

References:
- `swarmpilot/planner/scheduler_registry.py:list_all()` (line ~100)

**Testing approach:** Use `httpx.AsyncClient` with FastAPI `TestClient`. Mock `PyLetClient`, `InstanceManager`, `DeploymentService` to avoid needing a real PyLet cluster. Use `respx` for any outgoing HTTP calls.

**Step 5: Commit**

```bash
git commit -m "feat(planner): add /v1/serve, /v1/run, /v1/register, /v1/deploy and management endpoints [SDK-004]"
```

---

## Task 5: Scheduler — Predictor Management Endpoints

Add new REST endpoints on the Scheduler for preprocessor upload/register/remove/list and MLP training management. These endpoints wrap the existing `PredictorLowLevel` and preprocessor registry that are already embedded in the Scheduler.

**Files:**
- Create: `swarmpilot/scheduler/routes/predictor_api.py` — new FastAPI APIRouter
- Modify: `swarmpilot/scheduler/api.py` — mount the new router at `/v1/predictor`
- Create: `swarmpilot/scheduler/models/predictor_api.py` — Pydantic request/response models
- Create: `tests/scheduler/test_predictor_api.py`

**Endpoints to implement:**

### 5a. `POST /v1/predictor/preprocessor/upload`

Logic:
1. Accept multipart form: `archive` (file), `name` (str), `feature` (str).
2. Extract archive to `{config.preprocessor_storage}/{name}/`.
3. Validate the folder contains a class inheriting `BasePreprocessorV2`.
4. Register in preprocessor chain for the Scheduler's model.

References:
- `swarmpilot/predictor/preprocessor/registry_v2.py` — V2 preprocessor registry
- `swarmpilot/predictor/preprocessor/base_preprocessor_v2.py` — base interface

### 5b. `POST /v1/predictor/preprocessor/register`

Logic:
1. Accept JSON: `name`, `feature`, `path` (remote path).
2. `shutil.copytree(path, {config.preprocessor_storage}/{name}/)`.
3. Validate + register (same as 5a step 3-4).

### 5c. `DELETE /v1/predictor/preprocessor/{name}`

Logic:
1. Remove from preprocessor chain.
2. Delete local storage folder.

### 5d. `GET /v1/predictor/preprocessors`

Logic:
1. List all registered preprocessors with name, feature, path.

### 5e. `GET /v1/predictor/status`

Logic:
1. Query `PredictorLowLevel` for model info (samples count, last trained, metrics).
2. Query current scheduling strategy.
3. List bound preprocessors.
4. Return `ModelStatus`.

References:
- `swarmpilot/predictor/api/core.py:PredictorLowLevel.get_model_info()` — model metadata
- `swarmpilot/scheduler/api.py` — current strategy state

### 5f. `POST /v1/predictor/retrain`

Logic:
1. Call `TrainingClient` or `PredictorLowLevel.train_predictor()` with accumulated samples.
2. On success, switch strategy to `probabilistic` via internal strategy setter.
3. Return `TrainResult` with metrics and new strategy.

References:
- `swarmpilot/scheduler/clients/training_library_client.py` — training buffer
- `swarmpilot/predictor/api/core.py:PredictorCore.train()` — training logic
- Existing strategy setter in `swarmpilot/scheduler/api.py` (search for `strategy/set`)

### 5g. `POST /v1/predictor/train`

Logic:
1. Accept JSON body with custom training data.
2. Train both ExpectError and Quantile predictors.
3. Auto-switch to `probabilistic` strategy.
4. Return `TrainResult`.

### 5h. `GET /v1/predictor/models`

Logic:
1. Query `ModelStorage.list_models()`.
2. Return list of trained model metadata.

References:
- `swarmpilot/predictor/storage/model_storage.py:list_models()`

### 5i. `POST /v1/predictor/predict`

Logic:
1. Accept features JSON.
2. Run through preprocessor chain + MLP prediction.
3. Return `PredictResult`.

References:
- `swarmpilot/predictor/api/core.py:PredictorCore.predict()`

**Config addition needed:** Add `PREDICTOR_PREPROCESSOR_STORAGE` env var to Scheduler config for preprocessor local storage directory.

Reference: `swarmpilot/scheduler/config.py` — add field to `Config` dataclass.

**Testing approach:** Mock `PredictorLowLevel`, `TrainingClient`, filesystem operations. Use `tmp_path` fixture for preprocessor storage. Test multipart upload with `httpx`.

**Step 5: Commit**

```bash
git commit -m "feat(scheduler): add /v1/predictor/* endpoints for preprocessor and MLP management [SDK-005]"
```

---

## Task 6: SDK Client — Core Module

Implement the main SDK: `swarmpilot.init()`, `swarmpilot.connect()`, and the `SwarmPilotClient` class that communicates with Planner REST API. This is the core that `serve()`, `run()`, `register()`, `deploy()`, `ps()`, `scale()`, `terminate()` are built on.

**Files:**
- Create: `swarmpilot/sdk/client.py`
- Modify: `swarmpilot/__init__.py` — expose top-level functions
- Create: `tests/sdk/test_client.py`

**Key implementation details:**

### 6a. `SwarmPilotClient` class

```python
class SwarmPilotClient:
    def __init__(self, planner_url: str):
        self._planner_url = planner_url.rstrip("/")
        self._http = httpx.Client(base_url=self._planner_url, timeout=600)
        self._scheduler_map: dict[str, str] = {}  # model → scheduler URL
        self._refresh_scheduler_map()

    def _refresh_scheduler_map(self) -> None:
        resp = self._http.get("/v1/schedulers")
        resp.raise_for_status()
        self._scheduler_map = resp.json().get("schedulers", {})

    def _resolve_scheduler(self, model: str, scheduler: str | None) -> str | None:
        if scheduler is None:
            return None
        if scheduler != "auto":
            return scheduler
        # auto-discover
        self._refresh_scheduler_map()
        url = self._scheduler_map.get(model)
        if url is None:
            raise SchedulerNotFound(model=model)
        return url

    def _get_scheduler_for_model(self, model: str) -> str:
        self._refresh_scheduler_map()
        url = self._scheduler_map.get(model)
        if url is None:
            raise ModelNotDeployed(model=model)
        return url
```

### 6b. `serve()` method

```python
def serve(
    self,
    model_or_command: str,
    gpu: int,
    replicas: int = 1,
    name: str | None = None,
    scheduler: str | None = "auto",
    memory: int | None = None,
    env: dict | None = None,
) -> Instance | InstanceGroup:
    resp = self._http.post("/v1/serve", json={...})
    # parse into Instance (replicas=1) or InstanceGroup (replicas>1)
```

### 6c. `predictor` sub-object

The `predictor` attribute is a `PredictorManager` that uses `_get_scheduler_for_model()` to direct-connect to the right Scheduler:

```python
class PredictorManager:
    def __init__(self, client: SwarmPilotClient):
        self._client = client

    def upload(self, model: str, name: str, feature: str, path: str) -> None:
        scheduler_url = self._client._get_scheduler_for_model(model)
        # tar.gz pack + multipart upload to scheduler_url
```

### 6d. Global state for `swarmpilot.init()`

```python
# swarmpilot/__init__.py
_global_client: SwarmPilotClient | None = None

def init(planner_url: str) -> None:
    global _global_client
    _global_client = SwarmPilotClient(planner_url)

def connect(planner_url: str) -> SwarmPilotClient:
    return SwarmPilotClient(planner_url)

def _get_client() -> SwarmPilotClient:
    if _global_client is None:
        raise SwarmPilotError("Call swarmpilot.init() first")
    return _global_client

def serve(*args, **kwargs):
    return _get_client().serve(*args, **kwargs)

# ... same pattern for run, register, deploy, ps, scale, terminate
```

**Testing approach:** Mock `httpx.Client` responses. Test each method returns the correct SDK model type. Test error cases (SchedulerNotFound, ModelNotDeployed). Test `init()`/`connect()` global state. Test `predictor.upload()` packs tar.gz correctly.

**Step 5: Commit**

```bash
git commit -m "feat(sdk): implement SwarmPilotClient with serve, run, register, deploy, ps, scale, terminate, predictor [SDK-006]"
```

---

## Task 7: SDK Client — wait_ready() and Polling

Implement the async handle lifecycle: `Instance.wait_ready()`, `InstanceGroup.wait_ready()`, `Process.wait_ready()`, `DeploymentResult.wait_ready()`, and `Instance.terminate()`, `InstanceGroup.scale()`, `InstanceGroup.terminate()`, `Process.terminate()`.

These methods need a back-reference to the `SwarmPilotClient` so they can poll the Planner for status updates.

**Files:**
- Modify: `swarmpilot/sdk/models.py` — add `_client` field and methods
- Modify: `swarmpilot/sdk/client.py` — inject client reference when constructing models
- Modify: `tests/sdk/test_models.py` — add lifecycle tests
- Create: `tests/sdk/test_lifecycle.py`

**Key implementation:**

```python
@dataclass
class Instance:
    # ... existing fields ...
    _client: SwarmPilotClient | None = field(default=None, repr=False)

    def wait_ready(self, timeout: int = 300) -> None:
        """Poll Planner until instance is running or timeout."""
        # GET /v1/instances/{self.name} repeatedly
        # Update self.endpoint, self.status on each poll

    def terminate(self) -> None:
        """Terminate this instance."""
        # POST /v1/terminate {name: self.name}
```

**Testing approach:** Mock Planner responses to simulate status transitions: `pending → deploying → running`. Test timeout raises `TimeoutError`. Test terminate calls the right endpoint.

**Step 5: Commit**

```bash
git commit -m "feat(sdk): add wait_ready(), terminate(), scale() lifecycle methods [SDK-007]"
```

---

## Task 8: CLI Extension — splanner Commands

Extend the `splanner` CLI with new Typer commands that delegate to the Planner's new REST API.

**Files:**
- Modify: `swarmpilot/planner/cli.py` — add `serve`, `run`, `register`, `deploy`, `ps`, `info`, `scale`, `terminate` commands and `predictor` sub-app
- Create: `tests/planner/test_cli_commands.py`

**Commands to implement:**

The CLI uses `httpx` to call the Planner (address from `--planner-url` option or `PLANNER_URL` env var, default `http://localhost:8002`).

### 8a. Core commands

```python
@app.command()
def serve(
    model_or_command: str = typer.Argument(...),
    gpu: int = typer.Option(..., "--gpu"),
    replicas: int = typer.Option(1, "--replicas"),
    name: str | None = typer.Option(None, "--name"),
    scheduler: str | None = typer.Option("auto", "--scheduler"),
    no_scheduler: bool = typer.Option(False, "--no-scheduler"),
    wait: bool = typer.Option(False, "--wait"),
    timeout: int = typer.Option(300, "--timeout"),
):
    """Deploy a model service or custom inference server."""
    # POST /v1/serve
    # If --wait: poll /v1/instances/{name} until ready

@app.command()
def run(
    command: str = typer.Argument(...),
    gpu: int = typer.Option(0, "--gpu"),
    name: str = typer.Option(..., "--name"),
):
    """Start a custom workload."""

@app.command()
def register(
    model: str = typer.Argument(...),
    gpu: int = typer.Option(..., "--gpu"),
    target_qps: float = typer.Option(..., "--target-qps"),
):
    """Register model requirements for optimized deployment."""

@app.command()
def deploy(
    wait: bool = typer.Option(False, "--wait"),
    timeout: int = typer.Option(600, "--timeout"),
):
    """Trigger optimized deployment of all registered models."""

@app.command()
def ps(model: str | None = typer.Option(None, "--model")):
    """List all instances."""
    # GET /v1/instances, format as table

@app.command()
def info(name: str = typer.Argument(...)):
    """Show instance details."""
    # GET /v1/instances/{name}

@app.command()
def scale(
    model: str = typer.Argument(...),
    replicas: int = typer.Option(..., "--replicas"),
):
    """Scale a model's replicas."""

@app.command()
def terminate(
    name: str | None = typer.Argument(None),
    model: str | None = typer.Option(None, "--model"),
    all_instances: bool = typer.Option(False, "--all"),
):
    """Terminate instances."""
```

### 8b. Predictor sub-app

```python
predictor_app = typer.Typer(name="predictor", help="Predictor management")
app.add_typer(predictor_app)

@predictor_app.command()
def upload(
    model: str = typer.Option(..., "--model"),
    name: str = typer.Option(..., "--name"),
    feature: str = typer.Option(..., "--feature"),
    path: str = typer.Option(..., "--path"),
):
    """Upload a preprocessor to a model's Scheduler."""
    # 1. GET /v1/schedulers → find scheduler for model
    # 2. tar.gz pack path
    # 3. POST {scheduler}/v1/predictor/preprocessor/upload

# ... register, remove, preprocessors, status, retrain, models, predict
```

**Testing approach:** Use `typer.testing.CliRunner`. Mock `httpx` calls. Test output formatting, error messages, `--wait` polling behavior.

**Step 5: Commit**

```bash
git commit -m "feat(cli): extend splanner with serve, run, register, deploy, ps, scale, terminate, predictor [SDK-008]"
```

---

## Task 9: Strategy Auto-Switch on Training

When MLP training completes successfully via the new `/v1/predictor/retrain` or `/v1/predictor/train` endpoints, the Scheduler should automatically switch to `probabilistic` strategy.

**Files:**
- Modify: `swarmpilot/scheduler/routes/predictor_api.py` — add strategy switch after training
- Modify: `tests/scheduler/test_predictor_api.py` — add strategy switch tests

**Key logic:**

In the `retrain` and `train` endpoint handlers, after successful training:

```python
# After training succeeds:
from swarmpilot.scheduler.algorithms import set_strategy
set_strategy("probabilistic")
```

**Testing:** Verify that after calling `POST /v1/predictor/retrain`, the current strategy changes to `probabilistic`. Verify that if training fails (e.g., <10 samples), strategy stays unchanged.

References:
- Check how `POST /v1/strategy/set` is implemented in `swarmpilot/scheduler/api.py` — reuse same internal function
- Check `swarmpilot/scheduler/algorithms/` for strategy selection mechanism

**Step 5: Commit**

```bash
git commit -m "feat(scheduler): auto-switch to probabilistic strategy after MLP training [SDK-009]"
```

---

## Task 10: Integration Tests

End-to-end tests that verify the complete flow without a real PyLet cluster.

**Files:**
- Create: `tests/integration/test_sdk_e2e.py`
- Create: `tests/integration/conftest.py`

**Test scenarios:**

### 10a. Manual deployment flow

```python
@pytest.mark.integration
async def test_serve_and_terminate():
    """serve → wait_ready → check scheduler → terminate"""
    # 1. Start Planner + Scheduler in-process (TestClient)
    # 2. Mock PyLetClient to return fake instances
    # 3. swarmpilot.init(planner_url)
    # 4. inst = swarmpilot.serve("Qwen/Qwen2.5-7B", gpu=1)
    # 5. assert inst.scheduler == expected_scheduler_url
    # 6. inst.terminate()
```

### 10b. Optimized deployment flow

```python
@pytest.mark.integration
async def test_register_deploy():
    """register → deploy → check plan → check scheduler"""
```

### 10c. Custom workload flow

```python
@pytest.mark.integration
async def test_run_custom_workload():
    """run → wait_ready → check endpoint → check scheduler is None"""
```

### 10d. Predictor flow

```python
@pytest.mark.integration
async def test_predictor_upload_and_train():
    """serve → upload preprocessor → train → check strategy switch"""
```

### 10e. Error scenarios

```python
@pytest.mark.integration
async def test_scheduler_not_found():
    """serve with auto-discover, no scheduler registered → SchedulerNotFound"""

@pytest.mark.integration
async def test_model_not_deployed_predictor():
    """predictor.upload before serve → ModelNotDeployed"""
```

**Step 5: Commit**

```bash
git commit -m "test: add SDK end-to-end integration tests [SDK-010]"
```

---

## Task 11: Documentation Update

Update existing docs to reflect the new SDK/CLI interface.

**Files:**
- Modify: `docs/DEPLOYMENT_GUIDE.md` — add SDK/CLI sections, mark old curl examples as legacy
- Modify: `docs/SYSTEM_WORKFLOW.md` — update deployment flow section
- Modify: `CLAUDE.md` — add SDK package to project structure, update quick reference

**Step 5: Commit**

```bash
git commit -m "docs: update deployment guide and workflow for new SDK/CLI interface [SDK-011]"
```

---

## Task Summary

| Task | Component | Scope | Depends On |
|------|-----------|-------|------------|
| 1 | Error types | `swarmpilot/errors.py` | — |
| 2 | SDK data models | `swarmpilot/sdk/models.py` | — |
| 3 | Planner request/response models | `swarmpilot/planner/models/sdk_api.py` | — |
| 4 | Planner REST endpoints | `swarmpilot/planner/routes/sdk_api.py` | 1, 3 |
| 5 | Scheduler predictor endpoints | `swarmpilot/scheduler/routes/predictor_api.py` | 1 |
| 6 | SDK client core | `swarmpilot/sdk/client.py`, `swarmpilot/__init__.py` | 1, 2, 3 |
| 7 | SDK lifecycle methods | `swarmpilot/sdk/models.py` (wait_ready, terminate) | 2, 6 |
| 8 | CLI commands | `swarmpilot/planner/cli.py` | 3, 4 |
| 9 | Strategy auto-switch | `swarmpilot/scheduler/routes/predictor_api.py` | 5 |
| 10 | Integration tests | `tests/integration/` | 4, 5, 6, 7 |
| 11 | Documentation | `docs/` | All |

**Parallelizable:** Tasks 1, 2, 3 can run in parallel. Tasks 4 and 5 can run in parallel. Tasks 6 and 8 can run in parallel (once their deps are done).

```
     1 ──┐
     2 ──┼── 6 ── 7 ──┐
     3 ──┤             ├── 10 ── 11
     1 ──┼── 4 ────────┤
     3 ──┘             │
     1 ── 5 ── 9 ─────┘
     3 ── 8 ───────────┘
```
