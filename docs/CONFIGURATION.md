# Configuration

All environment variables and CLI flags for SwarmPilot services. Every setting has a sensible default; none are required for local development.

---

## Scheduler

Source: `swarmpilot/scheduler/config.py`

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_HOST` | `0.0.0.0` | Bind address |
| `SCHEDULER_PORT` | `8000` | Bind port |
| `SCHEDULER_ENABLE_CORS` | `true` | Enable CORS middleware |

### Scheduling

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULING_STRATEGY` | `adaptive_bootstrap` | Default strategy. Options: `adaptive_bootstrap`, `min_time`, `probabilistic`, `round_robin`, `random`, `po2`, `severless` |
| `SCHEDULING_PROBABILISTIC_QUANTILE` | `0.9` | Target quantile for probabilistic strategy (0.0-1.0) |

### Training (Auto-Training Pipeline)

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_ENABLE_AUTO` | `false` | Enable automatic predictor model retraining |
| `TRAINING_BATCH_SIZE` | `100` | Batch size for training data collection |
| `TRAINING_FREQUENCY` | `3600` | Training frequency in seconds |
| `TRAINING_MIN_SAMPLES` | `10` | Minimum samples before training starts |
| `TRAINING_PREDICTION_TYPES` | `quantile` | Comma-separated prediction types to train |

### Predictor (Library Mode)

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTOR_STORAGE_DIR` | `models` | Model file storage directory |
| `PREDICTOR_CACHE_MAX_SIZE` | `100` | Max models in memory cache |

### Preprocessor

| Variable | Default | Description |
|----------|---------|-------------|
| `PREPROCESSOR_CONFIG_FILE` | _(empty)_ | Path to preprocessor rules JSON. Empty disables preprocessors |
| `PREPROCESSOR_STRICT` | `true` | Fail if configured preprocessor model is unavailable |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_LOGURU_LEVEL` | `INFO` | Log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL) |
| `SCHEDULER_LOG_DIR` | `logs` | Log file directory |
| `SCHEDULER_ENABLE_JSON_LOGS` | `false` | Enable structured JSON logging |

### Planner Reporting

| Variable | Default | Description |
|----------|---------|-------------|
| `PLANNER_URL` | _(empty)_ | Planner service URL. Empty disables reporting |
| `SCHEDULER_AUTO_REPORT` | `0` | Auto-report interval in seconds. `0` disables |
| `PLANNER_REPORT_TIMEOUT` | `5.0` | HTTP timeout for planner reports (seconds) |

### Proxy

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_ENABLED` | `true` | Enable transparent proxy router |
| `PROXY_TIMEOUT` | `300.0` | Proxy request timeout (seconds) |
| `WORKER_HTTP_TIMEOUT` | `300.0` | Worker queue thread HTTP timeout (seconds) |

### Planner Registration

When set, the Scheduler registers itself with a Planner on startup for multi-scheduler routing.

| Variable | Default | Description |
|----------|---------|-------------|
| `PLANNER_REGISTRATION_URL` | _(empty)_ | Planner URL for registration. Empty disables |
| `SCHEDULER_MODEL_ID` | _(empty)_ | Model ID this scheduler handles |
| `SCHEDULER_SELF_URL` | _(empty)_ | Advertised URL for this scheduler |
| `PLANNER_REGISTRATION_TIMEOUT` | `10.0` | Registration request timeout (seconds) |
| `PLANNER_REGISTRATION_MAX_RETRIES` | `3` | Max registration retries |
| `PLANNER_REGISTRATION_RETRY_DELAY` | `5.0` | Delay between retries (seconds) |

All three (`PLANNER_REGISTRATION_URL`, `SCHEDULER_MODEL_ID`, `SCHEDULER_SELF_URL`) must be set to enable registration.

---

## Predictor

Source: `swarmpilot/predictor/config.py`. Uses pydantic-settings with the `PREDICTOR_` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTOR_HOST` | `0.0.0.0` | Bind address |
| `PREDICTOR_PORT` | `8000` | Bind port |
| `PREDICTOR_RELOAD` | `false` | Enable auto-reload (development) |
| `PREDICTOR_WORKERS` | `1` | Number of worker processes |
| `PREDICTOR_STORAGE_DIR` | `models` | Trained model storage directory |
| `PREDICTOR_LOG_LEVEL` | `info` | Log level (debug, info, warning, error, critical) |
| `PREDICTOR_LOG_DIR` | `logs` | Log file directory |

The Predictor also reads a `.env` file if present and supports `predictor.toml` configuration files.

---

## Planner

Source: `swarmpilot/planner/config.py`

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `PLANNER_HOST` | `0.0.0.0` | Bind address |
| `PLANNER_PORT` | `8000` | Bind port |
| `SCHEDULER_URL` | _(none)_ | Default scheduler URL |

### Instance Management

| Variable | Default | Description |
|----------|---------|-------------|
| `INSTANCE_TIMEOUT` | `30` | Instance operation timeout (seconds) |
| `INSTANCE_MAX_RETRIES` | `3` | Max retries for instance operations |
| `INSTANCE_RETRY_DELAY` | `1.0` | Delay between retries (seconds) |

### Auto-Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_OPTIMIZE_ENABLED` | `false` | Enable periodic automatic optimization |
| `AUTO_OPTIMIZE_INTERVAL` | `60.0` | Optimization interval (seconds) |

### PyLet

| Variable | Default | Description |
|----------|---------|-------------|
| `PYLET_ENABLED` | `false` | Enable PyLet cluster integration |
| `PYLET_HEAD_URL` | _(none)_ | PyLet head node URL. **Required** when `PYLET_ENABLED=true` |
| `PYLET_BACKEND` | `vllm` | Backend engine: `vllm` or `sglang` |
| `PYLET_GPU_COUNT` | `1` | GPUs per instance |
| `PYLET_CPU_COUNT` | `1` | CPUs per instance |
| `PYLET_DEPLOY_TIMEOUT` | `300.0` | Deployment timeout (seconds) |
| `PYLET_DRAIN_TIMEOUT` | `30.0` | Drain timeout before termination (seconds) |
| `PYLET_CUSTOM_COMMAND` | _(none)_ | Custom command template (overrides backend). Use `$PORT` for auto-port |
| `PYLET_REUSE_CLUSTER` | `false` | Reuse existing PyLet cluster (skip `pylet.init`) |

---

## CLI Reference

All three tools accept `--help` for full option lists.

### sscheduler

```bash
sscheduler start [OPTIONS]
  --host TEXT     Bind host (default: from SCHEDULER_HOST)
  --port INT      Bind port (default: from SCHEDULER_PORT)
```

### spredictor

```bash
spredictor start [OPTIONS]
  --host TEXT     Bind host (default: from PREDICTOR_HOST)
  --port INT      Bind port (default: from PREDICTOR_PORT)
  --reload        Enable auto-reload
  --workers INT   Number of workers
```

### splanner

```bash
splanner start [OPTIONS]
  --host TEXT     Bind host (default: from PLANNER_HOST)
  --port INT      Bind port (default: from PLANNER_PORT)
```
