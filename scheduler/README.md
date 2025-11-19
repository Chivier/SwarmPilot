# Scheduler Service

Intelligent task scheduling service that distributes tasks across multiple compute instances using ML-based runtime predictions and configurable scheduling strategies.

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Or install as a package
uv pip install -e .
```

### Usage

```bash
# Start the scheduler service
sscheduler start

# Start with custom host and port
sscheduler start --host 0.0.0.0 --port 8080

# Start with configuration file
sscheduler start --config config.toml

# Show version
sscheduler version

# Show help
sscheduler --help
```

### Basic Configuration

Set environment variables to configure the scheduler:

```bash
# Required: Predictor service URL
export PREDICTOR_URL=http://localhost:8001

# Optional: Server settings
export SCHEDULER_HOST=0.0.0.0
export SCHEDULER_PORT=8000

# Optional: Scheduling strategy (min_time, probabilistic, round_robin)
export SCHEDULING_STRATEGY=probabilistic
```

### All Environment Variables

#### Server Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SCHEDULER_HOST` | `str` | `"0.0.0.0"` | Server bind address |
| `SCHEDULER_PORT` | `int` | `8000` | Server port |
| `SCHEDULER_ENABLE_CORS` | `bool` | `true` | Enable CORS |
| `SCHEDULER_LOGURU_LEVEL` | `str` | `"INFO"` | Log level (TRACE, DEBUG, INFO, WARNING, ERROR) |
| `SCHEDULER_LOG_DIR` | `str` | `"logs"` | Log directory |
| `SCHEDULER_ENABLE_JSON_LOGS` | `bool` | `false` | Enable JSON structured logging |

#### Predictor Integration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PREDICTOR_URL` | `str` | `"http://localhost:8001"` | Predictor service URL |
| `PREDICTOR_TIMEOUT` | `float` | `5.0` | Request timeout (seconds) |
| `PREDICTOR_MAX_RETRIES` | `int` | `3` | Max retry attempts |
| `PREDICTOR_RETRY_DELAY` | `float` | `1.0` | Initial retry delay (seconds) |

#### Scheduling Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SCHEDULING_STRATEGY` | `str` | `"probabilistic"` | Strategy: `min_time`, `probabilistic`, `round_robin` |
| `SCHEDULING_PROBABILISTIC_QUANTILE` | `float` | `0.9` | Target quantile for probabilistic scheduling |

#### Queue Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `QUEUE_HIGH_WATER_MARK` | `int` | `6` | Max pending tasks per instance before stopping dispatch |
| `QUEUE_LOW_WATER_MARK` | `int` | `3` | Resume dispatching when pending tasks drop below this |
| `QUEUE_MAX_CONCURRENT_DISPATCH` | `int` | `50` | Maximum concurrent dispatch operations |

#### Training Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TRAINING_ENABLE_AUTO` | `bool` | `false` | Enable automatic model training |
| `TRAINING_BATCH_SIZE` | `int` | `100` | Batch size for training data |
| `TRAINING_FREQUENCY` | `int` | `3600` | Training frequency (seconds) |
| `TRAINING_MIN_SAMPLES` | `int` | `10` | Minimum samples before training |
| `TRAINING_PREDICTION_TYPES` | `str` | `"expect_error,quantile"` | Prediction types to train (comma-separated) |

#### Planner Reporter Configuration

Automatically report uncompleted task counts to the planner service for auto-optimization.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PLANNER_URL` | `str` | `""` | Planner service URL (empty to disable) |
| `SCHEDULER_AUTO_REPORT` | `float` | `0` | Report interval in seconds (0 to disable) |
| `PLANNER_REPORT_TIMEOUT` | `float` | `5.0` | HTTP request timeout (seconds) |

**Example:**
```bash
# Enable auto-reporting to planner every 5 seconds
export PLANNER_URL=http://localhost:8002
export SCHEDULER_AUTO_REPORT=5
export PLANNER_REPORT_TIMEOUT=5.0
```

**Planner Reporter Workflow:**
1. When first instance registers, captures its `model_id`
2. Every `SCHEDULER_AUTO_REPORT` seconds, reports:
   - Total uncompleted tasks (`pending + running`)
   - Model ID to planner's `/submit_target` endpoint
3. Planner uses this data for auto-optimization decisions

For detailed documentation, see the [docs](docs/) directory.
