# Instance Service

A lightweight execution service for running model containers with task queue management. Each instance serves one model at a time, processing inference requests sequentially via Docker-based isolation.

## Quick Start

### Prerequisites

- Python 3.13+
- Docker
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
cd instance

# Install dependencies and package
uv sync

# Or install via pip
pip install .
```

### Start the Service

```bash
# Development mode
uv run sinstance start

# After pip install
sinstance start

# With custom configuration
INSTANCE_ID=my-instance INSTANCE_PORT=8000 sinstance start

# Development mode with auto-reload
sinstance start --reload
```

The service starts on `http://localhost:5000` by default.

### Quick Test

```bash
# 1. Start the service
sinstance start

# 2. Check health (in another terminal)
curl http://localhost:5000/health

# 3. Start the sleep model
curl -X POST http://localhost:5000/model/start \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sleep_model", "parameters": {}}'

# 4. Submit a task
curl -X POST http://localhost:5000/task/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-001",
    "model_id": "sleep_model",
    "task_input": {"sleep_time": 3}
  }'

# 5. Check task status
curl http://localhost:5000/task/task-001
```

### Configuration

Configure via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `INSTANCE_ID` | Unique instance identifier | `instance-default` |
| `INSTANCE_PORT` | Instance API server port | `5000` |
| `INSTANCE_LOG_LEVEL` | Logging level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL) | `INFO` |
| `INSTANCE_LOG_DIR` | Directory for log files | `logs` |
| `INSTANCE_ENABLE_JSON_LOGS` | Enable JSON structured logging | `false` |
| `MAX_QUEUE_SIZE` | Maximum tasks in queue | `100` |
| `DOCKER_NETWORK` | Docker network name | `instance_network` |
| `HEALTH_CHECK_INTERVAL` | Health check interval (seconds) | `10` |
| `HEALTH_CHECK_TIMEOUT` | Health check timeout (seconds) | `30` |

**Note:** Model containers run on `INSTANCE_PORT + 1000`.

### CLI Options

```bash
sinstance start [OPTIONS]
  --host, -h           Host to bind (default: 0.0.0.0)
  --port, -p           Port to bind (default: 5000)
  --log-level, -l      Logging level (default: INFO)
  --reload             Enable auto-reload for development

sinstance version      Show version information
```

### Logging

The Instance Service uses [loguru](https://github.com/Delgan/loguru) for structured logging with support for:

- **Colored console output** - Easy-to-read logs with colored levels
- **File rotation** - Daily log files with automatic rotation at midnight
- **Compression** - Old log files are automatically compressed as ZIP files
- **Retention** - Logs are kept for 30 days by default
- **JSON logging** - Optional structured JSON output for log aggregation systems
- **Unified logging** - Intercepts and formats logs from uvicorn, FastAPI, and httpx

#### Log Configuration Examples

```bash
# Set log level to DEBUG
INSTANCE_LOG_LEVEL=DEBUG sinstance start

# Custom log directory
INSTANCE_LOG_DIR=/var/log/instance sinstance start

# Enable JSON structured logs (for log aggregation)
INSTANCE_ENABLE_JSON_LOGS=true sinstance start

# Combined example
INSTANCE_LOG_LEVEL=DEBUG \
INSTANCE_LOG_DIR=/var/log/instance \
INSTANCE_ENABLE_JSON_LOGS=true \
sinstance start
```

#### Log Files

Logs are written to two locations:

1. **Console** (stderr): Colored output for development
2. **File**: `{INSTANCE_LOG_DIR}/instance_YYYY-MM-DD.log`
3. **JSON** (optional): `{INSTANCE_LOG_DIR}/instance_YYYY-MM-DD.json`

Log format:
```
2025-11-02 15:16:41.173 | INFO     | module:function:line - message
```

---

**Documentation:** See [docs/](./docs/) for detailed guides on API, architecture, and model containers.
