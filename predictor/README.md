# Predictor

MLP-based runtime prediction service with expect/error and quantile regression support. Provides REST API and WebSocket endpoints for training and real-time predictions.

## Quick Start

```bash
# Install dependencies
uv sync

# Install CLI
uv pip install -e .

# Start service
spredictor start

# Start with auto-reload (development)
spredictor start --reload

# Start with custom configuration
spredictor start --port 8080 --storage-dir ./models
```

The service will be available at `http://localhost:8000`.

## CLI Commands

```bash
spredictor start              # Start the service
spredictor health             # Check service health
spredictor version            # Show version info
spredictor list               # List trained models
spredictor config show        # Show current configuration
spredictor config init        # Create config file template
```

## Configuration

Configuration priority: **CLI arguments** > **Environment variables** > **Config file** > **Defaults**

### CLI Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--host` | `-h` | string | `0.0.0.0` | Host to bind the server |
| `--port` | `-p` | integer | `8000` | Port to bind the server |
| `--reload` | - | flag | `false` | Enable auto-reload (development) |
| `--workers` | `-w` | integer | `1` | Number of worker processes |
| `--storage-dir` | `-s` | string | `models` | Model storage directory |
| `--log-level` | `-l` | string | `info` | Logging level (debug/info/warning/error/critical) |
| `--config` | `-c` | path | - | Path to configuration file |

### Environment Variables

All configuration options can be set via environment variables with `PREDICTOR_` prefix:

```bash
export PREDICTOR_HOST=127.0.0.1
export PREDICTOR_PORT=8080
export PREDICTOR_RELOAD=false
export PREDICTOR_WORKERS=4
export PREDICTOR_STORAGE_DIR=/path/to/models
export PREDICTOR_LOG_LEVEL=info
export PREDICTOR_APP_NAME="Runtime Predictor Service"
export PREDICTOR_APP_VERSION=0.1.0
```

### Configuration File

Create a `predictor.toml` file (or use `spredictor config init`):

```toml
[predictor]
host = "0.0.0.0"
port = 8000
reload = false
workers = 1
storage_dir = "models"
log_level = "info"
app_name = "Runtime Predictor Service"
app_version = "0.1.0"
```

Then start with: `spredictor start --config predictor.toml`

## API Endpoints

- `GET /health` - Health check
- `GET /list` - List all trained models
- `POST /train` - Train or update a model
- `POST /predict` - Make predictions
- `WS /ws/predict` - WebSocket for real-time predictions

API documentation available at `http://localhost:8000/docs` when service is running.
