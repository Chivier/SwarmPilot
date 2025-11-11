# Planner Service

Model deployment optimization service for SwarmPilot that computes optimal model-to-instance assignments using configurable algorithms.

## Quick Start

### Installation

```bash
# Install using pip
pip install .

# Or install using uv (recommended)
uv sync
```

### Configuration

The service can be configured via environment variables. Create a `.env` file or set variables directly:

```bash
# Scheduler URL for instance registration (optional)
SCHEDULER_URL=http://localhost:8100

# Instance deployment settings
INSTANCE_TIMEOUT=30          # HTTP request timeout in seconds
INSTANCE_MAX_RETRIES=3       # Max retry attempts for failed requests
INSTANCE_RETRY_DELAY=1.0     # Initial retry delay (exponential backoff)
```

See `.env.example` for complete configuration options.

### Starting the Service

```bash
# Using the CLI
splanner start

# With custom configuration
splanner start --host 127.0.0.1 --port 9000 --log-level debug

# Or using uv run
uv run splanner start
```

The service will be available at `http://localhost:8000`.

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Basic Usage

```bash
# Health check
curl http://localhost:8000/health

# Service information
curl http://localhost:8000/info

# Compute optimal deployment plan
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "M": 4,
    "N": 3,
    "B": [[10, 5, 0], [8, 6, 4], [0, 10, 8], [6, 0, 12]],
    "initial": [0, 1, 2, 2],
    "a": 0.5,
    "target": [20, 30, 25],
    "algorithm": "simulated_annealing"
  }'
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src
```
