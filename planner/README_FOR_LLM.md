# Swarm Planner Service - LLM Documentation

## 1. SERVICE OVERVIEW

### 1.1 Purpose
The **Swarm Planner** is an optimization service that computes optimal model deployment strategies across multiple instance servers. It uses mathematical optimization algorithms (Simulated Annealing or Integer Programming) to determine which models should run on which instances to minimize resource usage while satisfying computational requirements.

The planner integrates with **PyLet** for cluster management, providing deploy, scale, migrate, and optimize operations through a unified API.

### 1.2 Technology Stack
```
Framework:          FastAPI (async REST API)
Web Server:         Uvicorn (ASGI server)
CLI:                Typer
Optimization:       PuLP (Integer Programming), NumPy (Simulated Annealing)
HTTP Client:        httpx (async)
Data Validation:    Pydantic v2
Cluster Management: PyLet integration
Testing:            pytest, pytest-asyncio
Package Manager:    uv
Python Version:     3.13 (minimum 3.11)
```

### 1.3 Architecture
- **Type**: Standalone HTTP microservice with PyLet integration
- **State**: Stateless (no database, no persistent storage)
- **Communication**: HTTP/REST with PyLet head service
- **Port**: 8000 (default)
- **Host**: 0.0.0.0 (binds to all interfaces)

---

## 2. QUICK START

### 2.1 Installation
```bash
# Clone repository and navigate to planner directory
cd /path/to/swarmpilot-refresh/planner

# Install dependencies using uv
uv sync

# Verify installation
uv run splanner --help
```

### 2.2 Starting the Service
```bash
# Start with default settings (0.0.0.0:8000)
uv run splanner start

# Start with custom host/port
uv run splanner start --host 127.0.0.1 --port 9000

# Start with debug logging
uv run splanner start --log-level debug

# Start with PyLet integration
PYLET_ENABLED=true PYLET_HEAD_URL=http://pylet-head:8000 uv run splanner start
```

### 2.3 Health Check
```bash
curl http://localhost:8000/health
# Response: {"status": "healthy", "timestamp": "..."}
```

---

## 3. ENVIRONMENT & CONFIGURATION

### 3.1 Environment Variables

#### Basic Configuration
```bash
PLANNER_HOST=0.0.0.0        # Host to bind to
PLANNER_PORT=8000            # Port to bind to
SCHEDULER_URL=http://localhost:8100    # Scheduler URL for instance registration
INSTANCE_TIMEOUT=30          # HTTP request timeout (seconds)
INSTANCE_MAX_RETRIES=3       # Max retry attempts for failed requests
INSTANCE_RETRY_DELAY=1.0     # Initial retry delay (exponential backoff)
```

#### PyLet Configuration
```bash
PYLET_ENABLED=false          # Enable PyLet integration
PYLET_HEAD_URL=              # PyLet head service URL (required if enabled)
PYLET_BACKEND=vllm           # Default backend (vllm or sglang)
PYLET_GPU_COUNT=1            # Default GPU count per instance
PYLET_DEPLOY_TIMEOUT=300.0   # Deployment timeout (seconds)
PYLET_DRAIN_TIMEOUT=60.0     # Drain timeout (seconds)
```

All environment variables are **OPTIONAL**. The service will use sensible defaults if not provided.

### 3.2 CLI Configuration Options
```
--host TEXT         Host to bind to [default: 0.0.0.0]
--port INTEGER      Port to bind to [default: 8000]
--log-level TEXT    Logging level [default: info]
                    Options: debug, info, warning, error, critical
--reload            Enable auto-reload (development only)
```

---

## 4. API ENDPOINTS REFERENCE

### 4.1 GET /health

**Purpose**: Health check endpoint to verify service availability.

**Response Schema**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-07T12:00:00Z"
}
```

---

### 4.2 GET /info

**Purpose**: Returns service metadata including version and available optimization algorithms.

**Response Schema**:
```json
{
  "service": "planner",
  "version": "0.1.0",
  "algorithms": ["simulated_annealing", "integer_programming"],
  "objective_methods": ["relative_error", "ratio_difference", "weighted_squared"],
  "description": "Model deployment optimization service"
}
```

---

### 4.3 POST /plan

**Purpose**: Compute optimal deployment plan WITHOUT executing it.

**Request Schema (PlannerInput)**:
```json
{
  "M": 4,                                    // Number of instances (> 0)
  "N": 3,                                    // Number of model types (> 0)
  "B": [[10, 5, 0], [8, 6, 4], ...],         // Batch capacity matrix [M×N]
  "initial": [0, 1, 2, 2],                   // Initial deployment [M], -1 = no model
  "a": 0.5,                                  // Change constraint (0 < a ≤ 1)
  "target": [20, 30, 25],                    // Target request distribution [N]
  "algorithm": "simulated_annealing",        // Algorithm selection
  "objective_method": "relative_error",      // Objective function
  "verbose": true                            // Enable logging
}
```

**Response Schema (PlannerOutput)**:
```json
{
  "deployment": [0, 1, 1, 2],               // Optimized assignment [M]
  "score": 0.0667,                          // Objective value (lower = better)
  "stats": {                                // Algorithm statistics
    "algorithm": "simulated_annealing",
    "iterations": 5000,
    "acceptance_rate": 0.247
  },
  "service_capacity": [10.0, 16.0, 12.0],   // Capacity per model [N]
  "changes_count": 1                        // Changes from initial state
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "M": 4, "N": 3,
    "B": [[10, 5, 0], [8, 6, 4], [0, 10, 8], [6, 0, 12]],
    "initial": [0, 1, 2, 2],
    "a": 0.5,
    "target": [20, 30, 25]
  }'
```

---

### 4.4 POST /instance/register

**Purpose**: Register an instance with the planner for tracking.

**Request Schema**:
```json
{
  "instance_id": "inst-123",
  "model_id": "model_a",
  "endpoint": "http://instance:8080",
  "platform_info": {
    "software_name": "vllm",
    "software_version": "0.3.0",
    "hardware_name": "nvidia-a100"
  }
}
```

**Response Schema**:
```json
{
  "success": true,
  "message": "Instance inst-123 registered successfully for model model_a"
}
```

---

## 5. PYLET API ENDPOINTS

PyLet endpoints provide cluster management for deploying and managing model instances.

### 5.1 GET /pylet/status

**Purpose**: Get PyLet service status and active instances.

**Response Schema**:
```json
{
  "enabled": true,
  "initialized": true,
  "current_state": {"model-a": 2, "model-b": 1},
  "total_instances": 3,
  "active_instances": [
    {
      "pylet_id": "p1",
      "instance_id": "i1",
      "model_id": "model-a",
      "endpoint": "http://localhost:8001",
      "status": "active",
      "error": null
    }
  ]
}
```

---

### 5.2 POST /pylet/deploy

**Purpose**: Deploy instances to target state via PyLet.

**Request Schema**:
```json
{
  "target_state": {"model-a": 2, "model-b": 1},
  "wait_for_ready": true,
  "register_with_scheduler": true
}
```

**Response Schema**:
```json
{
  "success": true,
  "added_count": 2,
  "removed_count": 0,
  "active_instances": [...],
  "failed_adds": [],
  "failed_removes": [],
  "error": null
}
```

---

### 5.3 POST /pylet/scale

**Purpose**: Scale a specific model to target count.

**Request Schema**:
```json
{
  "model_id": "model-a",
  "target_count": 3,
  "wait_for_ready": true
}
```

**Response Schema**:
```json
{
  "success": true,
  "model_id": "model-a",
  "previous_count": 2,
  "current_count": 3,
  "added": 1,
  "removed": 0,
  "active_instances": [...],
  "error": null
}
```

---

### 5.4 POST /pylet/migrate

**Purpose**: Migrate an instance to a different model.

**Request Schema**:
```json
{
  "pylet_id": "p1",
  "target_model_id": "model-b"
}
```

**Response Schema**:
```json
{
  "success": true,
  "old_pylet_id": "p1",
  "new_pylet_id": "p2",
  "model_id": "model-b",
  "endpoint": "http://localhost:8002",
  "error": null
}
```

---

### 5.5 POST /pylet/optimize

**Purpose**: Run optimizer and deploy result via PyLet.

**Request Schema**:
```json
{
  "target": [0.5, 0.3, 0.2],
  "model_ids": ["model-a", "model-b", "model-c"],
  "B": [[1.0, 0.8, 0.6], [0.9, 1.0, 0.7]],
  "a": 0.3,
  "objective_method": "ratio_difference",
  "algorithm": "simulated_annealing",
  "wait_for_ready": true
}
```

**Response Schema**:
```json
{
  "deployment": [0, 1],
  "objective_value": 0.05,
  "service_capacity": [1.0, 1.0, 0.0],
  "changes": 1,
  "optimization_stats": {...},
  "deployment_success": true,
  "added_count": 2,
  "removed_count": 0,
  "active_instances": [...],
  "error": null
}
```

---

### 5.6 POST /pylet/terminate-all

**Purpose**: Terminate all PyLet-managed instances.

**Query Parameters**:
- `wait_for_drain` (bool, default: false): Wait for instances to drain before termination

**Response Schema**:
```json
{
  "success": true,
  "total": 3,
  "succeeded": 3,
  "failed": 0,
  "details": {"p1": true, "p2": true, "p3": true}
}
```

---

## 6. SCHEDULER COMPATIBILITY ENDPOINTS

These endpoints provide compatibility with the scheduler interface.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/instance/drain` | POST | Signal instance to start draining |
| `/instance/drain/status` | GET | Check if instance can be safely removed |
| `/instance/remove` | POST | Remove instance from tracking |
| `/task/resubmit` | POST | Resubmit task from failed instance |
| `/timeline` | GET | Get timeline entries |
| `/timeline/clear` | POST | Clear timeline entries |

---

## 7. CORE ALGORITHMS

### 7.1 Algorithm Selection

| Algorithm | Best For | Trade-off | Key Parameters |
|-----------|----------|-----------|----------------|
| **Simulated Annealing** | Large search spaces, fast approximation | Speed vs Optimality | Temperature schedule, iterations |
| **Integer Programming** | Guaranteed optimal, smaller problems | Optimality vs Time | Solver, time limit |

### 7.2 Objective Functions

| Method | Formula Focus | Use Case |
|--------|---------------|----------|
| `relative_error` | Relative deviation | Balanced general optimization |
| `ratio_difference` | Ratio mismatch | Proportional fairness |
| `weighted_squared` | Squared errors | Penalize large deviations |

### 7.3 Simulated Annealing

**File**: `src/core/swarm_optimizer.py:SimulatedAnnealingOptimizer`

**Parameters**:
```python
initial_temp = 100.0      # Starting temperature
final_temp = 0.01         # Ending temperature
cooling_rate = 0.95       # Temperature decay
max_iterations = 5000     # Max iterations
iterations_per_temp = 100 # Iterations per temperature
```

**When to Use**:
- Fast optimization needed (< 1 second)
- Approximate solution acceptable
- Large problem sizes (100+ instances)

### 7.4 Integer Programming

**File**: `src/core/swarm_optimizer.py:IntegerProgrammingOptimizer`

**Parameters**:
```python
solver_name = "PULP_CBC_CMD"  # Solver backend
time_limit = 300              # Timeout (seconds)
```

**When to Use**:
- Optimal solution required
- Problem size manageable (< 50 instances)
- Execution time not critical

---

## 8. DATA MODELS

### 8.1 Core Models

#### PlannerInput
```python
{
  "M": int,                    # Number of instances (> 0)
  "N": int,                    # Number of model types (> 0)
  "B": List[List[float]],      # Batch capacity matrix [M×N]
  "initial": List[int],        # Initial deployment [M], -1 = no model
  "a": float,                  # Change constraint (0 < a ≤ 1)
  "target": List[float],       # Target request distribution [N]
  "algorithm": str,            # "simulated_annealing" or "integer_programming"
  "objective_method": str,     # Objective function method
  "verbose": bool              # Enable logging
}
```

#### PlannerOutput
```python
{
  "deployment": List[int],     # Optimized assignment [M]
  "score": float,              # Objective value
  "stats": dict,               # Algorithm statistics
  "service_capacity": List[float],  # Capacity per model [N]
  "changes_count": int         # Changes from initial state
}
```

#### InstanceInfo
```python
{
  "endpoint": str,             # Instance API endpoint
  "current_model": str         # Current model name
}
```

### 8.2 PyLet Models

#### PyLetDeploymentInput
```python
{
  "target_state": Dict[str, int],      # Model ID -> count mapping
  "wait_for_ready": bool,              # Wait for instances to be ready
  "register_with_scheduler": bool      # Register with scheduler after deploy
}
```

#### PyLetScaleInput
```python
{
  "model_id": str,                     # Model to scale
  "target_count": int,                 # Target instance count
  "wait_for_ready": bool               # Wait for instances to be ready
}
```

#### PyLetMigrateInput
```python
{
  "pylet_id": str,                     # PyLet instance to migrate
  "target_model_id": str               # Target model ID
}
```

---

## 9. DEVELOPMENT & TESTING

### 9.1 Project Structure
```
planner/
├── src/
│   ├── api.py                      # FastAPI application & endpoints
│   ├── cli.py                      # Typer CLI commands
│   ├── logging_config.py           # Logging configuration
│   ├── core/
│   │   ├── swarm_optimizer.py      # Optimization algorithms
│   │   └── base_optimizer.py       # Abstract optimizer interface
│   ├── models/
│   │   ├── base.py                 # Base models
│   │   ├── instance.py             # Instance models
│   │   ├── planner.py              # Planner I/O models
│   │   ├── pylet.py                # PyLet models
│   │   └── scheduler_compat.py     # Scheduler compatibility models
│   └── pylet/
│       ├── client.py               # PyLet HTTP client
│       └── service.py              # PyLet service layer
├── tests/
│   ├── conftest.py                 # Test fixtures
│   ├── test_api.py                 # API endpoint tests
│   ├── test_models.py              # Model validation tests
│   ├── test_e2e_api_behavior.py    # E2E behavior tests
│   ├── test_optimizers.py          # Algorithm tests
│   └── test_pylet/                 # PyLet integration tests
├── docs/
│   └── 1.API_REFERENCE.md          # API documentation
├── pyproject.toml                  # Project metadata & dependencies
└── uv.lock                         # Locked dependency versions
```

### 9.2 Running Tests
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest --cov=src

# Run PyLet tests only
uv run pytest tests/test_pylet/
```

### 9.3 Key Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `api.py` | HTTP endpoints, request validation, response formatting |
| `models/` | Data schemas, validation rules, type definitions |
| `core/swarm_optimizer.py` | Optimization algorithms (SA and IP) |
| `pylet/service.py` | PyLet cluster management operations |
| `pylet/client.py` | HTTP communication with PyLet head |
| `cli.py` | Command-line interface, server startup |

---

## 10. ERROR HANDLING

### 10.1 Common Error Scenarios

| Error | Cause | Resolution |
|-------|-------|------------|
| No feasible solution | Constraints cannot be satisfied | Adjust target or increase instances |
| PyLet not enabled | PYLET_ENABLED=false | Set environment variable |
| PyLet head unreachable | PYLET_HEAD_URL incorrect | Verify URL and service availability |
| Validation error (422) | Invalid request schema | Check request against schemas |
| Optimization timeout | Problem too large | Use simpler algorithm or reduce size |

### 10.2 Error Response Format
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## 11. OPERATIONAL CONSIDERATIONS

### 11.1 Performance Characteristics

| Metric | Value |
|--------|-------|
| Request processing time | 10ms - 30s (depends on algorithm & size) |
| Concurrent requests | Limited by uvicorn workers (default: 1) |
| Memory usage | ~100MB base + O(n×m) for optimization |
| CPU usage | High during optimization, idle otherwise |

### 11.2 Scaling Recommendations

- **Horizontal scaling**: Run multiple planner instances behind load balancer
- **Vertical scaling**: Increase CPU cores for faster integer programming
- **Optimization**: Use simulated annealing for large problems (>50 instances)

### 11.3 Security Considerations

**IMPORTANT**: This service has NO authentication or authorization.

Recommendations for production:
- Deploy behind API gateway with auth
- Use HTTPS for PyLet communication
- Implement rate limiting
- Add request size limits

---

## 12. INTEGRATION EXAMPLES

### 12.1 Python Client - Planning Only
```python
import httpx

async def compute_plan():
    async with httpx.AsyncClient() as client:
        request = {
            "M": 4, "N": 3,
            "B": [[10, 5, 0], [8, 6, 4], [0, 10, 8], [6, 0, 12]],
            "initial": [0, 1, 2, 2],
            "a": 0.5,
            "target": [20, 30, 25],
            "algorithm": "simulated_annealing"
        }

        response = await client.post(
            "http://localhost:8000/plan",
            json=request
        )

        result = response.json()
        print(f"Deployment: {result['deployment']}")
        print(f"Score: {result['score']}")
        return result
```

### 12.2 Python Client - PyLet Deployment
```python
import httpx

async def deploy_via_pylet():
    async with httpx.AsyncClient() as client:
        # Deploy specific model counts
        deploy_request = {
            "target_state": {"model-a": 2, "model-b": 1},
            "wait_for_ready": True,
            "register_with_scheduler": True
        }

        response = await client.post(
            "http://localhost:8000/pylet/deploy",
            json=deploy_request
        )

        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Added: {result['added_count']}")
        return result
```

### 12.3 cURL Examples
```bash
# Compute deployment plan
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "M": 4, "N": 3,
    "B": [[10, 5, 0], [8, 6, 4], [0, 10, 8], [6, 0, 12]],
    "initial": [0, 1, 2, 2],
    "a": 0.5,
    "target": [20, 30, 25]
  }'

# Deploy via PyLet
curl -X POST http://localhost:8000/pylet/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "target_state": {"model-a": 2, "model-b": 1},
    "wait_for_ready": true
  }'

# Scale a model
curl -X POST http://localhost:8000/pylet/scale \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model-a",
    "target_count": 3
  }'

# Check PyLet status
curl http://localhost:8000/pylet/status
```

---

## 13. FAQ

### Q1: What's the difference between /plan and /pylet/optimize?
**A**: `/plan` only computes an optimal deployment without executing it. `/pylet/optimize` computes the plan AND deploys it via PyLet.

### Q2: Can I use this service without PyLet?
**A**: Yes, the `/plan` endpoint works standalone. PyLet integration is optional and controlled by `PYLET_ENABLED`.

### Q3: How do I choose between algorithms?
**A**: Use `simulated_annealing` for speed (sub-second results) and `integer_programming` for guaranteed optimal solutions.

### Q4: Does this service persist deployment state?
**A**: No, this service is completely stateless. State is managed by PyLet or the instance services themselves.

---

## 14. VERSION HISTORY

### Version 0.1.0 (Current)
- Two optimization algorithms: Simulated Annealing and Integer Programming
- Three objective methods: relative_error, ratio_difference, weighted_squared
- PyLet integration for cluster management
- REST API endpoints: /health, /info, /plan, /pylet/*
- Comprehensive test coverage

---

**Document Version**: 2.0
**Last Updated**: 2025-01-07
**Target Audience**: LLM agents, automated systems, integration developers
