# Swarm Planner Service - LLM Documentation

## 1. SERVICE OVERVIEW

### 1.1 Purpose
The **Swarm Planner** is an optimization service that computes optimal model deployment strategies across multiple instance servers. It uses mathematical optimization algorithms (Simulated Annealing or Integer Programming) to determine which models should run on which instances to minimize resource usage while satisfying computational requirements.

### 1.2 Technology Stack
```
Framework:          FastAPI (async REST API)
Web Server:         Uvicorn (ASGI server)
CLI:                Typer
Optimization:       PuLP (Integer Programming), NumPy (Simulated Annealing)
HTTP Client:        httpx (async)
Data Validation:    Pydantic v2
Testing:            pytest, pytest-asyncio
Package Manager:    uv
Python Version:     3.13 (minimum 3.11)
```

### 1.3 Architecture
- **Type**: Standalone HTTP microservice
- **State**: Stateless (no database, no persistent storage)
- **Communication**: HTTP/REST with instance services
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

# Start in background
uv run splanner start &
```

### 2.3 Health Check
```bash
curl http://localhost:8000/health
# Response: {"status": "healthy"}
```

---

## 3. ENVIRONMENT & CONFIGURATION

### 3.1 Environment Variables
The service supports optional environment variables for configuration:

```bash
# Scheduler Configuration
SCHEDULER_URL=http://localhost:8100    # Default scheduler URL for instance registration

# Instance Deployment Configuration
INSTANCE_TIMEOUT=30                     # HTTP request timeout (seconds)
INSTANCE_MAX_RETRIES=3                  # Max retry attempts for failed requests
INSTANCE_RETRY_DELAY=1.0                # Initial retry delay (exponential backoff)

# Planner Service Configuration
PLANNER_HOST=0.0.0.0                    # Host to bind to
PLANNER_PORT=8000                       # Port to bind to
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

### 3.3 Default Settings
```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "log_level": "info",
  "optimization_timeout": 300,
  "instance_request_timeout": 10.0
}
```

---

## 4. API ENDPOINTS REFERENCE

### 4.1 GET /health

**Purpose**: Health check endpoint to verify service availability.

**Request**: None

**Response Schema**:
```json
{
  "status": "healthy"
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

**Response**: `200 OK`
```json
{
  "status": "healthy"
}
```

---

### 4.2 GET /info

**Purpose**: Returns service metadata including version and available optimization algorithms.

**Request**: None

**Response Schema**:
```json
{
  "service": "string",           // Service name
  "version": "string",           // Semantic version (e.g., "0.1.0")
  "optimization_methods": [      // List of available algorithms
    "string"
  ]
}
```

**Example**:
```bash
curl http://localhost:8000/info
```

**Response**: `200 OK`
```json
{
  "service": "swarm-planner",
  "version": "0.1.0",
  "optimization_methods": ["simulated_annealing", "integer_programming"]
}
```

---

### 4.3 POST /plan

**Purpose**: Compute optimal deployment plan WITHOUT executing it. Returns the recommended model-to-instance mapping.

**Request Schema**:
```json
{
  "tasks": [                              // REQUIRED: List of computation tasks
    {
      "model_id": "string",               // REQUIRED: Unique model identifier
      "required_vram_gb": "number",       // REQUIRED: GPU memory needed (GB)
      "required_flops_tflops": "number"   // REQUIRED: Compute power needed (TFLOPS)
    }
  ],
  "instances": [                          // REQUIRED: List of available instances
    {
      "instance_id": "string",            // REQUIRED: Unique instance identifier
      "base_url": "string",               // REQUIRED: HTTP URL (e.g., "http://192.168.1.10:8001")
      "total_vram_gb": "number",          // REQUIRED: Total GPU memory (GB)
      "total_flops_tflops": "number"      // REQUIRED: Total compute power (TFLOPS)
    }
  ],
  "optimization_method": "string"         // OPTIONAL: "simulated_annealing" or "integer_programming"
                                          // Default: "simulated_annealing"
}
```

**Response Schema**:
```json
{
  "deployment": {                         // Computed optimal mapping
    "model_id_1": "instance_id_A",        // Maps each model to an instance
    "model_id_2": "instance_id_A",
    "model_id_3": "instance_id_B"
  },
  "method_used": "string",                // Algorithm that was used
  "total_instances_used": "integer",      // Number of instances required
  "executed": false                       // Always false for /plan endpoint
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {
        "model_id": "llama-7b",
        "required_vram_gb": 14.0,
        "required_flops_tflops": 50.0
      },
      {
        "model_id": "gpt-neo-2.7b",
        "required_vram_gb": 6.0,
        "required_flops_tflops": 20.0
      }
    ],
    "instances": [
      {
        "instance_id": "gpu-server-1",
        "base_url": "http://192.168.1.10:8001",
        "total_vram_gb": 24.0,
        "total_flops_tflops": 80.0
      },
      {
        "instance_id": "gpu-server-2",
        "base_url": "http://192.168.1.11:8001",
        "total_vram_gb": 16.0,
        "total_flops_tflops": 60.0
      }
    ],
    "optimization_method": "integer_programming"
  }'
```

**Response**: `200 OK`
```json
{
  "deployment": {
    "llama-7b": "gpu-server-1",
    "gpt-neo-2.7b": "gpu-server-1"
  },
  "method_used": "integer_programming",
  "total_instances_used": 1,
  "executed": false
}
```

**Error Responses**:
- `422 Unprocessable Entity`: Invalid request schema
- `500 Internal Server Error`: Optimization failed or no feasible solution

---

### 4.4 POST /deploy

**Purpose**: Compute optimal deployment plan AND execute it by communicating with instance services to start/stop models.

**Request Schema**: Similar to `/plan` endpoint with additional scheduler configuration
```json
{
  "instances": [                          // REQUIRED: List of instance information
    {
      "endpoint": "string",               // REQUIRED: Instance endpoint URL
      "current_model": "string"           // REQUIRED: Currently deployed model name
    }
  ],
  "planner_input": {                      // REQUIRED: Optimization parameters
    "M": "integer",                       // REQUIRED: Number of instances
    "N": "integer",                       // REQUIRED: Number of model types
    "B": [[]],                            // REQUIRED: Benefit matrix
    "initial": [],                        // REQUIRED: Initial deployment (auto-computed from instances)
    "a": "number",                        // REQUIRED: Service capacity weight
    "target": [],                         // REQUIRED: Target service capacity
    "algorithm": "string",                // OPTIONAL: "simulated_annealing" or "integer_programming"
    "objective_method": "string"          // OPTIONAL: Objective function method
  },
  "scheduler_url": "string"               // OPTIONAL: Scheduler URL for instance registration
                                          // Overrides SCHEDULER_URL environment variable if provided
}
```

**Response Schema**:
```json
{
  "deployment": {                         // Computed optimal mapping
    "model_id_1": "instance_id_A",
    "model_id_2": "instance_id_B"
  },
  "method_used": "string",                // Algorithm used
  "total_instances_used": "integer",      // Number of instances used
  "executed": true,                       // Always true for /deploy endpoint
  "execution_details": {                  // Detailed execution results
    "instance_id_A": {
      "stopped_models": ["old-model-1"],  // Models stopped on this instance
      "started_models": ["model_id_1"],   // Models started on this instance
      "errors": []                        // Any errors encountered
    },
    "instance_id_B": {
      "stopped_models": [],
      "started_models": ["model_id_2"],
      "errors": []
    }
  }
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {
        "model_id": "llama-7b",
        "required_vram_gb": 14.0,
        "required_flops_tflops": 50.0
      }
    ],
    "instances": [
      {
        "instance_id": "gpu-server-1",
        "base_url": "http://192.168.1.10:8001",
        "total_vram_gb": 24.0,
        "total_flops_tflops": 80.0
      }
    ],
    "optimization_method": "simulated_annealing"
  }'
```

**Response**: `200 OK`
```json
{
  "deployment": {
    "llama-7b": "gpu-server-1"
  },
  "method_used": "simulated_annealing",
  "total_instances_used": 1,
  "executed": true,
  "execution_details": {
    "gpu-server-1": {
      "stopped_models": ["old-model-xyz"],
      "started_models": ["llama-7b"],
      "errors": []
    }
  }
}
```

**Error Responses**:
- `422 Unprocessable Entity`: Invalid request schema
- `500 Internal Server Error`: Optimization or deployment execution failed

---

## 5. EXTERNAL COMPONENT INTERACTIONS

### 5.1 Instance Service Communication

The planner communicates with instance services via HTTP to:
1. **Query current state** (which models are running)
2. **Stop running models** (free up resources)
3. **Start new models** (deploy according to plan)

### 5.2 Instance Service API Contract

Instance services MUST implement the following endpoints:

#### 5.2.1 GET /info
**Purpose**: Query current instance state

**Response Schema**:
```json
{
  "instance_id": "string",
  "running_models": [
    {
      "model_id": "string",
      "vram_usage_gb": "number",
      "flops_usage_tflops": "number"
    }
  ],
  "available_vram_gb": "number",
  "available_flops_tflops": "number"
}
```

**Example**:
```bash
# Planner calls this to check instance state
GET http://192.168.1.10:8001/info
```

#### 5.2.2 GET /model/stop
**Purpose**: Stop the currently running model

**Request**: None (no request body)

**Response Schema**:
```json
{
  "success": "boolean",
  "message": "string",
  "model_id": "string"
}
```

**Example**:
```bash
# Planner calls this to stop the current model
GET http://192.168.1.10:8001/model/stop
```

**Note**: This endpoint stops whatever model is currently running on the instance. The planner queries `/info` first to determine which model is running.

#### 5.2.3 POST /model/start
**Purpose**: Start a new model

**Request Schema**:
```json
{
  "model_id": "string",
  "parameters": {},                     // Optional model-specific parameters
  "scheduler_url": "string"             // REQUIRED: Scheduler URL for instance registration
}
```

**Response Schema**:
```json
{
  "success": "boolean",
  "message": "string",
  "model_id": "string",
  "status": "string"
}
```

**Example**:
```bash
# Planner calls this to start a model
POST http://192.168.1.10:8001/model/start
Content-Type: application/json

{
  "model_id": "llama-7b",
  "parameters": {},
  "scheduler_url": "http://scheduler:8100"
}
```

**Note**: The `scheduler_url` parameter is required and tells the instance where to register itself after starting the model.

#### 5.2.4 POST /model/restart
**Purpose**: Gracefully restart to a new model (drain tasks, then switch)

**Request Schema**:
```json
{
  "model_id": "string",
  "parameters": {},                     // Optional model-specific parameters
  "scheduler_url": "string"             // OPTIONAL: Scheduler URL for instance registration
}
```

**Response Schema**:
```json
{
  "success": "boolean",
  "message": "string",
  "operation_id": "string"              // ID to track restart progress
}
```

**Example**:
```bash
# Initiate graceful restart
POST http://192.168.1.10:8001/model/restart
Content-Type: application/json

{
  "model_id": "llama-7b",
  "parameters": {},
  "scheduler_url": "http://scheduler:8100"
}
```

**Note**: This endpoint performs a graceful restart that:
1. Drains the current scheduler (stops accepting new tasks)
2. Waits for pending tasks to complete
3. Stops the current model
4. Starts the new model
5. Registers with the scheduler

#### 5.2.5 GET /model/restart/status
**Purpose**: Check status of a restart operation

**Request**: Query parameter `operation_id`

**Response Schema**:
```json
{
  "operation_id": "string",
  "status": "string",                   // "pending", "in_progress", "completed", "failed"
  "current_phase": "string",            // "draining", "stopping", "starting", "registering"
  "message": "string"
}
```

**Example**:
```bash
# Check restart status
GET http://192.168.1.10:8001/model/restart/status?operation_id=abc123
```

### 5.3 Communication Flow Diagram

```
┌─────────────┐                    ┌──────────────────┐
│   Client    │                    │ Swarm Planner    │
│             │                    │  (this service)  │
└──────┬──────┘                    └────────┬─────────┘
       │                                    │
       │  POST /deploy                      │
       │  (tasks + instances)               │
       ├───────────────────────────────────>│
       │                                    │
       │                                    │ 1. Optimize
       │                                    │    (compute mapping)
       │                                    │
       │                              ┌─────┴─────┐
       │                              │           │
       │                              │  GET /info (each instance)
       │                              │  - Query current state
       │                              │
       │                              │  POST /model/stop (as needed)
       │                              │  - Free resources
       │                              │
       │                              │  POST /model/start (new models)
       │                              │  - Deploy models
       │                              │
       │                              └─────┬─────┘
       │                                    │
       │  Deployment Result                 │
       │  (mapping + execution details)     │
       │<───────────────────────────────────┤
       │                                    │
```

### 5.4 Retry Logic and Error Handling

The planner includes robust retry logic for transient failures:

**Retry Configuration**:
- Max retries: 3 (configurable via `INSTANCE_MAX_RETRIES`)
- Initial delay: 1.0 seconds (configurable via `INSTANCE_RETRY_DELAY`)
- Backoff strategy: Exponential (delay = initial × 2^attempt)

**Retryable Errors**:
- Connection errors (httpx.ConnectError)
- Timeout errors (httpx.TimeoutException)
- HTTP 5xx errors (except 501 Not Implemented)

**Non-Retryable Errors**:
- HTTP 4xx errors (client errors)
- HTTP 501 Not Implemented
- Validation errors

**Error Response Extraction**:
The planner attempts to extract detailed error messages from instance responses:
1. Try to parse JSON error response
2. Extract "error" or "detail" field
3. Fall back to response text
4. Fall back to HTTP status code

**Example Retry Flow**:
```
Attempt 1: Connection refused → Wait 1s → Retry
Attempt 2: Timeout → Wait 2s → Retry
Attempt 3: 503 Service Unavailable → Wait 4s → Retry
Attempt 4: Fail (max retries exceeded)
```

### 5.5 No External Dependencies

This service does NOT require:
- Database (stateless design)
- Message queue
- Cache server (Redis, Memcached)
- Service mesh / discovery
- Authentication service (no auth implemented)

---

## 6. CORE ALGORITHMS

### 6.1 Algorithm Selection

Two optimization algorithms are available:

| Algorithm               | Use Case                          | Speed      | Quality    |
|-------------------------|-----------------------------------|------------|------------|
| `simulated_annealing`   | General purpose, quick results    | Fast       | Good       |
| `integer_programming`   | Optimal solution guaranteed       | Slower     | Optimal    |

### 6.2 Simulated Annealing

**File**: `src/core/swarm_optimizer.py:SimulatedAnnealingOptimizer`

**Algorithm Overview**:
1. Start with random model-to-instance assignment
2. Iteratively swap assignments to reduce total instances used
3. Accept worse solutions probabilistically (temperature-based)
4. Cool down temperature over iterations
5. Return best solution found

**Parameters**:
```python
max_iterations = 10000
initial_temperature = 100.0
cooling_rate = 0.95
min_temperature = 0.01
```

**Time Complexity**: O(n × m × k) where:
- n = number of models
- m = number of instances
- k = number of iterations

**When to Use**:
- Fast optimization needed (< 1 second)
- Approximate solution acceptable
- Large problem sizes (100+ models)

### 6.3 Integer Programming

**File**: `src/core/swarm_optimizer.py:IntegerProgrammingOptimizer`

**Algorithm Overview**:
1. Formulate as Mixed Integer Linear Programming (MILP) problem
2. Binary variables: x[model][instance] ∈ {0, 1}
3. Objective: Minimize number of instances used
4. Constraints:
   - Each model assigned to exactly one instance
   - VRAM capacity not exceeded per instance
   - FLOPS capacity not exceeded per instance
5. Solve using PuLP (CBC solver backend)

**Formulation**:
```
Minimize: Σ y[i]  (sum of instance usage indicators)

Subject to:
  - Σ x[m][i] = 1  for all models m  (each model assigned once)
  - Σ (x[m][i] × vram[m]) ≤ capacity_vram[i] × y[i]  (VRAM constraint)
  - Σ (x[m][i] × flops[m]) ≤ capacity_flops[i] × y[i]  (FLOPS constraint)
  - x[m][i] ∈ {0, 1}  (binary assignment)
  - y[i] ∈ {0, 1}  (instance used indicator)
```

**Time Complexity**: O(2^(n×m)) worst case, but typically much faster with branch-and-bound

**When to Use**:
- Optimal solution required
- Problem size manageable (< 50 models)
- Execution time not critical (may take 10+ seconds)

---

## 7. DATA MODELS

### 7.1 Request Models

#### ComputationalTask
```python
{
  "model_id": str,                    # Unique identifier for the model
                                      # Example: "llama-7b", "gpt-neo-2.7b"

  "required_vram_gb": float,          # GPU memory required in gigabytes
                                      # Must be > 0
                                      # Example: 14.0, 6.5, 24.0

  "required_flops_tflops": float      # Compute power required in teraflops
                                      # Must be > 0
                                      # Example: 50.0, 20.0, 100.0
}
```

#### InstanceInfo
```python
{
  "instance_id": str,                 # Unique identifier for the instance
                                      # Example: "gpu-server-1", "node-a"

  "base_url": str,                    # HTTP base URL for instance API
                                      # Must be valid HTTP/HTTPS URL
                                      # Example: "http://192.168.1.10:8001"

  "total_vram_gb": float,             # Total GPU memory available (GB)
                                      # Must be > 0
                                      # Example: 24.0, 80.0

  "total_flops_tflops": float         # Total compute power available (TFLOPS)
                                      # Must be > 0
                                      # Example: 80.0, 150.0
}
```

#### OptimizationRequest
```python
{
  "tasks": List[ComputationalTask],   # REQUIRED: List of models to deploy
                                      # Must have at least 1 task

  "instances": List[InstanceInfo],    # REQUIRED: List of available instances
                                      # Must have at least 1 instance

  "optimization_method": str          # OPTIONAL: Algorithm selection
                                      # Default: "simulated_annealing"
                                      # Options: "simulated_annealing", "integer_programming"
}
```

### 7.2 Response Models

#### DeploymentPlan
```python
{
  "deployment": Dict[str, str],       # Maps model_id -> instance_id
                                      # Example: {"llama-7b": "gpu-server-1"}

  "method_used": str,                 # Algorithm that was used
                                      # Example: "simulated_annealing"

  "total_instances_used": int,        # Count of instances in the solution
                                      # Example: 3

  "executed": bool                    # Whether deployment was executed
                                      # False for /plan, True for /deploy
}
```

#### DeploymentResult (extends DeploymentPlan)
```python
{
  # All fields from DeploymentPlan, plus:

  "execution_details": Dict[str, InstanceExecutionDetail]
                                      # Maps instance_id -> execution results
                                      # Only present when executed=True
}
```

#### InstanceExecutionDetail
```python
{
  "stopped_models": List[str],        # Model IDs that were stopped
                                      # Example: ["old-model-1", "old-model-2"]

  "started_models": List[str],        # Model IDs that were started
                                      # Example: ["llama-7b"]

  "errors": List[str]                 # Any error messages encountered
                                      # Example: ["Failed to stop model xyz"]
}
```

#### HealthResponse
```python
{
  "status": str                       # Always "healthy" if service is up
}
```

#### InfoResponse
```python
{
  "service": str,                     # Service name: "swarm-planner"
  "version": str,                     # Semantic version: "0.1.0"
  "optimization_methods": List[str]   # ["simulated_annealing", "integer_programming"]
}
```

---

## 8. DEVELOPMENT & TESTING

### 8.1 Project Structure
```
planner/
├── src/
│   ├── api.py                      # FastAPI application & endpoints
│   ├── models.py                   # Pydantic data models
│   ├── deployment_service.py       # Deployment orchestration
│   ├── cli.py                      # Typer CLI commands
│   └── core/
│       ├── swarm_optimizer.py      # Optimization algorithms
│       └── base_optimizer.py       # Abstract optimizer interface
├── tests/
│   ├── test_api.py                 # API endpoint tests
│   ├── test_deployment_service.py  # Service logic tests
│   └── test_optimizers.py          # Algorithm tests
├── pyproject.toml                  # Project metadata & dependencies
├── uv.lock                         # Locked dependency versions
└── README.md                       # Human-readable documentation
```

### 8.2 Running Tests
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest --cov=src

# Run in quiet mode (minimal output)
uv run pytest -q
```

### 8.3 Key Module Responsibilities

| Module                     | Responsibility                                          |
|----------------------------|---------------------------------------------------------|
| `api.py`                   | HTTP endpoints, request validation, response formatting |
| `models.py`                | Data schemas, validation rules, type definitions        |
| `deployment_service.py`    | Orchestrates instance communication & deployment        |
| `cli.py`                   | Command-line interface, server startup                  |
| `swarm_optimizer.py`       | Optimization algorithms (SA and IP)                     |
| `base_optimizer.py`        | Abstract base class for optimizers                      |

### 8.4 Adding New Optimization Algorithm

1. Create new class inheriting from `BaseOptimizer`:
```python
# src/core/my_optimizer.py
from .base_optimizer import BaseOptimizer

class MyOptimizer(BaseOptimizer):
    def optimize(
        self,
        tasks: List[ComputationalTask],
        instances: List[InstanceInfo]
    ) -> Dict[str, str]:
        # Implement your algorithm
        return deployment_mapping
```

2. Register in `swarm_optimizer.py`:
```python
OPTIMIZERS = {
    "simulated_annealing": SimulatedAnnealingOptimizer,
    "integer_programming": IntegerProgrammingOptimizer,
    "my_algorithm": MyOptimizer,  # Add here
}
```

3. Add tests in `tests/test_optimizers.py`

---

## 9. ERROR HANDLING

### 9.1 Common Error Scenarios

| Error                               | Cause                                      | Resolution                              |
|-------------------------------------|--------------------------------------------|-----------------------------------------|
| No feasible solution                | Tasks exceed total instance capacity       | Add more instances or reduce task sizes |
| Instance unreachable                | base_url incorrect or service down         | Verify instance URL and availability    |
| Model start/stop failed             | Instance API error                         | Check instance logs and API contract    |
| 422 Unprocessable Entity            | Invalid request schema                     | Validate request against schemas        |
| Optimization timeout                | Problem too large for algorithm            | Use simpler algorithm or reduce problem |

### 9.2 Error Response Format
```json
{
  "detail": "Error message describing what went wrong"
}
```

**Example**:
```json
{
  "detail": "No feasible deployment found: total required VRAM (100 GB) exceeds total available (80 GB)"
}
```

---

## 10. OPERATIONAL CONSIDERATIONS

### 10.1 Performance Characteristics

| Metric                    | Value                                    |
|---------------------------|------------------------------------------|
| Request processing time   | 10ms - 30s (depends on algorithm & size) |
| Concurrent requests       | Limited by uvicorn workers (default: 1)  |
| Memory usage              | ~100MB base + O(n×m) for optimization    |
| CPU usage                 | High during optimization, idle otherwise |

### 10.2 Scaling Recommendations

- **Horizontal scaling**: Run multiple planner instances behind load balancer
- **Vertical scaling**: Increase CPU cores for faster integer programming
- **Optimization**: Use simulated annealing for large problems (>50 models)

### 10.3 Monitoring

Key metrics to monitor:
- `/health` endpoint response time
- Optimization request duration (from logs)
- Instance communication failures (from execution_details.errors)
- HTTP 500 error rate

### 10.4 Security Considerations

**IMPORTANT**: This service has NO authentication or authorization.

Recommendations for production:
- Deploy behind API gateway with auth
- Use HTTPS for instance communication
- Implement rate limiting
- Validate instance URLs (prevent SSRF)
- Add request size limits

---

## 11. TROUBLESHOOTING

### 11.1 Service Won't Start

**Symptom**: `splanner start` fails

**Checks**:
```bash
# 1. Verify uv installation
uv --version

# 2. Verify dependencies
uv sync

# 3. Check port availability
lsof -i :8000

# 4. Try with explicit host/port
uv run splanner start --host 127.0.0.1 --port 9000
```

### 11.2 Optimization Fails

**Symptom**: POST /plan or /deploy returns 500 error

**Checks**:
1. Verify request schema is correct
2. Check that total capacity ≥ total requirements
3. Try different optimization method
4. Check logs for detailed error messages

### 11.3 Deployment Execution Fails

**Symptom**: /deploy returns success but execution_details shows errors

**Checks**:
1. Verify instance base_url is accessible
2. Test instance endpoints manually:
   ```bash
   curl http://<instance-url>/info
   ```
3. Check instance service logs
4. Verify instance API contract compliance

---

## 12. INTEGRATION EXAMPLES

### 12.1 Python Client Example
```python
import httpx

async def deploy_models():
    async with httpx.AsyncClient() as client:
        # Prepare request
        request = {
            "tasks": [
                {
                    "model_id": "llama-7b",
                    "required_vram_gb": 14.0,
                    "required_flops_tflops": 50.0
                }
            ],
            "instances": [
                {
                    "instance_id": "gpu-1",
                    "base_url": "http://192.168.1.10:8001",
                    "total_vram_gb": 24.0,
                    "total_flops_tflops": 80.0
                }
            ],
            "optimization_method": "integer_programming"
        }

        # Call planner
        response = await client.post(
            "http://localhost:8000/deploy",
            json=request
        )

        result = response.json()
        print(f"Deployment: {result['deployment']}")
        print(f"Instances used: {result['total_instances_used']}")

        return result
```

### 12.2 cURL Example
```bash
# Plan deployment (dry-run)
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {"model_id": "model-a", "required_vram_gb": 10, "required_flops_tflops": 30},
      {"model_id": "model-b", "required_vram_gb": 8, "required_flops_tflops": 25}
    ],
    "instances": [
      {"instance_id": "gpu-1", "base_url": "http://192.168.1.10:8001", "total_vram_gb": 24, "total_flops_tflops": 80}
    ]
  }'

# Execute deployment
curl -X POST http://localhost:8000/deploy \
  -H "Content-Type: application/json" \
  -d @deployment_request.json
```

### 12.3 JavaScript/TypeScript Example
```typescript
interface DeploymentRequest {
  tasks: {
    model_id: string;
    required_vram_gb: number;
    required_flops_tflops: number;
  }[];
  instances: {
    instance_id: string;
    base_url: string;
    total_vram_gb: number;
    total_flops_tflops: number;
  }[];
  optimization_method?: "simulated_annealing" | "integer_programming";
}

async function deployModels(request: DeploymentRequest) {
  const response = await fetch("http://localhost:8000/deploy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  const result = await response.json();
  console.log("Deployment result:", result);
  return result;
}
```

---

## 13. CHANGELOG & VERSION HISTORY

### Version 0.1.0 (Current)
- Initial release
- Two optimization algorithms: Simulated Annealing and Integer Programming
- Four REST API endpoints: /health, /info, /plan, /deploy
- Async HTTP communication with instance services
- Comprehensive test coverage
- Zero-configuration design

---

## 14. FREQUENTLY ASKED QUESTIONS (FAQ)

### Q1: Can I use this service without instance services?
**A**: Yes, the `/plan` endpoint works standalone and doesn't communicate with instances. It only computes the optimal mapping. The `/deploy` endpoint requires instance services to execute the deployment.

### Q2: What happens if an instance is unreachable during deployment?
**A**: The deployment will attempt to continue with other instances. Errors will be recorded in `execution_details[instance_id].errors`. The overall request will still return 200 OK with partial results.

### Q3: Can I add custom optimization algorithms?
**A**: Yes, see section 8.4 "Adding New Optimization Algorithm" for instructions.

### Q4: Does this service persist deployment state?
**A**: No, this service is completely stateless. It doesn't remember previous deployments. State is managed by the instance services themselves.

### Q5: How do I choose between simulated_annealing and integer_programming?
**A**: Use `simulated_annealing` for speed (sub-second results) and `integer_programming` for guaranteed optimal solutions (may take 10+ seconds for large problems).

### Q6: Can I run multiple planner instances?
**A**: Yes, the service is stateless so you can run multiple instances behind a load balancer for high availability.

### Q7: What if no feasible solution exists?
**A**: The API will return a 500 error with details about why no solution could be found (typically insufficient total resources).

---

## 15. CONTACT & SUPPORT

- **Repository**: `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/planner`
- **Documentation**: This file (`README_FOR_LLM.md`)
- **Tests**: Run `uv run pytest` to verify functionality
- **Version**: 0.1.0

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Target Audience**: LLM agents, automated systems, integration developers
