# SwarmPilot Instance - LLM Documentation

## Project Overview

**Project Name:** swarmpilot-instance

**Purpose:** A lightweight execution service for running model containers with task queue management in the SwarmPilot distributed inference ecosystem.

**Role in Architecture:** The Instance service acts as a worker node that:
- Manages the lifecycle of a single model container (Docker-based)
- Maintains a FIFO task queue for sequential task processing
- Provides RESTful APIs for model and task management
- Optionally integrates with a central Scheduler service for distributed workload coordination

**Technology Stack:**
- Python 3.11+
- FastAPI (async web framework)
- Docker (container runtime)
- uv (dependency manager)
- Typer (CLI framework)

---

## Quick Start

### Installation

```bash
# Install dependencies using uv
uv sync

# Verify installation
uv run sinstance --help
```

### Running the Service

```bash
# Method 1: Using CLI command
uv run sinstance start

# Method 2: With custom configuration
sinstance start --port 5000 --reload

# Method 3: Direct Python execution
uv run python -m src.cli start
```

### Basic Usage Flow

```bash
# 1. Start a model container
curl -X POST http://localhost:5000/model/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sleep_model",
    "parameters": {}
  }'

# 2. Submit a task
curl -X POST http://localhost:5000/task/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-001",
    "model_id": "sleep_model",
    "task_input": {"sleep_time": 3}
  }'

# 3. Check task status
curl http://localhost:5000/task/task-001

# 4. List all tasks
curl http://localhost:5000/task/list

# 5. Get instance info
curl http://localhost:5000/info
```

---

## Environment Variables

### Required Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `INSTANCE_ID` | `"instance-default"` | No | Unique identifier for this instance |
| `INSTANCE_PORT` | `5000` | No | Port for the FastAPI server |
| `LOG_LEVEL` | `"INFO"` | No | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `MAX_QUEUE_SIZE` | `100` | No | Maximum number of tasks in queue |
| `DOCKER_NETWORK` | `"instance_network"` | No | Docker network for model containers |
| `HEALTH_CHECK_INTERVAL` | `10` | No | Seconds between model health checks |
| `HEALTH_CHECK_TIMEOUT` | `30` | No | Seconds before health check times out |

### Optional Scheduler Integration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `SCHEDULER_URL` | `None` | No | URL of the Scheduler service (e.g., `http://scheduler:8000`) |
| `INSTANCE_ENDPOINT` | `None` | No | Public endpoint for this instance (for scheduler registration) |

### Configuration Example

```bash
# .env file
INSTANCE_ID=worker-001
INSTANCE_PORT=5000
LOG_LEVEL=DEBUG
MAX_QUEUE_SIZE=50
SCHEDULER_URL=http://scheduler:8000
INSTANCE_ENDPOINT=http://192.168.1.10:5000
```

---

## API Endpoints Reference

### Base URL
`http://<host>:<INSTANCE_PORT>`

---

### Model Management

#### POST /model/start

**Purpose:** Start a new model container.

**Request:**
```json
{
  "model_id": "string",         // Required: ID from model_registry.yaml
  "parameters": {               // Optional: Model-specific parameters
    "key": "value"
  }
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "message": "Model sleep_model started successfully",
  "model_info": {
    "model_id": "sleep_model",
    "container_id": "abc123...",
    "container_name": "model_sleep_model_5000",
    "port": 6000,
    "health_status": "healthy"
  }
}
```

**Response (Error - 400/500):**
```json
{
  "detail": "Error message describing the failure"
}
```

**Functionality:**
1. Validates model_id exists in model registry
2. Stops any existing model container
3. Builds Docker image if not present
4. Starts container with port mapping (INSTANCE_PORT + 1000)
5. Performs health checks until model is ready
6. Registers with scheduler (if configured)

---

#### GET /model/stop

**Purpose:** Stop the currently running model container.

**Request:** No body required

**Response (Success - 200):**
```json
{
  "status": "success",
  "message": "Model stopped successfully"
}
```

**Response (No Model - 404):**
```json
{
  "detail": "No model is currently running"
}
```

**Functionality:**
1. Stops and removes the model container
2. Clears current model state
3. Deregisters from scheduler (if configured)

---

### Task Management

#### POST /task/submit

**Purpose:** Submit a new task to the queue for processing.

**Request:**
```json
{
  "task_id": "string",          // Required: Unique task identifier
  "model_id": "string",         // Required: Must match current model
  "task_input": {               // Required: Model-specific input data
    "key": "value"
  },
  "callback_url": "string"      // Optional: URL for result callback
}
```

**Response (Success - 200):**
```json
{
  "task_id": "task-001",
  "model_id": "sleep_model",
  "task_input": {"sleep_time": 3},
  "status": "pending",
  "queue_position": 5,
  "submitted_at": "2025-11-02T10:30:00.123456",
  "started_at": null,
  "completed_at": null,
  "result": null,
  "error": null,
  "callback_url": null
}
```

**Response (Error - 400):**
```json
{
  "detail": "No model is currently running"
}
// OR
{
  "detail": "Task task-001 already exists"
}
// OR
{
  "detail": "Task queue is full (max size: 100)"
}
```

**Functionality:**
1. Validates model is running
2. Checks task_id uniqueness
3. Checks queue capacity
4. Adds task to FIFO queue
5. Returns task object with status "pending"
6. Task will be processed sequentially

---

#### GET /task/list

**Purpose:** List all tasks with optional filtering.

**Query Parameters:**
- `status` (optional): Filter by status (pending/processing/completed/failed)
- `limit` (optional): Maximum number of tasks to return

**Request Example:**
```
GET /task/list?status=completed&limit=10
```

**Response (Success - 200):**
```json
{
  "total": 42,
  "tasks": [
    {
      "task_id": "task-001",
      "model_id": "sleep_model",
      "status": "completed",
      "queue_position": null,
      "submitted_at": "2025-11-02T10:30:00.123456",
      "started_at": "2025-11-02T10:30:05.234567",
      "completed_at": "2025-11-02T10:30:08.345678",
      "result": {"output": "data"},
      "error": null
    }
  ]
}
```

**Functionality:**
1. Retrieves all tasks from task queue
2. Applies status filter if provided
3. Applies limit if provided
4. Returns task list with metadata

---

#### GET /task/{task_id}

**Purpose:** Get detailed information about a specific task.

**Path Parameters:**
- `task_id`: The unique task identifier

**Request Example:**
```
GET /task/task-001
```

**Response (Success - 200):**
```json
{
  "task_id": "task-001",
  "model_id": "sleep_model",
  "task_input": {"sleep_time": 3},
  "status": "completed",
  "queue_position": null,
  "submitted_at": "2025-11-02T10:30:00.123456",
  "started_at": "2025-11-02T10:30:05.234567",
  "completed_at": "2025-11-02T10:30:08.345678",
  "result": {
    "message": "Slept for 3 seconds",
    "actual_sleep_time": 3.001
  },
  "error": null,
  "callback_url": null
}
```

**Response (Not Found - 404):**
```json
{
  "detail": "Task task-999 not found"
}
```

**Task Status Values:**
- `pending`: Task is in queue waiting
- `processing`: Task is currently being executed
- `completed`: Task finished successfully
- `failed`: Task execution failed

---

#### DELETE /task/{task_id}

**Purpose:** Cancel or remove a task from the queue.

**Path Parameters:**
- `task_id`: The unique task identifier

**Request Example:**
```
DELETE /task/task-001
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "message": "Task task-001 removed successfully"
}
```

**Response (Not Found - 404):**
```json
{
  "detail": "Task task-001 not found"
}
```

**Functionality:**
1. Finds task in queue
2. Removes task if status is "pending"
3. Cannot remove tasks that are "processing"
4. Completed/failed tasks can be removed from history

---

### Instance Management

#### GET /info

**Purpose:** Get instance information and statistics.

**Request:** No parameters

**Response (Success - 200):**
```json
{
  "instance_id": "instance-default",
  "status": "running",
  "current_model": {
    "model_id": "sleep_model",
    "container_id": "abc123...",
    "container_name": "model_sleep_model_5000",
    "port": 6000,
    "health_status": "healthy"
  },
  "queue_stats": {
    "total_tasks": 42,
    "pending_tasks": 5,
    "processing_tasks": 1,
    "completed_tasks": 30,
    "failed_tasks": 6,
    "queue_capacity": 100
  },
  "scheduler_connected": true
}
```

**Response (No Model Running):**
```json
{
  "instance_id": "instance-default",
  "status": "idle",
  "current_model": null,
  "queue_stats": {
    "total_tasks": 0,
    "pending_tasks": 0,
    "processing_tasks": 0,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "queue_capacity": 100
  },
  "scheduler_connected": false
}
```

**Functionality:**
- Returns current instance state
- Provides model container information
- Shows task queue statistics
- Indicates scheduler connection status

---

#### GET /health

**Purpose:** Health check endpoint for monitoring.

**Request:** No parameters

**Response (Success - 200):**
```json
{
  "status": "healthy",
  "instance_id": "instance-default",
  "timestamp": "2025-11-02T10:30:00.123456"
}
```

**Functionality:**
- Simple health check for load balancers/monitoring
- Always returns 200 if service is running
- Does not check model container health (use /info for that)

---

## Component Interactions

### 1. Docker Daemon Integration

**Purpose:** Manage model container lifecycle

**Operations:**
- Build Docker images from model directories
- Start/stop containers with port mapping
- Monitor container health via HTTP endpoints
- Clean up stopped containers

**Container Port Mapping:**
- Model container exposes port 8000 internally
- Mapped to `INSTANCE_PORT + 1000` on host
- Example: Instance on 5000 → Model on 6000

**Network Configuration:**
- Containers run on dedicated Docker network (`DOCKER_NETWORK`)
- Network created automatically if not exists
- Enables container-to-container communication

---

### 2. Scheduler Service Integration (Optional)

**Purpose:** Coordinate with central scheduler for distributed workload

**Registration Flow:**
```
1. Instance starts model → POST to Scheduler /instance/register
   Request: {
     "instance_id": "worker-001",
     "endpoint": "http://192.168.1.10:5000",
     "model_id": "sleep_model",
     "capacity": 100
   }

2. Scheduler assigns instance_id and tracks capacity

3. Instance stops model → POST to Scheduler /instance/deregister
   Request: {
     "instance_id": "worker-001"
   }
```

**Task Result Callbacks:**
```
1. Task completes → Instance sends results to callback_url
   POST to callback_url:
   {
     "task_id": "task-001",
     "status": "completed",
     "result": {...},
     "error": null
   }

2. Scheduler updates task status and notifies clients
```

**Configuration:**
- Set `SCHEDULER_URL` to enable integration
- Set `INSTANCE_ENDPOINT` for scheduler to reach this instance
- Instance will auto-register on model start
- Instance will auto-deregister on model stop

---

### 3. Model Container Communication

**Purpose:** Execute inference tasks on model containers

**Required Model Container Interface:**

All model containers must implement:

#### POST /inference

**Request:**
```json
{
  "task_id": "task-001",
  "input": {
    // Model-specific input data
  }
}
```

**Response:**
```json
{
  "task_id": "task-001",
  "output": {
    // Model-specific output data
  }
}
```

#### GET /health

**Response:**
```json
{
  "status": "healthy"
}
```

**Communication Flow:**
```
1. Task submitted to Instance → Added to queue
2. Queue processor picks task → POST to http://localhost:{MODEL_PORT}/inference
3. Model processes → Returns result
4. Instance updates task status → Stores result
5. If callback_url → POST result to callback
```

---

## Model Container Specification

### Directory Structure

Each model must be in `/dockers/{model_id}/` with:

```
/dockers/
  └── {model_id}/
      ├── Dockerfile          # Container build instructions
      ├── pyproject.toml      # Python dependencies (uv format)
      ├── start.sh            # Container startup script
      └── src/
          └── main.py         # FastAPI application
```

### Dockerfile Requirements

```dockerfile
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy application files
WORKDIR /app
COPY . .

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Start command
CMD ["bash", "start.sh"]
```

### start.sh Requirements

```bash
#!/bin/bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### FastAPI Application (src/main.py)

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/inference")
async def inference(request: dict):
    task_id = request["task_id"]
    task_input = request["input"]

    # Process task
    result = process_inference(task_input)

    return {
        "task_id": task_id,
        "output": result
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Model Registry Configuration

Models must be registered in `/dockers/model_registry.yaml`:

```yaml
models:
  - model_id: "sleep_model"
    description: "Example model that sleeps for specified time"
    version: "1.0.0"
    build_context: "/dockers/sleep_model"
    requirements:
      - "fastapi"
      - "uvicorn"
```

**Registry Fields:**
- `model_id`: Unique identifier used in API requests
- `description`: Human-readable description
- `version`: Model version
- `build_context`: Absolute path to model directory

---

## Internal Architecture

### Core Modules

#### 1. src/api.py (645 lines)
**Purpose:** FastAPI application and route handlers

**Key Components:**
- `app`: FastAPI application instance
- Route handlers for all endpoints
- Startup/shutdown lifecycle management
- CORS configuration

**Dependencies:**
- DockerManager: Container lifecycle
- TaskQueue: Task processing
- ModelRegistry: Model metadata
- SchedulerClient: Scheduler integration
- Config: Configuration management

---

#### 2. src/task_queue.py
**Purpose:** FIFO task queue with async processing

**Key Features:**
- Thread-safe task storage
- Automatic sequential processing
- Status tracking (pending/processing/completed/failed)
- Queue capacity management
- Callback execution on completion

**Task Lifecycle:**
```
1. submit_task() → Task created with status "pending"
2. Background worker picks task → Status changes to "processing"
3. Task sent to model /inference endpoint
4. On success → Status "completed", result stored
5. On failure → Status "failed", error stored
6. If callback_url → POST result to callback
```

---

#### 3. src/docker_manager.py
**Purpose:** Docker container lifecycle management

**Key Operations:**
- `build_image()`: Build Docker image from model directory
- `start_container()`: Start model container with port mapping
- `stop_container()`: Stop and remove container
- `check_health()`: Verify model container health
- `get_container_info()`: Get container status

**Health Check Logic:**
```
1. Wait for container to start
2. Poll GET http://localhost:{port}/health
3. Retry with exponential backoff
4. Timeout after HEALTH_CHECK_TIMEOUT seconds
5. Mark as healthy if 200 response received
```

---

#### 4. src/scheduler_client.py
**Purpose:** Integration with Scheduler service

**Key Operations:**
- `register_instance()`: Register this instance with scheduler
- `deregister_instance()`: Unregister on shutdown
- `send_callback()`: Send task results to callback URLs

**Error Handling:**
- Graceful degradation if scheduler unavailable
- Retry logic for network failures
- Logging for debugging integration issues

---

#### 5. src/model_registry.py
**Purpose:** Load and validate model registry

**Operations:**
- Load `/dockers/model_registry.yaml`
- Validate model configurations
- Provide model lookup by ID

---

#### 6. src/config.py
**Purpose:** Configuration management

**Features:**
- Load environment variables with defaults
- Pydantic-based validation
- Singleton pattern for global access

---

#### 7. src/models.py
**Purpose:** Pydantic data models

**Key Models:**
- `ModelStartRequest`: Model start request schema
- `TaskSubmitRequest`: Task submission schema
- `Task`: Task object with status tracking
- `InstanceInfo`: Instance information response

---

#### 8. src/cli.py
**Purpose:** Command-line interface (Typer)

**Commands:**
```bash
sinstance start [--port PORT] [--reload]
```

**Features:**
- Start FastAPI server with uvicorn
- Hot-reload for development
- Port configuration

---

### Task Processing Workflow

```
User Request
    │
    ▼
POST /task/submit
    │
    ├─► Validate model running
    ├─► Validate task_id unique
    ├─► Check queue capacity
    │
    ▼
TaskQueue.submit_task()
    │
    ├─► Create Task object (status: pending)
    ├─► Add to queue
    ├─► Return task to user
    │
    ▼
Background Worker (async)
    │
    ├─► Pick next pending task
    ├─► Update status: processing
    │
    ▼
POST http://localhost:{MODEL_PORT}/inference
    │
    ├─► Success?
    │   ├─► Yes: status=completed, store result
    │   └─► No: status=failed, store error
    │
    ▼
Callback (if configured)
    │
    └─► POST result to callback_url
```

---

## Development & Testing

### Running in Development Mode

```bash
# Start with hot-reload
uv run sinstance start --reload

# Start on custom port
uv run sinstance start --port 8080

# Set environment variables
export LOG_LEVEL=DEBUG
export MAX_QUEUE_SIZE=50
uv run sinstance start
```

### Testing with Example Model

The repository includes `sleep_model` as a reference implementation:

```bash
# 1. Start the instance
uv run sinstance start

# 2. Start the sleep model
curl -X POST http://localhost:5000/model/start \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sleep_model"}'

# 3. Submit a test task
curl -X POST http://localhost:5000/task/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "model_id": "sleep_model",
    "task_input": {"sleep_time": 2}
  }'

# 4. Monitor task progress
watch -n 1 curl -s http://localhost:5000/task/test-001

# 5. Check instance stats
curl http://localhost:5000/info | jq
```

### Common Testing Scenarios

#### Scenario 1: Queue Processing
```bash
# Submit multiple tasks
for i in {1..5}; do
  curl -X POST http://localhost:5000/task/submit \
    -H "Content-Type: application/json" \
    -d "{\"task_id\": \"task-$i\", \"model_id\": \"sleep_model\", \"task_input\": {\"sleep_time\": 1}}"
done

# Watch queue drain
watch -n 1 'curl -s http://localhost:5000/task/list | jq ".tasks[] | {task_id, status}"'
```

#### Scenario 2: Error Handling
```bash
# Try to submit task without model
curl -X POST http://localhost:5000/task/submit \
  -H "Content-Type: application/json" \
  -d '{"task_id": "fail-001", "model_id": "nonexistent"}'
# Expected: 400 error

# Try to submit duplicate task
curl -X POST http://localhost:5000/task/submit \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task-1", "model_id": "sleep_model", "task_input": {}}'
curl -X POST http://localhost:5000/task/submit \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task-1", "model_id": "sleep_model", "task_input": {}}'
# Expected: Second request returns 400 error
```

#### Scenario 3: Model Switching
```bash
# Stop current model
curl http://localhost:5000/model/stop

# Start different model
curl -X POST http://localhost:5000/model/start \
  -H "Content-Type: application/json" \
  -d '{"model_id": "another_model"}'
```

### Debugging Tips

1. **Check Logs:**
   ```bash
   # Set debug logging
   export LOG_LEVEL=DEBUG
   uv run sinstance start
   ```

2. **Inspect Docker Containers:**
   ```bash
   # List running containers
   docker ps

   # View container logs
   docker logs <container_id>

   # Inspect container
   docker inspect <container_id>
   ```

3. **Test Model Container Directly:**
   ```bash
   # Get model port from /info endpoint
   curl http://localhost:5000/info | jq '.current_model.port'

   # Test model health
   curl http://localhost:6000/health

   # Test model inference
   curl -X POST http://localhost:6000/inference \
     -H "Content-Type: application/json" \
     -d '{"task_id": "direct-test", "input": {"sleep_time": 1}}'
   ```

4. **Monitor Resource Usage:**
   ```bash
   # Docker stats
   docker stats

   # Instance resource usage
   htop
   ```

---

## File Locations Reference

### Core Source Files
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/src/api.py` - FastAPI application
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/src/config.py` - Configuration
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/src/models.py` - Data models
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/src/task_queue.py` - Task queue
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/src/docker_manager.py` - Docker manager
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/src/model_registry.py` - Model registry
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/src/scheduler_client.py` - Scheduler client
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/src/cli.py` - CLI entry point

### Configuration Files
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/pyproject.toml` - Python dependencies
- `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/instance/uv.lock` - Lock file
- `/dockers/model_registry.yaml` - Model registry

### Example Model
- `/dockers/sleep_model/` - Reference model implementation

---

## Additional Notes for LLMs

### When Modifying This Codebase

1. **Adding New API Endpoints:**
   - Add route handler in `src/api.py`
   - Add request/response models in `src/models.py`
   - Update this documentation

2. **Changing Task Processing Logic:**
   - Modify `src/task_queue.py`
   - Ensure thread safety for concurrent access
   - Test with multiple concurrent tasks

3. **Adding Model Container Features:**
   - Update model container specification
   - Modify `src/docker_manager.py` if needed
   - Update example model (`/dockers/sleep_model/`)

4. **Integrating New External Services:**
   - Create client module (similar to `scheduler_client.py`)
   - Add configuration in `src/config.py`
   - Add environment variables section in this doc

### Common Pitfalls

1. **Port Conflicts:**
   - Instance port and model port must not conflict
   - Model port = INSTANCE_PORT + 1000
   - Ensure ports are available before starting

2. **Docker Network Issues:**
   - Ensure Docker network exists
   - Containers must be on same network for communication
   - Check firewall rules for port access

3. **Task Queue Blocking:**
   - Tasks process sequentially (FIFO)
   - Long-running tasks block the queue
   - Consider timeout mechanisms for stuck tasks

4. **Model Health Check Failures:**
   - Model container must respond to /health quickly
   - Check HEALTH_CHECK_TIMEOUT is sufficient
   - Verify model startup logs for errors

### Performance Considerations

1. **Queue Size:**
   - Default MAX_QUEUE_SIZE=100 may need tuning
   - Large queues consume memory
   - Consider task expiration for old pending tasks

2. **Docker Image Building:**
   - First model start is slow (image build)
   - Subsequent starts are faster (cached image)
   - Consider pre-building images in CI/CD

3. **Health Check Frequency:**
   - HEALTH_CHECK_INTERVAL affects responsiveness
   - Too frequent checks waste resources
   - Balance between detection speed and overhead

---

## Version Information

- **Document Version:** 1.0.0
- **Last Updated:** 2025-11-02
- **Instance Version:** Based on commit 8f28433
- **Python Version:** 3.11+
- **FastAPI Version:** ^0.115.6
- **Docker API Version:** docker-py ^7.1.0

---

## Support & Contact

For questions or issues:
- Repository: swarmpilot-refresh monorepo
- Instance Package: swarmpilot-instance
- Maintainer: cydia2001 <cydia2001@duck.com>
