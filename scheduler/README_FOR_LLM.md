# README FOR LLM: Scheduler Service

**Version:** 0.1.0
**Python Requirement:** >=3.11
**Primary Entry Point:** `src/cli.py:118` (sscheduler CLI)
**Main API Definition:** `src/api.py`

---

## SYSTEM OVERVIEW

### Purpose
An intelligent task scheduling service that distributes computational tasks across multiple compute instances using ML-based runtime predictions and configurable scheduling strategies.

### Core Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        Scheduler Service                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  FastAPI     │  │  Scheduling  │  │  WebSocket Manager   │  │
│  │  Endpoints   │  │  Strategies  │  │  (Real-time notify)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘  │
│         │                 │                                      │
│  ┌──────▼─────────────────▼───────┐  ┌──────────────────────┐  │
│  │     Task Registry              │  │  Instance Registry   │  │
│  │  (Thread-safe task lifecycle)  │  │  (Queue management)  │  │
│  └────────────────────────────────┘  └──────────────────────┘  │
│         │                                      │                │
│  ┌──────▼──────────────┐            ┌─────────▼─────────────┐  │
│  │  Task Dispatcher    │            │  Predictor Client     │  │
│  │  (Async execution)  │            │  (WebSocket to ML)    │  │
│  └─────────────────────┘            └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐
│ Compute Instance │◄─── Callback ────┤ Predictor Service│
│  (HTTP endpoint) │                  │  (ML predictions)│
└──────────────────┘                  └──────────────────┘
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **API Layer** | `src/api.py` | FastAPI endpoints for all operations |
| **Task Registry** | `src/task_registry.py` | Thread-safe task state management |
| **Instance Registry** | `src/instance_registry.py` | Thread-safe instance & queue management |
| **Scheduling Strategies** | `src/scheduler.py` | MinTime, Probabilistic, RoundRobin algorithms |
| **Task Dispatcher** | `src/task_dispatcher.py` | Async task execution on instances |
| **Predictor Client** | `src/predictor_client.py` | WebSocket client for ML predictions |
| **Training Client** | `src/training_client.py` | HTTP client for training data collection |
| **WebSocket Manager** | `src/websocket_manager.py` | Real-time result notifications |
| **Data Models** | `src/model.py` | Pydantic schemas for all data structures |
| **Configuration** | `src/config.py` | Environment-based configuration |

### Data Flow: Task Submission to Completion

```
1. Client → POST /task/submit
2. API validates request → checks available instances
3. Scheduling Strategy → requests predictions from PredictorClient
4. PredictorClient → WebSocket to Predictor Service (ML inference)
5. Strategy selects optimal instance based on algorithm
6. TaskRegistry stores task with prediction metadata
7. InstanceRegistry updates queue info + pending count
8. TaskDispatcher → HTTP POST to Instance endpoint (async)
9. Instance processes task → POST /callback/task_result
10. Callback Handler → updates TaskRegistry + InstanceRegistry
11. TrainingClient collects sample (if auto-training enabled)
12. WebSocketManager broadcasts result to subscribers
```

---

## QUICK START

### Installation

```bash
# Clone repository
cd /path/to/scheduler

# Install dependencies using uv (recommended)
uv sync

# Alternatively, install as package
uv pip install -e .
```

### Basic Startup

```bash
# Default configuration (0.0.0.0:8000)
uv run sscheduler start

# Custom host and port
uv run sscheduler start --host 127.0.0.1 --port 9000

# With configuration file
uv run sscheduler start --config config.toml

# Check version
uv run sscheduler version
```

### Minimal Working Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Register a compute instance
response = requests.post(f"{BASE_URL}/instance/register", json={
    "instance_id": "worker-001",
    "model_id": "llama-7b",
    "endpoint": "http://worker-001:8080",
    "platform_info": {
        "gpu_type": "A100",
        "gpu_count": 1,
        "memory_gb": 80
    }
})
print(response.json())

# 2. Submit a task
response = requests.post(f"{BASE_URL}/task/submit", json={
    "task_id": "task-001",
    "model_id": "llama-7b",
    "task_input": {
        "prompt": "Explain quantum computing",
        "max_tokens": 100
    },
    "metadata": {"user_id": "user-123"}
})
print(response.json())

# 3. Check task status
response = requests.get(f"{BASE_URL}/task/info", params={"task_id": "task-001"})
print(response.json())
```

---

## ENVIRONMENT VARIABLES

All environment variables are defined in `src/config.py` and loaded via Pydantic settings.

### Predictor Service Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PREDICTOR_URL` | str | `"http://localhost:8001"` | Base URL of the predictor service |
| `PREDICTOR_TIMEOUT` | float | `5.0` | HTTP request timeout in seconds |
| `PREDICTOR_MAX_RETRIES` | int | `3` | Maximum retry attempts for failed requests |
| `PREDICTOR_RETRY_DELAY` | float | `1.0` | Initial retry delay in seconds (exponential backoff) |
| `PREDICTOR_CACHE_TTL` | int | `300` | Prediction cache TTL in seconds (5 minutes) |
| `PREDICTOR_ENABLE_CACHE` | bool | `true` | Enable prediction caching |

**Reference:** `src/config.py:14-34`

### Scheduling Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SCHEDULING_STRATEGY` | str | `"probabilistic"` | Default strategy: "min_time", "probabilistic", "round_robin" |
| `SCHEDULING_PROBABILISTIC_QUANTILE` | float | `0.9` | Target quantile for probabilistic strategy (0.0-1.0) |

**Reference:** `src/config.py:36-46`

### Training Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TRAINING_ENABLE_AUTO` | bool | `false` | Enable automatic training data collection |
| `TRAINING_BATCH_SIZE` | int | `100` | Batch size before auto-flush to training service |
| `TRAINING_FREQUENCY` | int | `3600` | Training frequency in seconds (1 hour) |
| `TRAINING_MIN_SAMPLES` | int | `10` | Minimum samples required before training |

**Reference:** `src/config.py:49-66`

### Logging Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SCHEDULER_LOG_LEVEL` | str | `"INFO"` | Log level: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL |
| `SCHEDULER_LOG_DIR` | str | `"logs"` | Directory for log files (relative or absolute) |
| `SCHEDULER_ENABLE_JSON_LOGS` | bool | `false` | Enable JSON structured logging |

**Reference:** `src/config.py:68-79`

### Server Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SCHEDULER_HOST` | str | `"0.0.0.0"` | Server bind host |
| `SCHEDULER_PORT` | int | `8000` | Server bind port |
| `SCHEDULER_ENABLE_CORS` | bool | `true` | Enable CORS middleware |

**Reference:** `src/config.py:82-96`

### Example Configuration

```bash
# .env file
PREDICTOR_URL=http://predictor-service:8001
SCHEDULING_STRATEGY=probabilistic
SCHEDULING_PROBABILISTIC_QUANTILE=0.95
TRAINING_ENABLE_AUTO=true
SCHEDULER_LOG_LEVEL=DEBUG
SCHEDULER_PORT=9000
```

---

## API ENDPOINTS

All endpoints are defined in `src/api.py` as a FastAPI application.

### Instance Management

#### 1. Register Instance
**Endpoint:** `POST /instance/register`
**Reference:** `src/api.py:184-253`

**Purpose:** Register a compute instance to make it available for task scheduling.

**Input Schema:**
```json
{
  "instance_id": "string (required)",
  "model_id": "string (required)",
  "endpoint": "string (required, HTTP URL)",
  "platform_info": {
    "gpu_type": "string (optional)",
    "gpu_count": "int (optional)",
    "memory_gb": "float (optional)",
    "custom_field": "any (optional)"
  }
}
```

**Output Schema:**
```json
{
  "success": "bool",
  "message": "string",
  "instance": {
    "instance_id": "string",
    "model_id": "string",
    "endpoint": "string",
    "platform_info": "dict",
    "status": "string (active/inactive/error)",
    "registered_at": "string (ISO 8601 timestamp)",
    "last_heartbeat": "string (ISO 8601 timestamp)",
    "total_tasks": "int",
    "completed_tasks": "int",
    "failed_tasks": "int",
    "pending_tasks": "int"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/instance/register \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "gpu-worker-1",
    "model_id": "llama-7b",
    "endpoint": "http://192.168.1.100:8080",
    "platform_info": {"gpu_type": "A100", "gpu_count": 2}
  }'
```

---

#### 2. Remove Instance
**Endpoint:** `POST /instance/remove`
**Reference:** `src/api.py:256-302`

**Purpose:** Deregister an instance from the scheduler.

**Input Schema:**
```json
{
  "instance_id": "string (required)"
}
```

**Output Schema:**
```json
{
  "success": "bool",
  "message": "string",
  "instance_id": "string"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/instance/remove \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "gpu-worker-1"}'
```

---

#### 3. List Instances
**Endpoint:** `GET /instance/list?model_id={optional}`
**Reference:** `src/api.py:305-323`

**Purpose:** List all registered instances, optionally filtered by model_id.

**Query Parameters:**
- `model_id` (optional, string): Filter instances by model ID

**Output Schema:**
```json
{
  "success": "bool",
  "count": "int",
  "instances": [
    {
      "instance_id": "string",
      "model_id": "string",
      "endpoint": "string",
      "platform_info": "dict",
      "status": "string",
      "registered_at": "string",
      "last_heartbeat": "string",
      "total_tasks": "int",
      "completed_tasks": "int",
      "failed_tasks": "int",
      "pending_tasks": "int"
    }
  ]
}
```

**Example:**
```bash
# List all instances
curl http://localhost:8000/instance/list

# Filter by model
curl "http://localhost:8000/instance/list?model_id=llama-7b"
```

---

#### 4. Get Instance Info
**Endpoint:** `GET /instance/info?instance_id={required}`
**Reference:** `src/api.py:326-371`

**Purpose:** Get detailed information about a specific instance including queue state.

**Query Parameters:**
- `instance_id` (required, string): Instance identifier

**Output Schema:**
```json
{
  "success": "bool",
  "instance": {
    "instance_id": "string",
    "model_id": "string",
    "endpoint": "string",
    "platform_info": "dict",
    "status": "string",
    "registered_at": "string",
    "last_heartbeat": "string",
    "total_tasks": "int",
    "completed_tasks": "int",
    "failed_tasks": "int",
    "pending_tasks": "int"
  },
  "queue_info": {
    "expected_completion_time_ms": "float",
    "queue_length": "int",
    "is_idle": "bool"
  },
  "stats": {
    "success_rate": "float (0.0-1.0)",
    "avg_execution_time_ms": "float or null"
  }
}
```

**Example:**
```bash
curl "http://localhost:8000/instance/info?instance_id=gpu-worker-1"
```

---

### Task Management

#### 5. Submit Task
**Endpoint:** `POST /task/submit`
**Reference:** `src/api.py:378-484`

**Purpose:** Submit a new task for execution. The scheduler will select the optimal instance, dispatch the task asynchronously, and return immediately.

**Input Schema:**
```json
{
  "task_id": "string (required, unique)",
  "model_id": "string (required)",
  "task_input": "dict (required, arbitrary structure)",
  "metadata": "dict (optional, arbitrary key-value pairs)"
}
```

**Output Schema:**
```json
{
  "success": "bool",
  "message": "string",
  "task": {
    "task_id": "string",
    "model_id": "string",
    "task_input": "dict",
    "metadata": "dict",
    "status": "string (pending/running/completed/failed)",
    "assigned_instance_id": "string or null",
    "submitted_at": "string (ISO 8601 timestamp)",
    "started_at": "string or null",
    "completed_at": "string or null",
    "result": "dict or null",
    "error": "string or null",
    "execution_time_ms": "float or null",
    "predicted_time_ms": "float or null",
    "prediction_metadata": "dict or null"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/task/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-12345",
    "model_id": "llama-7b",
    "task_input": {
      "prompt": "What is the capital of France?",
      "max_tokens": 50,
      "temperature": 0.7
    },
    "metadata": {
      "user_id": "user-789",
      "priority": "high"
    }
  }'
```

---

#### 6. List Tasks
**Endpoint:** `GET /task/list?status=&model_id=&instance_id=&limit=&offset=`
**Reference:** `src/api.py:487-537`

**Purpose:** List tasks with filtering and pagination support.

**Query Parameters:**
- `status` (optional, string): Filter by status (pending/running/completed/failed)
- `model_id` (optional, string): Filter by model ID
- `instance_id` (optional, string): Filter by assigned instance
- `limit` (optional, int, default=100, max=1000): Number of results per page
- `offset` (optional, int, default=0): Pagination offset

**Output Schema:**
```json
{
  "success": "bool",
  "count": "int (number of results returned)",
  "total": "int (total matching tasks)",
  "offset": "int",
  "limit": "int",
  "tasks": [
    {
      "task_id": "string",
      "model_id": "string",
      "status": "string",
      "assigned_instance_id": "string or null",
      "submitted_at": "string",
      "completed_at": "string or null",
      "execution_time_ms": "float or null"
    }
  ]
}
```

**Example:**
```bash
# Get all completed tasks
curl "http://localhost:8000/task/list?status=completed&limit=50"

# Get tasks for specific model
curl "http://localhost:8000/task/list?model_id=llama-7b&offset=100&limit=50"
```

---

#### 7. Get Task Info
**Endpoint:** `GET /task/info?task_id={required}`
**Reference:** `src/api.py:540-579`

**Purpose:** Get detailed information about a specific task including result or error.

**Query Parameters:**
- `task_id` (required, string): Task identifier

**Output Schema:**
```json
{
  "success": "bool",
  "task": {
    "task_id": "string",
    "model_id": "string",
    "task_input": "dict",
    "metadata": "dict",
    "status": "string",
    "assigned_instance_id": "string or null",
    "submitted_at": "string",
    "started_at": "string or null",
    "completed_at": "string or null",
    "result": "dict or null",
    "error": "string or null",
    "execution_time_ms": "float or null",
    "predicted_time_ms": "float or null",
    "prediction_metadata": "dict or null"
  }
}
```

**Example:**
```bash
curl "http://localhost:8000/task/info?task_id=task-12345"
```

---

#### 8. Clear Tasks
**Endpoint:** `POST /task/clear`
**Reference:** `src/api.py:582-601`

**Purpose:** Clear all tasks from the registry (maintenance operation).

**Input Schema:** None (empty body)

**Output Schema:**
```json
{
  "success": "bool",
  "message": "string",
  "cleared_count": "int"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/task/clear
```

---

### Callback Endpoints

#### 9. Task Result Callback
**Endpoint:** `POST /callback/task_result`
**Reference:** `src/api.py:604-653`

**Purpose:** Callback endpoint for compute instances to report task completion. This is called by instances after they finish executing a task.

**Input Schema:**
```json
{
  "task_id": "string (required)",
  "status": "string (required: completed/failed)",
  "result": "dict (optional, task result data)",
  "error": "string (optional, error message if failed)",
  "execution_time_ms": "float (required, actual execution time)"
}
```

**Output Schema:**
```json
{
  "success": "bool",
  "message": "string"
}
```

**Example (from instance to scheduler):**
```bash
curl -X POST http://scheduler:8000/callback/task_result \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-12345",
    "status": "completed",
    "result": {
      "generated_text": "The capital of France is Paris.",
      "tokens_used": 8
    },
    "execution_time_ms": 234.56
  }'
```

---

### WebSocket Endpoints

#### 10. Get Task Result (WebSocket)
**Endpoint:** `WS /task/get_result`
**Reference:** `src/api.py:656-762`

**Purpose:** Real-time task result notifications via WebSocket. Clients can subscribe to specific task IDs and receive results as they complete.

**Connection:** `ws://localhost:8000/task/get_result`

**Message Types (Client → Server):**

1. **Subscribe:**
```json
{
  "type": "subscribe",
  "task_ids": ["task-001", "task-002"]
}
```

2. **Unsubscribe:**
```json
{
  "type": "unsubscribe",
  "task_ids": ["task-001"]
}
```

**Message Types (Server → Client):**

1. **Result (on task completion):**
```json
{
  "type": "result",
  "task_id": "task-001",
  "status": "completed",
  "result": {"generated_text": "..."},
  "error": null,
  "execution_time_ms": 234.56,
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

2. **Error (on task failure):**
```json
{
  "type": "error",
  "task_id": "task-001",
  "status": "failed",
  "error": "GPU out of memory",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

3. **Acknowledgment:**
```json
{
  "type": "ack",
  "message": "Subscribed to 2 task(s)"
}
```

**Python Example:**
```python
import asyncio
import websockets
import json

async def listen_for_results():
    uri = "ws://localhost:8000/task/get_result"
    async with websockets.connect(uri) as websocket:
        # Subscribe to tasks
        await websocket.send(json.dumps({
            "type": "subscribe",
            "task_ids": ["task-001", "task-002"]
        }))

        # Listen for results
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "result":
                print(f"Task {data['task_id']} completed: {data['result']}")
            elif data["type"] == "error":
                print(f"Task {data['task_id']} failed: {data['error']}")

asyncio.run(listen_for_results())
```

---

### Strategy Management

#### 11. Get Current Strategy
**Endpoint:** `GET /strategy/get`
**Reference:** `src/api.py:852-865`

**Purpose:** Get information about the currently active scheduling strategy.

**Output Schema:**
```json
{
  "success": "bool",
  "strategy_info": {
    "name": "string (min_time/probabilistic/round_robin)",
    "description": "string",
    "parameters": {
      "target_quantile": "float (only for probabilistic)"
    }
  }
}
```

**Example:**
```bash
curl http://localhost:8000/strategy/get
```

---

#### 12. Set Strategy
**Endpoint:** `POST /strategy/set`
**Reference:** `src/api.py:868-948`

**Purpose:** Switch the scheduling strategy. **Requirements:** No tasks can be running (all instances must be idle). This operation clears the task registry and reinitializes all instances.

**Input Schema:**
```json
{
  "strategy_name": "string (required: min_time/probabilistic/round_robin)",
  "target_quantile": "float (optional, only for probabilistic, default: 0.9)"
}
```

**Output Schema:**
```json
{
  "success": "bool",
  "message": "string",
  "cleared_tasks": "int",
  "reinitialized_instances": "int",
  "strategy_info": {
    "name": "string",
    "description": "string",
    "parameters": "dict"
  }
}
```

**Example:**
```bash
# Switch to probabilistic with 95th percentile target
curl -X POST http://localhost:8000/strategy/set \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "probabilistic",
    "target_quantile": 0.95
  }'

# Switch to round robin
curl -X POST http://localhost:8000/strategy/set \
  -H "Content-Type: application/json" \
  -d '{"strategy_name": "round_robin"}'
```

---

### Health Check

#### 13. Health Check
**Endpoint:** `GET /health`
**Reference:** `src/api.py:955-1002`

**Purpose:** Service health check with comprehensive statistics.

**Output Schema:**
```json
{
  "success": "bool",
  "status": "string (healthy/degraded/unhealthy)",
  "timestamp": "string (ISO 8601)",
  "version": "string",
  "stats": {
    "total_tasks": "int",
    "pending_tasks": "int",
    "running_tasks": "int",
    "completed_tasks": "int",
    "failed_tasks": "int",
    "total_instances": "int",
    "active_instances": "int",
    "inactive_instances": "int",
    "current_strategy": "string",
    "predictor_connected": "bool",
    "training_enabled": "bool"
  }
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Profiling Support

All endpoints support optional performance profiling via query parameters:

**Query Parameters:**
- `profile=true` - Enable profiling for this request
- `profile_format=speedscope` or `profile_format=html` - Output format

**Reference:** `src/api.py:90-125`

**Example:**
```bash
# Profile with speedscope format (JSON)
curl "http://localhost:8000/task/submit?profile=true&profile_format=speedscope" \
  -X POST -H "Content-Type: application/json" -d '{...}'

# Profile with HTML format
curl "http://localhost:8000/instance/list?profile=true&profile_format=html"
```

---

## COMPONENT INTERACTIONS

### External Dependencies

#### 1. Predictor Service (ML Runtime Prediction)

**Type:** External HTTP/WebSocket service
**Client:** `src/predictor_client.py` (PredictorClient class)
**Configuration:** See "Predictor Service Configuration" in environment variables

**Endpoints Used:**

1. **WebSocket: `/ws/predict`** (Primary prediction interface)
   - **Purpose:** Get runtime predictions for task scheduling decisions
   - **Connection:** Persistent WebSocket with automatic reconnection
   - **Request Format:**
   ```json
   {
     "model_id": "string",
     "task_input": "dict",
     "platform_info": "dict"
   }
   ```
   - **Response Format:**
   ```json
   {
     "mean_ms": "float",
     "std_ms": "float",
     "quantiles": {
       "0.1": "float",
       "0.5": "float",
       "0.9": "float",
       "0.95": "float",
       "0.99": "float"
     },
     "confidence": "float (0.0-1.0)"
   }
   ```
   - **Features:**
     - Platform-based batching (groups requests by platform to minimize calls)
     - Prediction caching with TTL (default 300s)
     - Automatic reconnection on connection loss
   - **Reference:** `src/predictor_client.py:173-208` (connect), `src/predictor_client.py:284-367` (predict)

2. **HTTP: `POST /train`** (Training data submission)
   - **Purpose:** Send execution samples to improve predictor accuracy
   - **Client:** `src/training_client.py` (TrainingClient class)
   - **Request Format:**
   ```json
   {
     "samples": [
       {
         "model_id": "string",
         "task_input": "dict",
         "platform_info": "dict",
         "actual_time_ms": "float"
       }
     ]
   }
   ```
   - **Auto-Training:** Enabled via `TRAINING_ENABLE_AUTO=true`, batches samples automatically
   - **Reference:** `src/training_client.py:45-91`

3. **HTTP: `GET /health`** (Health check)
   - **Purpose:** Verify predictor service availability
   - **Reference:** `src/predictor_client.py:369-389`

---

#### 2. Compute Instances (Task Execution Workers)

**Type:** External HTTP services registered dynamically
**Client:** `src/task_dispatcher.py` (TaskDispatcher class)
**Registration:** Instances register via `POST /instance/register`

**Endpoints Expected on Instances:**

1. **`POST <instance_endpoint>/task/submit`** (Task execution)
   - **Purpose:** Execute a task on the instance
   - **Request Format:**
   ```json
   {
     "task_id": "string",
     "model_id": "string",
     "task_input": "dict",
     "callback_url": "string (scheduler's /callback/task_result endpoint)"
   }
   ```
   - **Response:** Accepted (202) or Error (4xx/5xx)
   - **Instance Responsibility:**
     - Execute the task
     - Call back to `callback_url` with result
   - **Reference:** `src/task_dispatcher.py:45-99`

**Callback Flow:**
```
Scheduler → Instance: POST /task/submit
Instance executes task
Instance → Scheduler: POST /callback/task_result (with result)
Scheduler updates task status + notifies WebSocket subscribers
```

---

### Internal Component Interactions

#### Task Submission Flow (Detailed Sequence)

```
1. API Endpoint (src/api.py:378-484)
   ↓
2. Validate request + check available instances
   ↓
3. SchedulingStrategy.schedule() (src/scheduler.py)
   ├─→ PredictorClient.predict() (src/predictor_client.py)
   │   ├─→ Check cache (if enabled)
   │   ├─→ WebSocket request to Predictor Service
   │   └─→ Store prediction in cache
   ├─→ Apply algorithm (MinTime/Probabilistic/RoundRobin)
   └─→ Return selected instance_id
   ↓
4. TaskRegistry.add_task() (src/task_registry.py)
   └─→ Store task with status="pending", prediction metadata
   ↓
5. InstanceRegistry.update_queue_on_submission() (src/instance_registry.py)
   ├─→ Add predicted time to queue
   ├─→ Increment pending_tasks counter
   └─→ Update last_heartbeat
   ↓
6. TaskDispatcher.dispatch_task() (src/task_dispatcher.py)
   └─→ Async HTTP POST to instance endpoint
       └─→ Returns immediately (fire-and-forget)
   ↓
7. API returns TaskSubmitResponse to client
```

#### Task Completion Flow (Callback Handling)

```
1. Instance → POST /callback/task_result (src/api.py:604-653)
   ↓
2. Validate callback request
   ↓
3. TaskRegistry.update_task_status() (src/task_registry.py)
   └─→ Update status, result/error, execution_time_ms, completed_at
   ↓
4. InstanceRegistry.update_instance_stats() (src/instance_registry.py)
   ├─→ Increment completed_tasks or failed_tasks
   ├─→ Update average execution time
   └─→ Decrement pending_tasks
   ↓
5. InstanceRegistry.update_queue_on_completion() (src/instance_registry.py)
   ├─→ Strategy-specific queue adjustment:
   │   ├─ MinTime: Subtract actual time, propagate error
   │   └─ Probabilistic: Monte Carlo resample (1000 samples)
   └─→ Update is_idle status
   ↓
6. TrainingClient.collect_sample() (if auto-training enabled)
   └─→ Add to batch, auto-flush if batch size reached
   ↓
7. WebSocketManager.broadcast_result() (src/websocket_manager.py)
   └─→ Send to all clients subscribed to this task_id
   ↓
8. Return TaskResultCallbackResponse to instance
```

#### Queue Management (Strategy-Specific)

**MinTime Strategy:**
- **On Submission:** `queue_time += predicted_mean_ms`
- **On Completion:** `queue_time -= actual_time_ms`, propagate error to remaining tasks
- **Reference:** `src/instance_registry.py:174-221`

**Probabilistic Strategy:**
- **On Submission:** Add `(predicted_mean_ms, predicted_std_ms)` to queue
- **On Completion:** Adjust prediction error, resample entire queue using Monte Carlo (1000 samples)
- **Reference:** `src/instance_registry.py:223-276`

**RoundRobin Strategy:**
- **No queue tracking:** Simple cyclic assignment
- **Reference:** `src/scheduler.py:350-415`

---

## SCHEDULING STRATEGIES

All strategies are implemented in `src/scheduler.py` and inherit from `SchedulingStrategy` base class.

### 1. Round Robin
**Class:** `RoundRobinSchedulingStrategy`
**Reference:** `src/scheduler.py:350-415`

**Algorithm:**
- Simple cyclic distribution across instances
- No predictor dependency (predictor_client can be None)
- Maintains internal counter, increments on each schedule call

**Use Cases:**
- Testing/development without predictor service
- Equal distribution regardless of instance performance
- Baseline comparison

**Configuration:**
```bash
export SCHEDULING_STRATEGY=round_robin
```

**Mathematical Definition:**
```
selected_index = counter % len(instances)
counter += 1
```

---

### 2. Minimum Expected Time (Shortest Queue)
**Class:** `MinimumExpectedTimeSchedulingStrategy`
**Reference:** `src/scheduler.py:417-558`

**Algorithm:**
- Greedy selection based on expected completion time
- **Expected completion time** = current queue time + predicted task time
- Selects instance with minimum expected completion

**Use Cases:**
- Minimizing average task latency
- Heterogeneous instances with varying performance
- Balanced load distribution

**Configuration:**
```bash
export SCHEDULING_STRATEGY=min_time
```

**Mathematical Definition:**
```
For each instance i:
  queue_time_i = Σ(predicted_times of pending tasks)
  expected_completion_i = queue_time_i + predict(task, instance_i).mean_ms

selected = argmin(expected_completion_i)
```

**Queue Update Logic:**
- **On submission:** Add predicted mean to queue
- **On completion:** Subtract actual time, propagate error to remaining tasks

---

### 3. Probabilistic Scheduling (Quantile-Based)
**Class:** `ProbabilisticSchedulingStrategy`
**Reference:** `src/scheduler.py:560-651`

**Algorithm:**
- Targets specific quantile (e.g., 90th percentile) for tail latency optimization
- Uses Monte Carlo sampling (1000 samples) to estimate queue distribution
- Selects instance with minimum quantile value

**Use Cases:**
- SLA-based scheduling (e.g., p95 < 500ms)
- Tail latency optimization for user-facing services
- Handling variance in task execution times

**Configuration:**
```bash
export SCHEDULING_STRATEGY=probabilistic
export SCHEDULING_PROBABILISTIC_QUANTILE=0.9  # Target 90th percentile
```

**Mathematical Definition:**
```
For each instance i:
  queue_samples = monte_carlo_sample(queue_i, n=1000)
  task_samples = monte_carlo_sample(predict(task, instance_i), n=1000)
  completion_samples = queue_samples + task_samples
  quantile_value_i = percentile(completion_samples, target_quantile)

selected = argmin(quantile_value_i)
```

**Monte Carlo Sampling:**
- Each task prediction has `(mean, std)` modeled as normal distribution
- Queue state: list of `(mean, std)` tuples for pending tasks
- Sample 1000 times: `queue_sample = Σ normal(mean_i, std_i)` for all tasks in queue
- **Reference:** `src/scheduler.py:484-524` (Monte Carlo implementation)

**Queue Update Logic:**
- **On submission:** Add `(predicted_mean, predicted_std)` tuple to queue
- **On completion:** Adjust prediction error, resample entire queue

---

### Strategy Comparison Table

| Feature | Round Robin | Minimum Time | Probabilistic |
|---------|-------------|--------------|---------------|
| **Predictor Required** | No | Yes | Yes |
| **Optimization Goal** | Equal distribution | Minimize avg latency | Minimize tail latency |
| **Complexity** | O(1) | O(n) predictions | O(n) predictions + O(1000n) sampling |
| **Queue Tracking** | None | Mean time | Mean + std distribution |
| **Best For** | Testing, homogeneous | Heterogeneous, avg latency | SLA-based, tail latency |
| **Variance Handling** | None | Assumes mean | Probabilistic model |

---

## DATA MODELS

All Pydantic models are defined in `src/model.py`.

### Core Enumerations

**TaskStatus** (`src/model.py:11-18`)
```python
class TaskStatus(str, Enum):
    PENDING = "pending"      # Submitted, not yet dispatched
    RUNNING = "running"      # Currently executing on instance
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Execution failed
```

**InstanceStatus** (`src/model.py:20-24`)
```python
class InstanceStatus(str, Enum):
    ACTIVE = "active"      # Registered and available
    INACTIVE = "inactive"  # Registered but unavailable
    ERROR = "error"        # Error state
```

---

### Task Models

**Task** (`src/model.py:27-85`)
```python
class Task(BaseModel):
    task_id: str                          # Unique task identifier
    model_id: str                         # Model to use (e.g., "llama-7b")
    task_input: Dict[str, Any]            # Arbitrary input payload
    metadata: Dict[str, Any] = {}         # User-defined metadata
    status: TaskStatus                    # Current task status
    assigned_instance_id: Optional[str]   # Assigned instance (None if pending)
    submitted_at: datetime                # Submission timestamp
    started_at: Optional[datetime]        # Execution start time
    completed_at: Optional[datetime]      # Completion time
    result: Optional[Dict[str, Any]]      # Task result (if completed)
    error: Optional[str]                  # Error message (if failed)
    execution_time_ms: Optional[float]    # Actual execution time
    predicted_time_ms: Optional[float]    # Predicted time from predictor
    prediction_metadata: Optional[Dict]   # Additional prediction info
```

**TaskSubmitRequest** (`src/model.py:88-99`)
```python
class TaskSubmitRequest(BaseModel):
    task_id: str
    model_id: str
    task_input: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

**TaskResultCallbackRequest** (`src/model.py:160-172`)
```python
class TaskResultCallbackRequest(BaseModel):
    task_id: str
    status: Literal["completed", "failed"]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float
```

---

### Instance Models

**Instance** (`src/model.py:175-215`)
```python
class Instance(BaseModel):
    instance_id: str                      # Unique instance identifier
    model_id: str                         # Model supported by instance
    endpoint: str                         # HTTP endpoint URL
    platform_info: Dict[str, Any]         # GPU type, memory, etc.
    status: InstanceStatus                # Current status
    registered_at: datetime               # Registration time
    last_heartbeat: datetime              # Last activity time
    total_tasks: int = 0                  # Total tasks received
    completed_tasks: int = 0              # Successfully completed
    failed_tasks: int = 0                 # Failed tasks
    pending_tasks: int = 0                # Currently pending/running
```

**InstanceRegisterRequest** (`src/model.py:218-228`)
```python
class InstanceRegisterRequest(BaseModel):
    instance_id: str
    model_id: str
    endpoint: str                         # Must be valid HTTP URL
    platform_info: Dict[str, Any] = Field(default_factory=dict)
```

---

### Prediction Models

**PredictionResponse** (`src/model.py:335-367`)
```python
class PredictionResponse(BaseModel):
    mean_ms: float                        # Mean predicted time
    std_ms: float                         # Standard deviation
    quantiles: Dict[str, float]           # Quantile map (0.1, 0.5, 0.9, ...)
    confidence: float                     # Prediction confidence (0.0-1.0)
    cached: bool = False                  # Whether from cache
    metadata: Dict[str, Any] = {}         # Additional info
```

**TrainingSample** (`src/model.py:370-382`)
```python
class TrainingSample(BaseModel):
    model_id: str
    task_input: Dict[str, Any]
    platform_info: Dict[str, Any]
    actual_time_ms: float
```

---

### Queue Models

**QueueInfo** (`src/model.py:385-394`)
```python
class QueueInfo(BaseModel):
    expected_completion_time_ms: float    # Expected time to finish queue
    queue_length: int                     # Number of tasks in queue
    is_idle: bool                         # True if no pending tasks
```

---

## DEVELOPMENT GUIDE

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html --cov-report=term

# Run specific test file
uv run pytest tests/test_scheduler.py

# Run with verbose output
uv run pytest -v

# Run async tests only
uv run pytest -k "async"
```

**Test Configuration:** `pyproject.toml:31-49`
**Coverage Requirement:** ≥90% (enforced in pyproject.toml:49)

---

### Profiling Endpoints

All API endpoints support performance profiling via query parameters.

**Enable Profiling:**
```bash
# Speedscope format (JSON, for speedscope.app)
curl "http://localhost:8000/task/submit?profile=true&profile_format=speedscope" \
  -X POST -H "Content-Type: application/json" -d '{...}' > profile.json

# HTML format (self-contained HTML report)
curl "http://localhost:8000/instance/list?profile=true&profile_format=html" > profile.html
```

**View Results:**
- **Speedscope:** Upload `profile.json` to https://speedscope.app
- **HTML:** Open `profile.html` in browser

**Implementation:** `src/api.py:90-125` (profiling middleware)

---

### Logging Configuration

**Log Levels:**
```bash
export SCHEDULER_LOG_LEVEL=TRACE   # Most verbose
export SCHEDULER_LOG_LEVEL=DEBUG   # Debug information
export SCHEDULER_LOG_LEVEL=INFO    # Default, general info
export SCHEDULER_LOG_LEVEL=SUCCESS # Success events only
export SCHEDULER_LOG_LEVEL=WARNING # Warnings and errors
export SCHEDULER_LOG_LEVEL=ERROR   # Errors only
export SCHEDULER_LOG_LEVEL=CRITICAL # Critical errors only
```

**Log Directory:**
```bash
export SCHEDULER_LOG_DIR=/var/log/scheduler  # Custom directory
# Logs will be written to scheduler.log in this directory
```

**JSON Structured Logging:**
```bash
export SCHEDULER_ENABLE_JSON_LOGS=true
# Outputs logs in JSON format for log aggregation systems
```

**Log Rotation:**
- Daily rotation at midnight
- 7-day retention (7 backup files)
- 10 MB max size per file
- **Reference:** `src/config.py:104-117`

---

### Development Workflow

1. **Setup:**
```bash
uv sync
uv run sscheduler start --host 127.0.0.1 --port 8000
```

2. **Mock Predictor (for testing without predictor service):**
```bash
# Use round_robin strategy (no predictor needed)
export SCHEDULING_STRATEGY=round_robin
uv run sscheduler start
```

3. **Enable Debug Logging:**
```bash
export SCHEDULER_LOG_LEVEL=DEBUG
uv run sscheduler start
```

4. **Run Tests with Coverage:**
```bash
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## FILE STRUCTURE MAP

```
scheduler/
├── src/                          # Main source code
│   ├── api.py                    # FastAPI app, all endpoints (955 lines)
│   ├── cli.py                    # Typer CLI (sscheduler command)
│   ├── config.py                 # Environment-based configuration
│   ├── model.py                  # Pydantic data models
│   ├── scheduler.py              # Scheduling strategies
│   ├── task_registry.py          # Thread-safe task management
│   ├── instance_registry.py      # Thread-safe instance management
│   ├── task_dispatcher.py        # Async task execution dispatcher
│   ├── predictor_client.py       # WebSocket client for predictor
│   ├── training_client.py        # HTTP client for training service
│   └── websocket_manager.py      # WebSocket connection manager
│
├── tests/                        # Test suite
│   ├── test_scheduler.py         # Scheduling strategy tests
│   ├── test_task_registry.py     # Task registry tests
│   ├── test_instance_registry.py # Instance registry tests
│   └── ...                       # Additional tests
│
├── docs/                         # Documentation
│   ├── 1.ARCHITECTURE.md         # System architecture
│   ├── 2.API_DESIGN.md           # API design document
│   ├── 8.SCHEDULING_STRATEGIES.md # Strategy details
│   ├── 9.USAGE_EXAMPLES.md       # Code examples
│   └── ...                       # Additional docs
│
├── cli.py                        # Dev entry point (wrapper for src.cli)
├── pyproject.toml                # Project metadata, dependencies
├── uv.lock                       # Dependency lock file
├── .env.example                  # Example environment variables
└── README.md                     # User-facing README

Entry Points:
├── CLI: sscheduler (defined in pyproject.toml:20)
│   └── Implementation: src/cli.py:app (Typer CLI)
│       └── Commands:
│           ├── start: src/cli.py:118-199
│           └── version: src/cli.py:201-205
│
└── API Server: src.api:app (FastAPI)
    └── Started by: uvicorn in src/cli.py:187-192
```

**Key File Locations:**
- **Main API:** `src/api.py` (all 13 REST endpoints + 1 WebSocket)
- **CLI Entry:** `src/cli.py:118` (start command)
- **Config:** `src/config.py` (all environment variables)
- **Models:** `src/model.py` (all Pydantic schemas)
- **Strategies:** `src/scheduler.py` (3 scheduling algorithms)

---

## APPENDIX: QUICK REFERENCE

### Status Values
- **Task:** pending, running, completed, failed
- **Instance:** active, inactive, error

### Strategy Names
- `round_robin` - Cyclic distribution
- `min_time` - Shortest queue (minimum expected time)
- `probabilistic` - Quantile-based (tail latency optimization)

### Default Ports
- **Scheduler:** 8000 (configurable via `SCHEDULER_PORT`)
- **Predictor:** 8001 (configurable via `PREDICTOR_URL`)

### Common URLs
- Health: `GET http://localhost:8000/health`
- List instances: `GET http://localhost:8000/instance/list`
- Submit task: `POST http://localhost:8000/task/submit`
- WebSocket: `ws://localhost:8000/task/get_result`

### Important File References
| Component | File | Key Lines |
|-----------|------|-----------|
| Task submission endpoint | src/api.py | 378-484 |
| Callback handler | src/api.py | 604-653 |
| WebSocket endpoint | src/api.py | 656-762 |
| Probabilistic scheduling | src/scheduler.py | 560-651 |
| Monte Carlo sampling | src/scheduler.py | 484-524 |
| Predictor WebSocket | src/predictor_client.py | 173-367 |
| Queue management | src/instance_registry.py | 174-276 |
| CLI start command | src/cli.py | 118-199 |
| Environment config | src/config.py | 14-96 |

---

**Document Version:** 1.0
**Last Updated:** 2025-11-02
**Scheduler Version:** 0.1.0

For additional information, see the `docs/` directory or the source code references provided throughout this document.
