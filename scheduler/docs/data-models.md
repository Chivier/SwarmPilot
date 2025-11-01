# Data Models

This document describes all data models used by the scheduler service for API requests, responses, and internal state management.

> **See Also**: [API Reference](./README.md#api-reference) for endpoint documentation

## Core Models

### Task

Represents a computational task to be executed.

```python
class Task(BaseModel):
    task_id: str                    # Unique identifier for this task
    model_id: str                   # Model/tool to use for execution
    task_input: Dict[str, Any]      # Input data for the model
    metadata: Dict[str, Any]        # Metadata for prediction (e.g., image dimensions)
```

**Example:**
```json
{
  "task_id": "task-ocr-dec-0",
  "model_id": "easyocr/dec",
  "task_input": {
    "image": "base64_encoded_image_data"
  },
  "metadata": {
    "height": 512,
    "width": 512
  }
}
```

### Instance

Represents a compute instance that can execute tasks.

```python
class Instance(BaseModel):
    instance_id: str                # Unique identifier for this instance
    model_id: str                   # Model running on this instance
    endpoint: str                   # HTTP endpoint for task dispatch
    platform_info: Dict[str, str]   # Hardware/software information
```

**Example:**
```json
{
  "instance_id": "instance-gpu-0",
  "model_id": "easyocr/dec",
  "endpoint": "http://localhost:8300",
  "platform_info": {
    "software_name": "docker",
    "software_version": "20.10",
    "hardware_name": "nvidia-rtx-3090"
  }
}
```

### Task Status

```python
class TaskStatus(str, Enum):
    PENDING = "pending"             # Task submitted, waiting to be dispatched
    RUNNING = "running"             # Task dispatched to instance, executing
    COMPLETED = "completed"         # Task finished successfully
    FAILED = "failed"               # Task failed with error
```

## Queue Information Models

### InstanceQueueBase

Base class for queue information. Each scheduling strategy uses its own derived type.

```python
class InstanceQueueBase(BaseModel):
    instance_id: str                # ID of the instance
```

### InstanceQueueProbabilistic

Queue information for probabilistic scheduling strategy.

```python
class InstanceQueueProbabilistic(InstanceQueueBase):
    quantiles: List[float]          # Quantile positions (e.g., [0.5, 0.9, 0.95, 0.99])
    values: List[float]             # Estimated queue time at each quantile (ms)
```

**Example:**
```json
{
  "instance_id": "instance-gpu-0",
  "quantiles": [0.5, 0.9, 0.95, 0.99],
  "values": [120.5, 250.3, 300.7, 450.2]
}
```

### InstanceQueueExpectError

Queue information for shortest queue (minimum expected time) strategy.

```python
class InstanceQueueExpectError(InstanceQueueBase):
    expected_time_ms: float         # Expected queue completion time (ms)
    error_margin_ms: float          # Uncertainty/error margin (ms)
```

**Example:**
```json
{
  "instance_id": "instance-gpu-0",
  "expected_time_ms": 150.5,
  "error_margin_ms": 25.3
}
```

## Instance Management Models

### InstanceRegisterRequest

```python
class InstanceRegisterRequest(BaseModel):
    instance_id: str                # Unique identifier for the instance
    model_id: str                   # Model/tool ID running on this instance
    endpoint: str                   # HTTP endpoint URL of the instance
    platform_info: Dict[str, str]   # Platform information for predictions
```

### InstanceRegisterResponse

```python
class InstanceRegisterResponse(BaseModel):
    success: bool
    message: str
    instance: Instance
```

### InstanceRemoveRequest

```python
class InstanceRemoveRequest(BaseModel):
    instance_id: str                # ID of the instance to remove
```

### InstanceRemoveResponse

```python
class InstanceRemoveResponse(BaseModel):
    success: bool
    message: str
    instance_id: str
```

### InstanceListResponse

```python
class InstanceListResponse(BaseModel):
    success: bool
    count: int                      # Number of instances returned
    instances: List[Instance]
```

### InstanceStats

Statistics for an instance's task execution.

```python
class InstanceStats(BaseModel):
    pending_tasks: int              # Tasks assigned but not yet dispatched
    completed_tasks: int            # Successfully completed tasks
    failed_tasks: int               # Failed tasks
```

### InstanceInfoResponse

```python
class InstanceInfoResponse(BaseModel):
    success: bool
    instance: Instance
    queue_info: InstanceQueueBase   # Can be InstanceQueueProbabilistic or InstanceQueueExpectError
    stats: InstanceStats
```

## Task Management Models

### TaskSubmitRequest

```python
class TaskSubmitRequest(BaseModel):
    task_id: str                    # Unique identifier for the task
    model_id: str                   # Model/tool ID to use for this task
    task_input: Dict[str, Any]      # Input data for the model
    metadata: Dict[str, Any]        # Metadata for runtime prediction
```

### TaskSubmitResponse

```python
class TaskSubmitResponse(BaseModel):
    success: bool
    message: str
    task: TaskInfo
```

### TaskInfo

Basic task information returned after submission.

```python
class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    assigned_instance: str          # Instance ID where task was assigned
    submitted_at: str               # ISO 8601 timestamp
```

### TaskListResponse

```python
class TaskListResponse(BaseModel):
    success: bool
    count: int                      # Number of tasks in current page
    total: int                      # Total tasks matching filter
    offset: int                     # Pagination offset
    limit: int                      # Pagination limit
    tasks: List[TaskSummary]
```

### TaskSummary

Summary information for task listings.

```python
class TaskSummary(BaseModel):
    task_id: str
    model_id: str
    status: TaskStatus
    assigned_instance: str
    submitted_at: str               # ISO 8601 timestamp
    completed_at: Optional[str]     # ISO 8601 timestamp, only if completed/failed
```

### TaskTimestamps

```python
class TaskTimestamps(BaseModel):
    submitted_at: str               # ISO 8601 timestamp
    started_at: Optional[str]       # ISO 8601 timestamp
    completed_at: Optional[str]     # ISO 8601 timestamp
```

### TaskDetailInfo

Complete task information including input, result, and timestamps.

```python
class TaskDetailInfo(BaseModel):
    task_id: str
    model_id: str
    status: TaskStatus
    assigned_instance: str
    task_input: Dict[str, Any]
    metadata: Dict[str, Any]
    result: Optional[Dict[str, Any]]    # Only present if status is "completed"
    error: Optional[str]                # Only present if status is "failed"
    timestamps: TaskTimestamps
    execution_time_ms: Optional[int]    # Actual execution time in milliseconds
```

### TaskDetailResponse

```python
class TaskDetailResponse(BaseModel):
    success: bool
    task: TaskDetailInfo
```

### TaskClearResponse

Response from the `/task/clear` endpoint indicating how many tasks were removed.

```python
class TaskClearResponse(BaseModel):
    success: bool
    message: str          # Human-readable confirmation message
    cleared_count: int    # Number of tasks that were cleared
```

**Example:**
```json
{
  "success": true,
  "message": "Successfully cleared 42 task(s)",
  "cleared_count": 42
}
```

### TaskResultCallbackRequest

Used by instances to report task completion via callback.

```python
class TaskResultCallbackRequest(BaseModel):
    task_id: str
    status: str                     # "completed" or "failed"
    result: Optional[Dict[str, Any]]    # Result data if completed
    error: Optional[str]                # Error message if failed
    execution_time_ms: Optional[float]  # Actual execution time
```

### TaskResultCallbackResponse

```python
class TaskResultCallbackResponse(BaseModel):
    success: bool
    message: str
```

## WebSocket Models

### WSMessageType

```python
class WSMessageType(str, Enum):
    SUBSCRIBE = "subscribe"         # Subscribe to task result updates
    UNSUBSCRIBE = "unsubscribe"     # Unsubscribe from task updates
    RESULT = "result"               # Task result notification
    ERROR = "error"                 # Error notification
    ACK = "ack"                     # Acknowledgment message
```

### WSSubscribeMessage

Client request to subscribe to task results.

```python
class WSSubscribeMessage(BaseModel):
    type: WSMessageType = WSMessageType.SUBSCRIBE
    task_ids: List[str]             # List of task IDs to subscribe to
```

### WSUnsubscribeMessage

Client request to unsubscribe from task results.

```python
class WSUnsubscribeMessage(BaseModel):
    type: WSMessageType = WSMessageType.UNSUBSCRIBE
    task_ids: List[str]             # List of task IDs to unsubscribe from
```

### WSAckMessage

Server acknowledgment of subscription/unsubscription.

```python
class WSAckMessage(BaseModel):
    type: WSMessageType = WSMessageType.ACK
    message: str
    subscribed_tasks: List[str]     # Currently subscribed task IDs
```

### WSTaskResultMessage

Server notification of task completion.

```python
class WSTaskResultMessage(BaseModel):
    type: WSMessageType = WSMessageType.RESULT
    task_id: str
    status: TaskStatus              # "completed" or "failed"
    result: Optional[Dict[str, Any]]    # Task result if completed
    error: Optional[str]                # Error message if failed
    timestamps: TaskTimestamps
    execution_time_ms: Optional[int]
```

### WSErrorMessage

Server error notification.

```python
class WSErrorMessage(BaseModel):
    type: WSMessageType = WSMessageType.ERROR
    error: str
    task_id: Optional[str]          # Task ID if error is task-specific
```

## Health Check Models

### HealthStats

System-wide statistics.

```python
class HealthStats(BaseModel):
    total_instances: int
    active_instances: int
    total_tasks: int
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
```

### HealthResponse

```python
class HealthResponse(BaseModel):
    success: bool
    status: str                     # "healthy" or "unhealthy"
    timestamp: str                  # ISO 8601 timestamp
    version: str                    # Service version
    stats: HealthStats
```

### HealthErrorResponse

```python
class HealthErrorResponse(BaseModel):
    success: bool
    status: str                     # "unhealthy"
    error: str
    timestamp: str                  # ISO 8601 timestamp
```

## Common Response Models

### ErrorResponse

Generic error response format.

```python
class ErrorResponse(BaseModel):
    success: bool = False
    error: str                      # Error description
```

---

## Model Validation

All models use [Pydantic](https://docs.pydantic.dev/) for automatic validation:
- **Type checking**: Ensures fields have correct types
- **Required fields**: Missing required fields trigger validation errors
- **Optional fields**: Use `Optional[T]` or `T | None` for optional fields
- **Enums**: String enums ensure only valid values are accepted

## Source Code Reference

All models are defined in `src/model.py`.
