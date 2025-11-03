# Model Restart API Implementation

## Overview

The `/model/restart` API provides a **non-blocking** mechanism for gracefully switching models and/or schedulers on an instance. The operation runs in the background, allowing the API to return immediately while the restart process executes asynchronously.

## Architecture

### State Machine

The restart operation follows a well-defined state machine with 9 states:

1. **PENDING** - Restart request received, operation starting
2. **DRAINING** - Requesting scheduler to stop sending new tasks
3. **WAITING_TASKS** - Waiting for pending tasks to complete
4. **STOPPING_MODEL** - Stopping the current model container
5. **DEREGISTERING** - Deregistering from the current scheduler
6. **STARTING_MODEL** - Starting the new model container
7. **REGISTERING** - Registering with the new scheduler
8. **COMPLETED** - Restart completed successfully
9. **FAILED** - Restart failed (error stored in operation)

### Components Modified

#### 1. `instance/src/models.py`
- Added `RestartStatus` enum for tracking restart states
- Added `RestartOperation` model to store restart operation state
  - Tracks operation ID, status, model IDs, parameters
  - Records timestamps, task completion progress, errors

#### 2. `instance/src/scheduler_client.py`
- Added `drain_instance()` method
  - Calls scheduler's `/instance/drain` endpoint
  - Returns drain status including pending task count

#### 3. `instance/src/api.py`
- Added request/response schemas:
  - `ModelRestartRequest`
  - `ModelRestartResponse`
  - `RestartStatusResponse`

- Added global state management:
  - `_restart_operations` dictionary to track operations
  - `_restart_operation_lock` for thread-safe access

- Implemented endpoints:
  - **POST `/model/restart`** - Initiates restart operation
  - **GET `/model/restart/status`** - Queries operation status

- Implemented background task:
  - `_perform_restart_operation()` - Executes the restart workflow

## API Usage

### Initiating a Restart

```bash
curl -X POST http://localhost:5000/model/restart \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "new_model_id",
    "parameters": {"key": "value"},
    "scheduler_url": "http://new-scheduler:8000"
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Model restart operation initiated",
  "operation_id": "abc-123-def-456",
  "status": "pending"
}
```

### Monitoring Restart Progress

```bash
curl "http://localhost:5000/model/restart/status?operation_id=abc-123-def-456"
```

**Response (in progress):**
```json
{
  "success": true,
  "operation_id": "abc-123-def-456",
  "status": "waiting_tasks",
  "old_model_id": "old_model",
  "new_model_id": "new_model",
  "initiated_at": "2025-10-31T10:30:00Z",
  "completed_at": null,
  "pending_tasks_at_start": 5,
  "pending_tasks_completed": 3,
  "error": null
}
```

**Response (completed):**
```json
{
  "success": true,
  "operation_id": "abc-123-def-456",
  "status": "completed",
  "old_model_id": "old_model",
  "new_model_id": "new_model",
  "initiated_at": "2025-10-31T10:30:00Z",
  "completed_at": "2025-10-31T10:32:15Z",
  "pending_tasks_at_start": 5,
  "pending_tasks_completed": 5,
  "error": null
}
```

## Workflow Details

### 1. Drain Phase (DRAINING)
- Calls scheduler's `/instance/drain` endpoint
- Scheduler stops assigning new tasks to this instance
- Continues even if drain fails (instance might not be registered)

### 2. Wait Phase (WAITING_TASKS)
- Polls task queue every second
- Tracks completion progress (`pending_tasks_completed`)
- Implements 5-minute timeout
- If timeout occurs, operation fails with `TimeoutError`

### 3. Stop Phase (STOPPING_MODEL)
- Stops the current model container
- Records the old model ID for tracking

### 4. Deregister Phase (DEREGISTERING)
- Calls scheduler's `/instance/remove` endpoint
- Continues even if deregistration fails

### 5. Start Phase (STARTING_MODEL)
- Validates new model exists in registry
- Starts new model container with provided parameters

### 6. Register Phase (REGISTERING)
- Updates scheduler URL if provided
- Registers instance with new/updated scheduler
- Continues even if registration fails (logs warning)

### 7. Completion
- Updates operation status to COMPLETED or FAILED
- Stores completion timestamp
- Stores error message if failed

## Error Handling

### API Validation Errors
- **400 Bad Request**: No model running, invalid model ID
- **409 Conflict**: Another restart operation in progress
- **404 Not Found**: Operation ID not found (status query)

### Runtime Errors
All runtime errors are caught and stored in the operation:
- Timeout waiting for tasks
- Model validation failures
- Container start/stop failures
- Scheduler communication failures

The operation status becomes `FAILED` and the error message is stored in the `error` field.

## Testing

A test script is provided at `instance/test_restart_api.py`:

```bash
# Make sure instance is running with a model
cd instance
./test_restart_api.py
```

The script:
1. Checks current instance state
2. Initiates a restart operation
3. Polls status every 5 seconds
4. Reports completion or failure
5. Verifies final instance state

## Concurrency & Thread Safety

- Uses `asyncio.Lock` (`_restart_operation_lock`) for thread-safe operation access
- Only one restart operation can run at a time
- Background task runs via `asyncio.create_task()`
- All state updates are atomic and protected by locks

## Configuration

### Timeout Settings
- Task completion timeout: **300 seconds** (5 minutes)
  - Configurable in `_perform_restart_operation()`
  - Can be adjusted based on expected task durations

### Polling Intervals
- Task queue polling: **1 second**
  - Configured in the `while True` loop in `_perform_restart_operation()`

## Future Enhancements

Potential improvements for future versions:

1. **Configurable Timeouts**: Make timeout values configurable via API or config file
2. **Operation History**: Implement cleanup of old completed/failed operations
3. **Cancellation Support**: Allow cancelling in-progress restart operations
4. **Webhooks**: Support callback URLs for operation completion notifications
5. **Dry Run Mode**: Validate restart parameters without executing
6. **Progress Events**: Emit detailed progress events for monitoring systems

## Integration with Scheduler

The restart API integrates with the scheduler's drain API:

1. **Drain Endpoint**: `POST /instance/drain`
   - Puts instance in draining mode
   - Stops scheduler from assigning new tasks
   - Returns pending task count and estimated completion time

2. **Remove Endpoint**: `POST /instance/remove`
   - Deregisters instance from scheduler
   - Required before registering with a new scheduler

3. **Register Endpoint**: `POST /instance/register`
   - Registers instance with scheduler
   - Provides model_id and platform_info

## Documentation

Complete API documentation is available in:
- [API Reference](docs/5.API_REFERENCE.md) - Full endpoint documentation
- Test script: `test_restart_api.py` - Usage examples

## Summary

The `/model/restart` API provides a robust, non-blocking mechanism for model and scheduler migration. It handles the complete workflow from draining to registration, with comprehensive error handling, progress tracking, and state management. The implementation follows the scheduler's drain API patterns and provides full visibility into the restart process through the status endpoint.
