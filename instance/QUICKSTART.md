# Quick Start Guide

This guide will help you get the Instance Service up and running with the example sleep_model.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+
- uv (Python package manager)

## Step 1: Install Dependencies

```bash
uv sync
```

## Step 2: Build the Sleep Model Docker Image

```bash
chmod +x build_sleep_model.sh
./build_sleep_model.sh
```

This will build the Docker image for the example sleep_model container.

## Step 3: Test Docker Management (Optional)

Run the test script to verify Docker operations:

```bash
uv run python test_docker.py
```

This will:
1. Start the sleep_model container
2. Check health
3. Submit a test task (sleep 2 seconds)
4. Wait for completion
5. Stop the container

## Step 4: Start the Instance Service

```bash
uv run python -m src.api
```

The service will start on `http://localhost:5000`.

## Step 5: Try the API

### Check Instance Health

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-31T12:00:00Z"
}
```

### Get Instance Info

```bash
curl http://localhost:5000/info
```

### Start the Sleep Model

```bash
curl -X POST http://localhost:5000/model/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sleep_model",
    "parameters": {}
  }'
```

Expected response:
```json
{
  "success": true,
  "message": "Model started successfully",
  "model_id": "sleep_model",
  "status": "running"
}
```

### Submit a Task

```bash
curl -X POST http://localhost:5000/task/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-001",
    "model_id": "sleep_model",
    "task_input": {
      "sleep_time": 3.5
    }
  }'
```

Expected response:
```json
{
  "success": true,
  "message": "Task submitted successfully",
  "task_id": "task-001",
  "status": "queued",
  "position": 1
}
```

### Check Task Status

```bash
curl http://localhost:5000/task/task-001
```

Response (when completed):
```json
{
  "success": true,
  "task": {
    "task_id": "task-001",
    "model_id": "sleep_model",
    "status": "completed",
    "task_input": {
      "sleep_time": 3.5
    },
    "submitted_at": "2025-10-31T12:00:00Z",
    "started_at": "2025-10-31T12:00:01Z",
    "completed_at": "2025-10-31T12:00:04Z",
    "result": {
      "success": true,
      "result": {
        "sleep_time": 3.5,
        "actual_sleep_time": 3.501,
        "model_id": "sleep_model",
        "instance_id": "instance-default",
        "message": "Slept for 3.501 seconds"
      },
      "execution_time": 3.501,
      "start_timestamp": 1234567890
    },
    "error": null
  }
}
```

### List All Tasks

```bash
curl http://localhost:5000/task/list
```

### Stop the Model

```bash
curl http://localhost:5000/model/stop
```

Expected response:
```json
{
  "success": true,
  "message": "Model stopped successfully",
  "model_id": "sleep_model"
}
```

## Configuration

You can customize the instance using environment variables:

```bash
# Start on a different port
INSTANCE_PORT=5001 uv run python -m src.api

# Set a custom instance ID
INSTANCE_ID=instance-1 INSTANCE_PORT=5001 uv run python -m src.api

# Enable debug logging
LOG_LEVEL=DEBUG uv run python -m src.api
```

## Troubleshooting

### Docker Container Won't Start

Check Docker daemon is running:
```bash
docker ps
```

Check if port is already in use:
```bash
lsof -i :6000
```

### Health Check Fails

Check container logs:
```bash
docker logs model_instance-default_sleep_model
```

### Task Fails

Check the task error message:
```bash
curl http://localhost:5000/task/task-001 | jq .task.error
```

## Next Steps

- Read the [Architecture Overview](./docs/1.ARCHITECTURE.md)
- Learn how to [Create Custom Models](./docs/2.MODEL_CONTAINER_SPEC.md)
- Explore the [API Reference](./docs/5.API_REFERENCE.md)
