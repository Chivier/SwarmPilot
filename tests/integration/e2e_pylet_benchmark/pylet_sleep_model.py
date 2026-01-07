"""PyLet-Compatible Sleep Model for E2E Testing.

This module provides a standalone FastAPI sleep model that can be deployed
via PyLet. It reads the port from the $PORT environment variable (set by
PyLet's automatic port allocation) and exposes /inference and /health endpoints.

Features:
- Reads $PORT from environment (PyLet auto-allocation)
- Registers with scheduler on startup via SCHEDULER_URL
- Handles SIGTERM gracefully for PyLet cancellation
- Tracks execution statistics

Endpoints:
- POST /inference - Sleep for specified time and return result
- GET /health - Health check endpoint

Environment Variables:
- PORT: Port to run on (required, set by PyLet)
- MODEL_ID: Model identifier (default: sleep_model)
- INSTANCE_ID: Instance identifier (auto-generated if not set)
- SCHEDULER_URL: Scheduler URL for registration (optional)
- LOG_LEVEL: Log level (default: INFO)

Usage:
    # PyLet deploys this with:
    pylet submit "python pylet_sleep_model.py" --gpu 0 --name sleep_model_a-001

    # Or run directly for testing:
    PORT=8300 MODEL_ID=sleep_model_a python pylet_sleep_model.py
"""

import asyncio
import os
import signal
import socket
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field


# Configuration from environment
PORT = int(os.environ.get("PORT", "8000"))
MODEL_ID = os.environ.get("MODEL_ID", "sleep_model")
INSTANCE_ID = os.environ.get("INSTANCE_ID", f"{MODEL_ID}-{uuid.uuid4().hex[:8]}")
SCHEDULER_URL = os.environ.get("SCHEDULER_URL", "")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Get hostname for endpoint construction
HOSTNAME = socket.gethostname()
try:
    # Try to get the IP address
    HOST_IP = socket.gethostbyname(HOSTNAME)
except socket.gaierror:
    HOST_IP = "127.0.0.1"

# Track statistics
_stats = {
    "requests_received": 0,
    "requests_completed": 0,
    "requests_failed": 0,
    "total_sleep_time": 0.0,
    "start_time": None,
}


# Request/Response Models
class InferenceRequest(BaseModel):
    """Inference request for sleep model."""

    sleep_time: float = Field(
        default=1.0,
        description="Time to sleep in seconds",
        ge=0.0,
        le=300.0,  # Max 5 minutes
    )

    class Config:
        populate_by_name = True
        extra = "allow"  # Allow additional fields


class InferenceResult(BaseModel):
    """Result of inference execution."""

    sleep_time: float
    actual_sleep_time: float
    model_id: str
    instance_id: str
    message: str
    start_time: float


class InferenceResponse(BaseModel):
    """Response from inference endpoint."""

    success: bool
    result: InferenceResult
    execution_time: float
    start_timestamp: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_id: str
    instance_id: str


class TaskSubmitRequest(BaseModel):
    """Request format from scheduler for task submission."""

    task_id: str
    model_id: str
    task_input: dict[str, Any] = Field(default_factory=dict)
    callback_url: str | None = None
    enqueue_time: float | None = None

    class Config:
        extra = "allow"


class TaskSubmitResponse(BaseModel):
    """Response to scheduler for task submission."""

    success: bool
    message: str = ""
    task_id: str = ""


# Registration logic
async def register_with_scheduler() -> bool:
    """Register this instance with the scheduler.

    Returns:
        True if registration succeeded, False otherwise
    """
    if not SCHEDULER_URL:
        logger.info("SCHEDULER_URL not set, skipping registration")
        return True

    endpoint = f"http://{HOST_IP}:{PORT}"
    registration_data = {
        "instance_id": INSTANCE_ID,
        "model_id": MODEL_ID,
        "endpoint": endpoint,
        "platform_info": {
            "software_name": "python",
            "software_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "hardware_name": "cpu",  # No GPU for sleep model
        },
    }

    logger.info(
        f"Registering with scheduler at {SCHEDULER_URL}: "
        f"instance_id={INSTANCE_ID}, model_id={MODEL_ID}, endpoint={endpoint}"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SCHEDULER_URL}/instance/register",
                json=registration_data,
            )

            if response.status_code == 200:
                logger.success(
                    f"Successfully registered with scheduler: {response.json()}"
                )
                return True
            else:
                logger.error(
                    f"Failed to register with scheduler: "
                    f"HTTP {response.status_code}: {response.text}"
                )
                return False

    except Exception as e:
        logger.error(f"Failed to register with scheduler: {e}")
        return False


async def deregister_from_scheduler() -> bool:
    """Deregister this instance from the scheduler.

    Returns:
        True if deregistration succeeded, False otherwise
    """
    if not SCHEDULER_URL:
        return True

    logger.info(f"Deregistering from scheduler: instance_id={INSTANCE_ID}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{SCHEDULER_URL}/instance/remove",
                json={"instance_id": INSTANCE_ID},
            )

            if response.status_code == 200:
                logger.success("Successfully deregistered from scheduler")
                return True
            else:
                logger.warning(
                    f"Deregistration returned non-200: "
                    f"HTTP {response.status_code}: {response.text}"
                )
                return False

    except Exception as e:
        logger.warning(f"Failed to deregister from scheduler: {e}")
        return False


# Shutdown handling
_shutdown_event = asyncio.Event()


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_event.set()


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _stats

    # Startup
    logger.info(f"PyLet Sleep Model starting...")
    logger.info(f"  Model ID: {MODEL_ID}")
    logger.info(f"  Instance ID: {INSTANCE_ID}")
    logger.info(f"  Port: {PORT}")
    logger.info(f"  Scheduler URL: {SCHEDULER_URL or 'Not configured'}")

    _stats["start_time"] = time.time()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    # Simulate model loading (brief delay)
    await asyncio.sleep(0.5)
    logger.info("Model loaded successfully")

    # Register with scheduler
    if SCHEDULER_URL:
        # Retry registration a few times
        for attempt in range(3):
            if await register_with_scheduler():
                break
            if attempt < 2:
                logger.warning(f"Registration attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2)
        else:
            logger.error("Failed to register after 3 attempts, continuing anyway")

    logger.success(f"PyLet Sleep Model ready on port {PORT}")

    yield

    # Shutdown
    logger.info("PyLet Sleep Model shutting down...")

    # Deregister from scheduler
    await deregister_from_scheduler()

    # Log final statistics
    uptime = time.time() - _stats["start_time"] if _stats["start_time"] else 0
    logger.info(
        f"Final statistics: "
        f"requests={_stats['requests_completed']}/{_stats['requests_received']}, "
        f"total_sleep={_stats['total_sleep_time']:.2f}s, "
        f"uptime={uptime:.2f}s"
    )

    logger.info("Shutdown complete")


# FastAPI Application
app = FastAPI(
    title=f"PyLet Sleep Model - {MODEL_ID}",
    description="Sleep model for E2E testing, deployable via PyLet",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest) -> InferenceResponse:
    """Execute inference - sleep for the specified duration.

    Args:
        request: InferenceRequest with sleep_time

    Returns:
        InferenceResponse with execution details

    Raises:
        HTTPException: If an error occurs during execution
    """
    global _stats
    _stats["requests_received"] += 1

    try:
        # Record start time
        start_time = time.time()
        start_timestamp = int(start_time)

        logger.debug(
            f"Inference request: sleep_time={request.sleep_time}s, "
            f"request_num={_stats['requests_received']}"
        )

        # Sleep for the specified duration
        await asyncio.sleep(request.sleep_time)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Update statistics
        _stats["requests_completed"] += 1
        _stats["total_sleep_time"] += execution_time

        # Build result
        result = InferenceResult(
            sleep_time=request.sleep_time,
            actual_sleep_time=execution_time,
            model_id=MODEL_ID,
            instance_id=INSTANCE_ID,
            message=f"Slept for {execution_time:.3f} seconds",
            start_time=start_time,
        )

        logger.debug(
            f"Inference completed: actual={execution_time:.3f}s, "
            f"requested={request.sleep_time}s"
        )

        return InferenceResponse(
            success=True,
            result=result,
            execution_time=execution_time,
            start_timestamp=start_timestamp,
        )

    except asyncio.CancelledError:
        _stats["requests_failed"] += 1
        logger.warning("Inference cancelled")
        raise HTTPException(status_code=499, detail="Request cancelled")

    except Exception as e:
        _stats["requests_failed"] += 1
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse indicating model status
    """
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_id=MODEL_ID,
        instance_id=INSTANCE_ID,
    )


@app.get("/stats")
async def stats() -> dict[str, Any]:
    """Get instance statistics.

    Returns:
        Statistics about requests processed
    """
    uptime = time.time() - _stats["start_time"] if _stats["start_time"] else 0
    return {
        **_stats,
        "uptime_seconds": uptime,
        "model_id": MODEL_ID,
        "instance_id": INSTANCE_ID,
        "port": PORT,
    }


async def _process_task_and_callback(
    task_id: str,
    sleep_time: float,
    callback_url: str | None,
    start_time: float,
) -> None:
    """Process a task in background and send callback.

    This function runs in the background after /task/submit returns.

    Args:
        task_id: The task ID from scheduler
        sleep_time: Duration to sleep
        callback_url: URL to send result callback
        start_time: Time task processing started
    """
    global _stats

    try:
        # Sleep for the specified duration
        await asyncio.sleep(sleep_time)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Update statistics
        _stats["requests_completed"] += 1
        _stats["total_sleep_time"] += execution_time

        logger.debug(
            f"Task {task_id} completed: actual={execution_time:.3f}s, "
            f"requested={sleep_time}s"
        )

        # Send callback if URL provided
        # Scheduler expects: task_id, status ('completed'|'failed'), result, error, execution_time_ms
        if callback_url:
            callback_payload = {
                "task_id": task_id,
                "status": "completed",
                "result": {
                    "sleep_time": sleep_time,
                    "actual_sleep_time": execution_time,
                    "model_id": MODEL_ID,
                    "instance_id": INSTANCE_ID,
                    "message": f"Slept for {execution_time:.3f} seconds",
                },
                "execution_time_ms": execution_time * 1000,  # Convert to milliseconds
            }

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(callback_url, json=callback_payload)
                    if response.status_code != 200:
                        logger.warning(
                            f"Callback for task {task_id} returned {response.status_code}: {response.text}"
                        )
                    else:
                        logger.debug(f"Callback sent for task {task_id}")
            except Exception as e:
                logger.error(f"Failed to send callback for task {task_id}: {e}")

    except asyncio.CancelledError:
        _stats["requests_failed"] += 1
        logger.warning(f"Task {task_id} cancelled")
        raise

    except Exception as e:
        _stats["requests_failed"] += 1
        logger.error(f"Task {task_id} failed: {e}")

        # Send failure callback
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        callback_url,
                        json={
                            "task_id": task_id,
                            "status": "failed",
                            "error": str(e),
                        },
                    )
            except Exception as cb_error:
                logger.error(f"Failed to send failure callback: {cb_error}")


@app.post("/task/submit", response_model=TaskSubmitResponse)
async def task_submit(request: TaskSubmitRequest) -> TaskSubmitResponse:
    """Handle task submission from scheduler.

    This is the endpoint the scheduler uses to dispatch tasks to instances.
    It accepts the task, starts background processing, and returns immediately.

    Args:
        request: TaskSubmitRequest from scheduler

    Returns:
        TaskSubmitResponse indicating acceptance
    """
    global _stats
    _stats["requests_received"] += 1

    # Extract sleep_time from task_input
    sleep_time = request.task_input.get("sleep_time", 1.0)
    if isinstance(sleep_time, str):
        try:
            sleep_time = float(sleep_time)
        except ValueError:
            sleep_time = 1.0

    # Clamp sleep_time to reasonable bounds
    sleep_time = max(0.0, min(300.0, float(sleep_time)))

    logger.debug(
        f"Task submitted: task_id={request.task_id}, "
        f"model_id={request.model_id}, sleep_time={sleep_time}s"
    )

    # Start background task processing
    start_time = time.time()
    asyncio.create_task(
        _process_task_and_callback(
            task_id=request.task_id,
            sleep_time=sleep_time,
            callback_url=request.callback_url,
            start_time=start_time,
        )
    )

    # Return immediately - task will be processed in background
    return TaskSubmitResponse(
        success=True,
        message="Task accepted for processing",
        task_id=request.task_id,
    )


if __name__ == "__main__":
    logger.info(f"Starting PyLet Sleep Model on port {PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level=LOG_LEVEL.lower(),
    )
