"""Mock vLLM Server with OpenAI-Compatible API.

This module provides a mock vLLM server that exposes OpenAI-compatible endpoints.
It simulates LLM inference by sleeping for a configurable random duration,
then returns a response in OpenAI chat completion format.

Runtime characteristics are configured per model with distinct distributions:
- llm_fast:   Exponential dist, mean ~100ms  (quick responses, some variance)
- llm_medium: Log-normal dist, mean ~500ms   (moderate with right-skew)
- llm_slow:   Gamma dist, mean ~2000ms       (longer with heavy tail)

Endpoints:
- POST /v1/chat/completions - OpenAI-compatible chat completion
- POST /task/submit - Scheduler task submission endpoint
- GET /health - Health check
- GET /v1/models - List available models

Environment Variables:
- PORT: Port to run on (required, set by PyLet)
- MODEL_ID: Model identifier (determines runtime characteristics)
- INSTANCE_ID: Instance identifier
- SCHEDULER_URL: Scheduler URL for registration (optional)
- LOG_LEVEL: Log level (default: INFO)
- BASE_LATENCY_MS: Base latency in ms (default: 100)
"""

import asyncio
import json
import math
import os
import random
import signal
import socket
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field


# Configuration from environment
PORT = int(os.environ.get("PORT", "8000"))
MODEL_ID = os.environ.get("MODEL_ID", "llm_fast")
INSTANCE_ID = os.environ.get("INSTANCE_ID", f"{MODEL_ID}-{uuid.uuid4().hex[:8]}")
SCHEDULER_URL = os.environ.get("SCHEDULER_URL", "")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
BASE_LATENCY_MS = float(os.environ.get("BASE_LATENCY_MS", "100"))


@dataclass
class LatencyDistribution:
    """Configuration for latency distribution per model."""

    mean_ms: float  # Target mean latency in ms
    distribution: str  # "exponential", "lognormal", "gamma"
    # Distribution-specific parameters
    shape: float = 2.0  # For gamma distribution
    sigma: float = 0.3  # For log-normal distribution

    def sample(self) -> float:
        """Sample a latency value from this distribution.

        Returns:
            Latency in seconds
        """
        if self.distribution == "exponential":
            # Exponential: mean = 1/lambda, so lambda = 1/mean
            # Good for fast responses with occasional spikes
            latency_ms = random.expovariate(1.0 / self.mean_ms)
            # Cap at 3x mean to avoid extreme outliers
            latency_ms = min(latency_ms, self.mean_ms * 3)

        elif self.distribution == "lognormal":
            # Log-normal: right-skewed, good for medium latencies
            # mu and sigma parameterize the underlying normal distribution
            # mean = exp(mu + sigma^2/2), so mu = ln(mean) - sigma^2/2
            mu = math.log(self.mean_ms) - (self.sigma ** 2) / 2
            latency_ms = random.lognormvariate(mu, self.sigma)
            # Cap at 4x mean
            latency_ms = min(latency_ms, self.mean_ms * 4)

        elif self.distribution == "gamma":
            # Gamma: flexible, good for longer latencies with heavy tail
            # mean = shape * scale, so scale = mean / shape
            scale = self.mean_ms / self.shape
            latency_ms = random.gammavariate(self.shape, scale)
            # Cap at 3x mean
            latency_ms = min(latency_ms, self.mean_ms * 3)

        else:
            # Fallback: uniform with +/- 20% variation
            latency_ms = self.mean_ms * random.uniform(0.8, 1.2)

        # Ensure minimum latency of 10ms
        latency_ms = max(latency_ms, 10.0)
        return latency_ms / 1000.0  # Convert to seconds


# Latency distributions for each model (ratio 1:5:20)
MODEL_LATENCY_DISTRIBUTIONS: dict[str, LatencyDistribution] = {
    "llm_fast": LatencyDistribution(
        mean_ms=100.0,
        distribution="exponential",
    ),
    "llm_medium": LatencyDistribution(
        mean_ms=500.0,
        distribution="lognormal",
        sigma=0.4,  # Moderate variance
    ),
    "llm_slow": LatencyDistribution(
        mean_ms=2000.0,
        distribution="gamma",
        shape=3.0,  # Shape > 1 gives bell-like curve with right tail
    ),
}

# Legacy multipliers for backward compatibility
MODEL_LATENCY_MULTIPLIERS = {
    "llm_fast": 1.0,      # ~100ms average
    "llm_medium": 5.0,    # ~500ms average
    "llm_slow": 20.0,     # ~2000ms average
}

# Get hostname for endpoint construction
HOSTNAME = socket.gethostname()
try:
    HOST_IP = socket.gethostbyname(HOSTNAME)
except socket.gaierror:
    HOST_IP = "127.0.0.1"


# Statistics tracking
_stats = {
    "requests_received": 0,
    "requests_completed": 0,
    "requests_failed": 0,
    "total_latency_ms": 0.0,
    "start_time": None,
}


def get_model_latency(model_id: str | None = None) -> float:
    """Get the simulated latency for a model.

    Args:
        model_id: Model ID to get latency for. If None, uses global MODEL_ID.

    Returns:
        Latency in seconds sampled from the model's distribution
    """
    target_model = model_id or MODEL_ID

    # Use distribution-based sampling if available
    if target_model in MODEL_LATENCY_DISTRIBUTIONS:
        distribution = MODEL_LATENCY_DISTRIBUTIONS[target_model]
        return distribution.sample()

    # Fallback to legacy multiplier-based approach
    multiplier = MODEL_LATENCY_MULTIPLIERS.get(target_model, 1.0)
    base_latency_s = (BASE_LATENCY_MS * multiplier) / 1000.0
    # Add +/- 20% random variation
    variation = random.uniform(-0.2, 0.2)
    return base_latency_s * (1 + variation)


# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""
    model: str = Field(..., description="Model ID to use")
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = Field(default=False, description="Enable streaming")
    user: str | None = None

    class Config:
        extra = "allow"


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mock-vllm"


class ModelsResponse(BaseModel):
    """List models response."""
    object: str = "list"
    data: list[ModelInfo]


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
    """Register this instance with the scheduler."""
    if not SCHEDULER_URL:
        logger.info("SCHEDULER_URL not set, skipping registration")
        return True

    endpoint = f"http://{HOST_IP}:{PORT}"
    registration_data = {
        "instance_id": INSTANCE_ID,
        "model_id": MODEL_ID,
        "endpoint": endpoint,
        "platform_info": {
            "software_name": "vllm",
            "software_version": "0.4.0",
            "hardware_name": "gpu",
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
                logger.success(f"Successfully registered with scheduler")
                return True
            else:
                logger.error(
                    f"Failed to register: HTTP {response.status_code}: {response.text}"
                )
                return False

    except Exception as e:
        logger.error(f"Failed to register with scheduler: {e}")
        return False


async def deregister_from_scheduler() -> bool:
    """Deregister this instance from the scheduler."""
    if not SCHEDULER_URL:
        return True

    logger.info(f"Deregistering from scheduler: instance_id={INSTANCE_ID}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{SCHEDULER_URL}/instance/remove",
                json={"instance_id": INSTANCE_ID},
            )
            return response.status_code == 200

    except Exception as e:
        logger.warning(f"Failed to deregister: {e}")
        return False


# Shutdown handling
_shutdown_event = asyncio.Event()


def handle_sigterm(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    logger.info(f"Received signal {signum}, shutting down...")
    _shutdown_event.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _stats

    logger.info(f"Mock vLLM Server starting...")
    logger.info(f"  Model ID: {MODEL_ID}")
    logger.info(f"  Instance ID: {INSTANCE_ID}")
    logger.info(f"  Port: {PORT}")

    # Log latency distribution info
    if MODEL_ID in MODEL_LATENCY_DISTRIBUTIONS:
        dist = MODEL_LATENCY_DISTRIBUTIONS[MODEL_ID]
        logger.info(f"  Latency Distribution: {dist.distribution}")
        logger.info(f"  Mean Latency: {dist.mean_ms:.0f}ms")
    else:
        logger.info(f"  Base Latency: {BASE_LATENCY_MS}ms")
        logger.info(f"  Latency Multiplier: {MODEL_LATENCY_MULTIPLIERS.get(MODEL_ID, 1.0)}x")

    _stats["start_time"] = time.time()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    # Simulate model loading
    await asyncio.sleep(0.5)
    logger.info("Model loaded successfully")

    # Register with scheduler
    if SCHEDULER_URL:
        for attempt in range(3):
            if await register_with_scheduler():
                break
            if attempt < 2:
                await asyncio.sleep(2)

    logger.success(f"Mock vLLM Server ready on port {PORT}")

    yield

    logger.info("Shutting down...")
    await deregister_from_scheduler()

    uptime = time.time() - _stats["start_time"] if _stats["start_time"] else 0
    logger.info(
        f"Final stats: requests={_stats['requests_completed']}/{_stats['requests_received']}, "
        f"total_latency={_stats['total_latency_ms']:.2f}ms, uptime={uptime:.2f}s"
    )


# FastAPI Application
app = FastAPI(
    title=f"Mock vLLM Server - {MODEL_ID}",
    description="OpenAI-compatible mock vLLM server for E2E testing",
    version="1.0.0",
    lifespan=lifespan,
)


def create_completion_response(
    model: str,
    content: str,
    latency_ms: float,
) -> ChatCompletionResponse:
    """Create an OpenAI-compatible chat completion response."""
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=10,  # Mock values
            completion_tokens=len(content.split()),
            total_tokens=10 + len(content.split()),
        ),
    )


async def stream_completion(model: str, content: str, latency_ms: float):
    """Stream completion response in SSE format."""
    # Send chunks
    chunks = content.split()
    chunk_delay = (latency_ms / 1000.0) / max(len(chunks), 1)

    for i, word in enumerate(chunks):
        chunk_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": word + " "},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        await asyncio.sleep(chunk_delay)

    # Send final chunk
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    global _stats
    _stats["requests_received"] += 1

    try:
        start_time = time.time()

        # Get simulated latency
        latency_s = get_model_latency()
        latency_ms = latency_s * 1000

        # Sleep to simulate inference
        await asyncio.sleep(latency_s)

        # Create response content with the sleep time
        content = (
            f"Mock response from {MODEL_ID} (instance: {INSTANCE_ID}). "
            f"Simulated latency: {latency_ms:.2f}ms"
        )

        execution_time = time.time() - start_time
        _stats["requests_completed"] += 1
        _stats["total_latency_ms"] += execution_time * 1000

        logger.debug(
            f"Chat completion: model={request.model}, latency={execution_time*1000:.2f}ms"
        )

        if request.stream:
            return StreamingResponse(
                stream_completion(request.model, content, latency_ms),
                media_type="text/event-stream",
            )
        else:
            return create_completion_response(request.model, content, latency_ms)

    except asyncio.CancelledError:
        _stats["requests_failed"] += 1
        raise HTTPException(status_code=499, detail="Request cancelled")

    except Exception as e:
        _stats["requests_failed"] += 1
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id=MODEL_ID,
                created=int(_stats["start_time"] or time.time()),
            )
        ]
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_id=MODEL_ID,
        instance_id=INSTANCE_ID,
    )


@app.get("/stats")
async def stats() -> dict[str, Any]:
    """Get instance statistics."""
    uptime = time.time() - _stats["start_time"] if _stats["start_time"] else 0
    return {
        **_stats,
        "uptime_seconds": uptime,
        "model_id": MODEL_ID,
        "instance_id": INSTANCE_ID,
        "port": PORT,
        "latency_multiplier": MODEL_LATENCY_MULTIPLIERS.get(MODEL_ID, 1.0),
    }


async def _process_task_and_callback(
    task_id: str,
    model_id: str,
    messages: list[dict],
    callback_url: str | None,
    start_time: float,
) -> None:
    """Process a task in background and send callback."""
    global _stats

    try:
        # Get simulated latency based on the task's model_id
        # This ensures each model gets its own latency distribution
        latency_s = get_model_latency(model_id)
        latency_ms = latency_s * 1000

        # Sleep to simulate inference
        await asyncio.sleep(latency_s)

        execution_time = time.time() - start_time
        _stats["requests_completed"] += 1
        _stats["total_latency_ms"] += execution_time * 1000

        # Create OpenAI-compatible response content
        response_content = (
            f"Mock response from {MODEL_ID} (instance: {INSTANCE_ID}). "
            f"Simulated latency: {latency_ms:.2f}ms"
        )

        # Create full OpenAI-compatible response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(response_content.split()),
                "total_tokens": 10 + len(response_content.split()),
            },
        }

        logger.debug(
            f"Task {task_id} completed: latency={execution_time*1000:.2f}ms"
        )

        # Send callback
        if callback_url:
            callback_payload = {
                "task_id": task_id,
                "status": "completed",
                "result": openai_response,
                "execution_time_ms": execution_time * 1000,
            }

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(callback_url, json=callback_payload)
                    if response.status_code != 200:
                        logger.warning(
                            f"Callback for {task_id} returned {response.status_code}"
                        )
            except Exception as e:
                logger.error(f"Failed to send callback for {task_id}: {e}")

    except asyncio.CancelledError:
        _stats["requests_failed"] += 1
        logger.warning(f"Task {task_id} cancelled")

    except Exception as e:
        _stats["requests_failed"] += 1
        logger.error(f"Task {task_id} failed: {e}")

        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        callback_url,
                        json={"task_id": task_id, "status": "failed", "error": str(e)},
                    )
            except Exception:
                pass


@app.post("/task/submit", response_model=TaskSubmitResponse)
async def task_submit(request: TaskSubmitRequest) -> TaskSubmitResponse:
    """Handle task submission from scheduler.

    Accepts tasks in scheduler format and processes them as OpenAI requests.
    """
    global _stats
    _stats["requests_received"] += 1

    # Extract messages from task_input
    messages = request.task_input.get("messages", [
        {"role": "user", "content": "Hello"}
    ])

    logger.debug(
        f"Task submitted: task_id={request.task_id}, model_id={request.model_id}"
    )

    # Start background processing
    start_time = time.time()
    asyncio.create_task(
        _process_task_and_callback(
            task_id=request.task_id,
            model_id=request.model_id,
            messages=messages,
            callback_url=request.callback_url,
            start_time=start_time,
        )
    )

    return TaskSubmitResponse(
        success=True,
        message="Task accepted for processing",
        task_id=request.task_id,
    )


if __name__ == "__main__":
    logger.info(f"Starting Mock vLLM Server on port {PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level=LOG_LEVEL.lower(),
    )
