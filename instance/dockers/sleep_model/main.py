"""
Sleep Model Container

A simple example model container that sleeps for a specified duration
and returns the sleep time. Useful for testing and demonstration.
"""

import asyncio
import os
import time
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = os.getenv("MODEL_ID", "sleep_model")
INSTANCE_ID = os.getenv("INSTANCE_ID", "unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=f"Sleep Model Container - {MODEL_ID}",
    description="Example model container that sleeps for a specified duration",
    version="1.0.0",
)

# Model state
model_loaded = False


# =============================================================================
# Request/Response Models
# =============================================================================

class InferenceRequest(BaseModel):
    """Request schema for inference"""
    sleep_time: float = Field(
        ...,
        description="Time to sleep in seconds",
        ge=0,
        le=60
    )


class InferenceResponse(BaseModel):
    """Response schema for inference"""
    success: bool
    result: Dict[str, Any]
    execution_time: float
    start_timestamp: int


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool


# =============================================================================
# Endpoints
# =============================================================================

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest) -> InferenceResponse:
    """
    Execute inference - sleep for the specified duration and return the time.

    Args:
        request: InferenceRequest with sleep_time

    Returns:
        InferenceResponse with the sleep time and execution details
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # Record start time
        start_time = time.time()
        start_timestamp = int(start_time)

        # Sleep for the specified duration
        await asyncio.sleep(request.sleep_time)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build result
        result = {
            "sleep_time": request.sleep_time,
            "actual_sleep_time": execution_time,
            "model_id": MODEL_ID,
            "instance_id": INSTANCE_ID,
            "message": f"Slept for {execution_time:.3f} seconds"
        }

        return InferenceResponse(
            success=True,
            result=result,
            execution_time=execution_time,
            start_timestamp=start_timestamp
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse indicating model status
    """
    if not model_loaded:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True
    )


# =============================================================================
# Lifecycle Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model_loaded

    print(f"Sleep Model Container starting...")
    print(f"Model ID: {MODEL_ID}")
    print(f"Instance ID: {INSTANCE_ID}")
    print(f"Log Level: {LOG_LEVEL}")

    # Simulate model loading
    print("Loading model...")
    await asyncio.sleep(1)

    model_loaded = True
    print("Model loaded successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global model_loaded

    print("Sleep Model Container shutting down...")
    model_loaded = False
    print("Shutdown complete")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=LOG_LEVEL.lower()
    )
