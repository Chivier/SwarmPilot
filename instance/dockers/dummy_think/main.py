"""
Sleep Model Container

A simple example model container that sleeps for a specified duration
and returns the sleep time. Useful for testing and demonstration.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Union
import random

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = os.getenv("MODEL_ID", "sleep_model")
INSTANCE_ID = os.getenv("INSTANCE_ID", "unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DATA_FILE = Path("data/dr_query.json")

# =============================================================================
# Application State
# =============================================================================

# Model state
model_loaded = False
# Data loaded from dr_query.json
dr_query_data: List[float] = []


# =============================================================================
# Lifespan Context Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown).

    This replaces the deprecated @app.on_event("startup") and
    @app.on_event("shutdown") decorators.

    Yields:
        Control to the application during its lifetime.
    """
    global model_loaded, dr_query_data

    # Startup
    print(f"Sleep Model Container starting...")
    print(f"Model ID: {MODEL_ID}")
    print(f"Instance ID: {INSTANCE_ID}")
    print(f"Log Level: {LOG_LEVEL}")

    # Load dr_query.json data
    try:
        if DATA_FILE.exists():
            print(f"Loading data from {DATA_FILE}...")
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                dr_query_data = json.load(f)
            print(f"Loaded {len(dr_query_data)} data points from {DATA_FILE}")
        else:
            print(f"Warning: {DATA_FILE} not found, using empty data")
            dr_query_data = []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse {DATA_FILE}: {e}")
        dr_query_data = []
    except Exception as e:
        print(f"Error: Failed to load {DATA_FILE}: {e}")
        dr_query_data = []

    # Simulate model loading
    print("Loading model...")
    await asyncio.sleep(1)

    model_loaded = True
    print("Model loaded successfully!")

    yield  # Application runs here

    # Shutdown
    print("Sleep Model Container shutting down...")
    model_loaded = False
    dr_query_data = []
    print("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=f"Sleep Model Container - {MODEL_ID}",
    description="Example model container that sleeps for a specified duration",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Request/Response Models
# =============================================================================

class InferenceRequest(BaseModel):
    """Request schema for inference.

    Attributes:
        sleep_time: Time to sleep in seconds (0-600). Also accepts alias
            'sleep_duration' for backward compatibility.
        fanout_num: Number of next models to fan out to.
    """
    # Accept both 'sleep_time' and 'sleep_duration' as field names
    sleep_time: float = Field(
        ...,
        description="Time to sleep in seconds",
        ge=0,
        le=600,
        alias="sleep_duration"
    )
    fanout_num: int = Field(
        ...,
        description="Number of next models to fan out to"
    )

    class Config:
        populate_by_name = True  # Allow both field name and alias


class InferenceResponse(BaseModel):
    """Response schema for inference.

    Attributes:
        success: Whether the inference completed successfully.
        result: Inference result data. Can be either a single dictionary or
            a list of dictionaries for batch results.
        execution_time: Total execution time in seconds.
        start_timestamp: Unix timestamp when inference started.
    """
    success: bool
    result: Union[Dict[str, Any], List[Dict[str, Any]]]
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
        results = []
        for _ in range(request.fanout_num):
            results.append({
                "sleep_time": request.sleep_time,
                "actual_sleep_time": execution_time,
                "model_id": MODEL_ID,
                "instance_id": INSTANCE_ID,
                "message": f"Slept for {execution_time:.3f} seconds",
                "next_input": {
                    "sleep_time": random.choice(dr_query_data)
                }
            })

        return InferenceResponse(
            success=True,
            result=results,
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
# Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=LOG_LEVEL.lower()
    )
