"""
T2Img Service - Text-to-Image Generation Container

A FastAPI service that provides FLUX-based text-to-image generation.
This service is used in the type3 workflow (Text2Image+Video) as the C step.

Input:
- sentence: Text prompt for image generation
- width: Image width (e.g., 512 or 1024)
- height: Image height (e.g., 512 or 1024)

Output:
- Generated image reference/metadata
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from argparse import ArgumentParser

import torch
from diffusers import FluxPipeline


parser = ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = os.getenv("MODEL_ID", "t2img")
INSTANCE_ID = os.getenv("INSTANCE_ID", "unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MODEL_PATH = os.getenv("MODEL_MODEL_PATH", "black-forest-labs/FLUX.1-dev")

# Model state
pipe = None


# =============================================================================
# Lifecycle Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: startup and shutdown."""
    global pipe

    # Startup
    print(f"T2Img Service Container starting...")
    print(f"Model ID: {MODEL_ID}")
    print(f"Instance ID: {INSTANCE_ID}")
    print(f"Log Level: {LOG_LEVEL}")

    # Load FLUX model
    print(f"Loading FLUX model from: {MODEL_PATH}")

    try:
        pipe = FluxPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        print("FLUX model loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load FLUX model: {e}")
        print("Service will run but inference will fail")
        pipe = None

    print("T2Img Service is running")

    yield

    # Shutdown
    print("T2Img Service is shutting down...")
    pipe = None
    print("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=f"T2Img Service - {MODEL_ID}",
    description="Text-to-Image generation service using FLUX model",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Request/Response Models
# =============================================================================

class InferenceRequest(BaseModel):
    """Request schema for inference"""
    sentence: str = Field(
        ...,
        description="Text prompt for image generation",
        min_length=1
    )
    width: int = Field(
        default=512,
        description="Image width in pixels (512 or 1024)",
        ge=256,
        le=2048
    )
    height: int = Field(
        default=512,
        description="Image height in pixels (512 or 1024)",
        ge=256,
        le=2048
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
    Execute inference - generate image from text prompt.

    Args:
        request: InferenceRequest with sentence, width, and height

    Returns:
        InferenceResponse with image metadata and execution details
    """
    if not pipe:
        raise HTTPException(
            status_code=503,
            detail="FLUX model not loaded"
        )

    try:
        # Record start time
        start_time = time.time()
        start_timestamp = int(start_time)

        generator = torch.Generator(device="cuda").manual_seed(42)

        # Generate image with FLUX
        # Note: FLUX uses different parameter names than SD
        image = pipe(
            prompt=request.sentence,
            height=request.height,
            width=request.width,
            guidance_scale=3.5,
            num_inference_steps=30,
            generator=generator
        ).images[0]

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build result (don't return actual image bytes, just metadata)
        result = {
            "input": request.sentence,
            "output": f"image_{request.width}x{request.height}",
            "resolution": f"{request.width}x{request.height}",
            "model_id": MODEL_ID,
            "instance_id": INSTANCE_ID,
            "start_time": start_time
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
    if not pipe:
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
        port=args.port,
        log_level=LOG_LEVEL.lower()
    )
