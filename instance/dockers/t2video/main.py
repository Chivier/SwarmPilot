"""
LLM Service Large Model Container

A FastAPI service that loads a model using sglang and provides inference endpoints.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from argparse import ArgumentParser

import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler


parser = ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = os.getenv("MODEL_ID", "llm_service_large_model")
INSTANCE_ID = os.getenv("INSTANCE_ID", "unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MODEL_PATH = os.getenv("MODEL_MODEL_PATH")

# Model state
pipe=None

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(20000 + args.port)

# =============================================================================
# Lifecycle Management
# =============================================================================


# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: startup and shutdown."""
    global pipe

    # Startup
    print(f"Wan Service Large Model Container starting...")
    print(f"Model ID: {MODEL_ID}")
    print(f"Instance ID: {INSTANCE_ID}")
    print(f"Log Level: {LOG_LEVEL}")


    # Check if MODEL_PATH is set
    if not MODEL_PATH:
        raise RuntimeError("MODEL_MODEL_PATH environment variable is not set")

    # if not os.path.exists(MODEL_PATH):
    #     raise RuntimeError(f"Model path does not exist: {MODEL_PATH}")

    # Load model with sglang
    print(f"Loading model from: {MODEL_PATH}")
    
    vae = AutoencoderKLWan.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_PATH, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    
    print("T2Video Service is running")

    yield

    # Shutdown
    print("T2Video Service  is shutting down...")
    

    pipe = None
    print("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=f"LLM Service Large Model Container - {MODEL_ID}",
    description="LLM service container with sglang model loading",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Request/Response Models
# =============================================================================

class InferenceRequest(BaseModel):
    """Request schema for inference"""
    prompt: str = Field(
        ...,
        description="Input sentence to send to the LLM",
        min_length=1
    )
    negative_prompt: str = Field(
        ...,
        description="Negative prompt for video generation",
        min_length=1
    )
    frames: int = Field(
        ...,
        description="frames of generated video"
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
    Execute inference - send sentence to LLM and return the result.

    Args:
        request: InferenceRequest with sentence

    Returns:
        InferenceResponse with the LLM result and execution details
    """
    if not pipe:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # Record start time
        start_time = time.time()
        start_timestamp = int(start_time)

        generator = torch.Generator(device="cuda").manual_seed(42)

        result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=480,
                width=832,
                num_frames=request.frames,
                guidance_scale=5.0,
                num_inference_steps=10,
                generator=generator
        ).frames[0]

        # Extract the generated text from the result dict
        # export_to_video(result, "output.mp4", fps=16)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build result
        result = {
            "input": f"positive: {request.prompt}, negative: {request.negative_prompt}",
            "output": "video",
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

