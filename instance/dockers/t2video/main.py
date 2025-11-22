"""
LLM Service Large Model Container - Optimized for Nvidia H20
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from argparse import ArgumentParser

import torch
from diffusers import AutoencoderKLWan, WanPipeline

# -----------------------------------------------------------------------------
# Optimization: H20 / Hopper Specific Settings
# -----------------------------------------------------------------------------
# 1. Set matmul precision to 'high' or 'medium' to fully utilize Tensor Cores on H20
torch.set_float32_matmul_precision("high")

# 2. Enable cudnn benchmark for faster convolution algorithms (helpful for VAE)
torch.backends.cudnn.benchmark = True

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

# Setup Logger
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger("WanService")

# Model state
pipe = None

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(20000 + args.port)

# =============================================================================
# Optimization Functions
# =============================================================================

def optimize_pipeline(pipeline):
    """Applies torch.compile and memory format optimizations."""
    logger.info("Applying optimizations (Channels Last + Torch Compile)...")
    
    # 1. Memory Format: Channels Last (Faster for CNNs/VAE on NV GPUs)
    pipeline.vae.to(memory_format=torch.channels_last)
    # Transformer is mostly linear, but channels_last doesn't hurt and can help specific layers
    pipeline.transformer.to(memory_format=torch.channels_last)

    # 2. Torch Compile
    # mode="max-autotune": Best performance but longest startup time. 
    # For a long-running service on H20, this is worth it.
    # fullgraph=False: Diffusers pipelines have dynamic control flow, so fullgraph is hard.
    
    # Compile the Diffusion Transformer
    # dynamic=True handles variable input sizes (e.g., varying frames/aspect ratios)
    pipeline.transformer = torch.compile(
        pipeline.transformer, 
        mode="max-autotune", 
        fullgraph=False,
        dynamic=True 
    )

    # Compile the VAE Decoder
    pipeline.vae.decoder = torch.compile(
        pipeline.vae.decoder, 
        mode="max-autotune", 
        fullgraph=False
    )
    
    return pipeline

def warmup_pipeline(pipeline):
    """Runs a dummy inference to trigger JIT compilation before traffic hits."""
    logger.info("Starting Warmup (compiling graphs)...")
    start = time.time()
    try:
        # Run a small/minimal generation
        pipeline(
            prompt="warmup",
            negative_prompt="",
            height=480,
            width=832,
            num_frames=33, # Typical frame count
            num_inference_steps=1, # Minimal steps to trigger graph capture
            guidance_scale=5.0
        )
        torch.cuda.synchronize()
        logger.info(f"Warmup complete in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")

# =============================================================================
# Lifecycle Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: startup and shutdown."""
    global pipe

    # Startup
    logger.info(f"Wan Service Large Model Container starting...")
    logger.info(f"Model ID: {MODEL_ID}")
    logger.info(f"H20 Optimization: Enabled")

    if not MODEL_PATH:
        raise RuntimeError("MODEL_MODEL_PATH environment variable is not set")

    # Load model
    logger.info(f"Loading model from: {MODEL_PATH}")
    
    # Optimization: Load directly to device if possible (but diffusers usually needs CPU first)
    # H20 has massive memory, so we don't need cpu_offload. 
    # We use bfloat16 which is native and fast on H20.
    vae = AutoencoderKLWan.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.float32)
    
    pipe = WanPipeline.from_pretrained(
        MODEL_PATH, 
        vae=vae, 
        torch_dtype=torch.bfloat16
    )
    
    # Move to GPU explicitly. Do NOT use enable_model_cpu_offload on H20 for this small model.
    pipe.to("cuda")
    
    # Apply optimizations
    pipe = optimize_pipeline(pipe)
    
    # Trigger Warmup
    warmup_pipeline(pipe)
    
    logger.info("T2Video Service is running and optimized")

    yield

    # Shutdown
    logger.info("T2Video Service is shutting down...")
    pipe = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=f"LLM Service Large Model Container - {MODEL_ID}",
    description="LLM service container with Wan2.1 optimizations",
    version="1.0.0",
    lifespan=lifespan,
)

# =============================================================================
# Request/Response Models
# =============================================================================

class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = Field(..., min_length=1)
    frames: int = Field(..., description="frames of generated video")

class InferenceResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    execution_time: float
    start_timestamp: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# =============================================================================
# Endpoints
# =============================================================================

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest) -> InferenceResponse:
    if not pipe:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()
        start_timestamp = int(start_time)

        # Use torch.inference_mode() for slight overhead reduction over no_grad
        with torch.inference_mode():
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=480,
                width=832,
                num_frames=request.frames,
                guidance_scale=5.0,
            ).frames[0]

        # GPU Sync for accurate timing measurement
        torch.cuda.synchronize()
        execution_time = time.time() - start_time

        result_data = {
            "input": f"positive: {request.prompt}, negative: {request.negative_prompt}",
            "output": "video",
            "model_id": MODEL_ID,
            "instance_id": INSTANCE_ID,
            "start_time": start_time
        }

        return InferenceResponse(
            success=True,
            result=result_data,
            execution_time=execution_time,
            start_timestamp=start_timestamp
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if not pipe:
        return HealthResponse(status="unhealthy", model_loaded=False)
    return HealthResponse(status="healthy", model_loaded=True)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level=LOG_LEVEL.lower()
    )