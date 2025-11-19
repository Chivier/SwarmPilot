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

os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "True"

# Import sglang
try:
    import sglang as sgl
except ImportError:
    sgl = None

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
model_loaded = False
runtime: Optional[Any] = None

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(20000 + args.port)

# =============================================================================
# Lifecycle Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: startup and shutdown."""
    global model_loaded, runtime

    # Startup
    print(f"LLM Service Large Model Container starting...")
    print(f"Model ID: {MODEL_ID}")
    print(f"Instance ID: {INSTANCE_ID}")
    print(f"Log Level: {LOG_LEVEL}")

    # Check if sglang is available
    if sgl is None:
        raise RuntimeError("sglang is not installed. Please install it with: pip install sglang")

    # Check if MODEL_PATH is set
    if not MODEL_PATH:
        raise RuntimeError("MODEL_MODEL_PATH environment variable is not set")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model path does not exist: {MODEL_PATH}")

    # Load model with sglang
    print(f"Loading model from: {MODEL_PATH}")
    try:
        # Load model using sglang Engine
        # sgl.Engine() creates an engine instance with the model
        runtime = sgl.Engine(model_path=MODEL_PATH)
        
        model_loaded = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

    yield

    # Shutdown
    print("LLM Service Large Model Container shutting down...")
    
    # Clean up runtime if it exists
    if runtime is not None:
        try:
            # sglang runtime cleanup if needed
            if hasattr(runtime, "shutdown"):
                runtime.shutdown()
            elif hasattr(runtime, "close"):
                runtime.close()
        except Exception as e:
            print(f"Error during runtime cleanup: {e}")
    
    runtime = None
    model_loaded = False
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
    sentence: str = Field(
        ...,
        description="Input sentence to send to the LLM",
        min_length=1
    )
    max_tokens: int = Field(
        default=512,
        description="Maximum number of tokens to generate",
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
    if not model_loaded or runtime is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # Record start time
        start_time = time.time()
        start_timestamp = int(start_time)

        # Run inference with sglang Engine
        # async_generate returns a dict with "text" key
        max_tokens = min(max(0, request.max_tokens), 4096)
        sampleing_parameters = {
            "max_new_tokens": max_tokens,
            "temperature": 0,
            "seed": 114514
        }
        result = await runtime.async_generate(
            [request.sentence],
            sampleing_parameters
        )

        # Extract the generated text from the result dict
        generated_text = result[0].get("text", "")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build result
        result = {
            "input": request.sentence,
            "output": generated_text,
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
        port=args.port,
        log_level=LOG_LEVEL.lower()
    )

