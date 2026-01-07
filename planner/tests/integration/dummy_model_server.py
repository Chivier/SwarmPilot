#!/usr/bin/env python3
"""Dummy model server for PyLet integration testing.

This server simulates a vLLM/sglang-like model backend for testing
the PyLet + Planner optimization loop without requiring actual models.

Reads $PORT from environment (set by PyLet) and simulates vLLM endpoints.

Environment Variables:
    PORT: Port to listen on (required, set by PyLet via $PORT substitution)
    MODEL_ID: Model identifier to report (default: "dummy-model")
    THROUGHPUT: Simulated requests/second for completions (default: 1.0)

Endpoints:
    GET  /health        - Health check (vLLM-compatible)
    GET  /v1/models     - List models (OpenAI-compatible)
    POST /v1/completions - Simulated completions with throughput delay

Example:
    PORT=8000 MODEL_ID=model-a THROUGHPUT=10.0 python dummy_model_server.py

PYLET-013: PyLet Optimizer Integration E2E Test
"""

import os
import sys
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Read configuration from environment
PORT = int(os.environ.get("PORT", 8000))
MODEL_ID = os.environ.get("MODEL_ID", "dummy-model")
THROUGHPUT = float(os.environ.get("THROUGHPUT", "1.0"))

app = FastAPI(
    title=f"Dummy Model Server ({MODEL_ID})",
    description="Simulates vLLM/sglang model backend for PyLet integration testing",
    version="1.0.0",
)

# Track request statistics
stats = {
    "start_time": datetime.now().isoformat(),
    "request_count": 0,
    "total_tokens_generated": 0,
}


@app.get("/health")
async def health():
    """Health check endpoint (vLLM-compatible).

    Returns:
        Dict with health status and model info.
    """
    return {
        "status": "healthy",
        "model_id": MODEL_ID,
        "throughput": THROUGHPUT,
        "port": PORT,
    }


@app.get("/v1/models")
async def list_models():
    """List models endpoint (OpenAI-compatible).

    Returns:
        Dict with model list.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "swarmpilot-test",
                "root": MODEL_ID,
                "parent": None,
            }
        ],
    }


class CompletionRequest(BaseModel):
    """Request model for completions endpoint."""

    prompt: str = "Hello"
    max_tokens: int = 16
    temperature: float = 0.7


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Simulated completions endpoint with throughput simulation.

    Sleeps for 1/THROUGHPUT seconds to simulate processing time,
    then returns a dummy completion.

    Args:
        request: Completion request with prompt and params.

    Returns:
        Dict with completion response.
    """
    stats["request_count"] += 1

    # Simulate processing time based on throughput
    processing_time = 1.0 / THROUGHPUT
    time.sleep(processing_time)

    # Generate dummy tokens
    generated_tokens = min(request.max_tokens, 16)
    stats["total_tokens_generated"] += generated_tokens

    return {
        "id": f"cmpl-{stats['request_count']:08d}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "text": f"Hello from {MODEL_ID}! Request #{stats['request_count']}",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": generated_tokens,
            "total_tokens": len(request.prompt.split()) + generated_tokens,
        },
    }


@app.get("/stats")
async def get_stats():
    """Get server statistics.

    Returns:
        Dict with request counts and uptime.
    """
    return {
        **stats,
        "model_id": MODEL_ID,
        "throughput_config": THROUGHPUT,
        "port": PORT,
    }


def main():
    """Start the dummy model server."""
    print(f"Starting dummy model server on port {PORT}")
    print(f"  MODEL_ID: {MODEL_ID}")
    print(f"  THROUGHPUT: {THROUGHPUT} req/s")
    print(f"  Health check: http://0.0.0.0:{PORT}/health")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
