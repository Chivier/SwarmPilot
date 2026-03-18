"""Minimal mock vLLM server for single-model example.

Simulates LLM inference with a random delay. No streaming,
no Pydantic models, no registration logic. Designed to be
started by deploy_model.sh and registered externally.

Environment Variables:
    PORT: Listen port (default: 8100)
    MODEL_ID: Model identifier (default: Qwen/Qwen3-8B-VL)
"""

import asyncio
import os
import random
import time
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

PORT = int(os.environ.get("PORT", "8100"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B-VL")

app = FastAPI(title=f"Mock vLLM - {MODEL_ID}")


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "healthy", "model": MODEL_ID})


@app.post("/v1/completions")
async def completions() -> JSONResponse:
    """Simulate inference with 0.1-0.5s random delay."""
    delay = random.uniform(0.1, 0.5)
    await asyncio.sleep(delay)

    return JSONResponse(
        {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [
                {
                    "text": f"Mock response from {MODEL_ID}",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "model": MODEL_ID,
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            },
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
