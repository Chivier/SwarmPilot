"""Minimal mock vLLM server for multi-model direct deployment example.

Self-contained FastAPI server simulating an OpenAI-compatible inference
endpoint.  No registration logic — the deploy script handles that.

Environment Variables:
    MODEL_ID: Model identifier (default: "mock-model")
    PORT: Listen port (default: 8100)
"""

import os
import time
import uuid

import uvicorn
from fastapi import FastAPI

MODEL_ID = os.environ.get("MODEL_ID", "mock-model")
PORT = int(os.environ.get("PORT", "8100"))

app = FastAPI(title=f"Mock vLLM - {MODEL_ID}")


@app.get("/health")
async def health() -> dict:
    """Health check — returns 200 when ready."""
    return {"status": "healthy", "model_id": MODEL_ID}


@app.post("/v1/completions")
async def completions(request: dict) -> dict:
    """Return a canned OpenAI-compatible completion response."""
    prompt = request.get("prompt", "")
    max_tokens = request.get("max_tokens", 16)

    content = (
        f"[mock] model={MODEL_ID} "
        f"prompt_len={len(prompt)} max_tokens={max_tokens}"
    )

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "text": content,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": max(len(prompt.split()), 1),
            "completion_tokens": max_tokens,
            "total_tokens": max(len(prompt.split()), 1) + max_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
