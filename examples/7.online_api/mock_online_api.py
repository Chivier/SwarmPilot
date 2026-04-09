"""Mock online API servers for the multi-provider example.

Starts 9 lightweight FastAPI instances (3 models x 3 providers),
each simulating an OpenAI-compatible chat/completions endpoint
with provider-specific latency profiles.

Port layout (provider x model):
                       Together    Fireworks   Lepton
  Qwen3-8B              9100        9101       9102
  Qwen3-Next-80B-A3B    9110        9111       9112
  Gemma4-41B             9120        9121       9122

Each provider has a distinct latency profile so the Predictor can
learn to prefer faster providers under load.

Usage:
    python mock_online_api.py                  # all 9 servers
    python mock_online_api.py qwen3_8b         # Qwen3-8B only (3 servers)
    python mock_online_api.py qwen3_next       # Qwen3-Next only
    python mock_online_api.py gemma4           # Gemma4 only
"""

import asyncio
import multiprocessing
import random
import sys
import time
import uuid

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# ── Endpoint definitions ─────────────────────────────────────────
# Each provider has a different latency profile for the SAME model.
# Together: fastest on average, low variance
# Fireworks: medium latency, medium variance
# Lepton:   slower average, higher variance (simulates burst throttling)

ENDPOINTS = {
    # ── Qwen3-8B ─────────────────────────────────────────────────
    "together-qwen3-8b": {
        "port": 9100,
        "group": "qwen3_8b",
        "model": "Qwen/Qwen3-8B",
        "provider": "Together AI",
        "delay": (0.08, 0.20),
    },
    "fireworks-qwen3-8b": {
        "port": 9101,
        "group": "qwen3_8b",
        "model": "Qwen/Qwen3-8B",
        "provider": "Fireworks AI",
        "delay": (0.12, 0.30),
    },
    "lepton-qwen3-8b": {
        "port": 9102,
        "group": "qwen3_8b",
        "model": "Qwen/Qwen3-8B",
        "provider": "Lepton AI",
        "delay": (0.15, 0.45),
    },
    # ── Qwen3-Next-80B-A3B ───────────────────────────────────────
    "together-qwen3-next": {
        "port": 9110,
        "group": "qwen3_next",
        "model": "Qwen/Qwen3-Next-80B-A3B",
        "provider": "Together AI",
        "delay": (0.10, 0.25),
    },
    "fireworks-qwen3-next": {
        "port": 9111,
        "group": "qwen3_next",
        "model": "Qwen/Qwen3-Next-80B-A3B",
        "provider": "Fireworks AI",
        "delay": (0.15, 0.40),
    },
    "lepton-qwen3-next": {
        "port": 9112,
        "group": "qwen3_next",
        "model": "Qwen/Qwen3-Next-80B-A3B",
        "provider": "Lepton AI",
        "delay": (0.20, 0.60),
    },
    # ── Gemma4-41B ───────────────────────────────────────────────
    "together-gemma4": {
        "port": 9120,
        "group": "gemma4",
        "model": "google/gemma-4-41b-it",
        "provider": "Together AI",
        "delay": (0.12, 0.30),
    },
    "fireworks-gemma4": {
        "port": 9121,
        "group": "gemma4",
        "model": "google/gemma-4-41b-it",
        "provider": "Fireworks AI",
        "delay": (0.18, 0.45),
    },
    "lepton-gemma4": {
        "port": 9122,
        "group": "gemma4",
        "model": "google/gemma-4-41b-it",
        "provider": "Lepton AI",
        "delay": (0.25, 0.70),
    },
}


def create_app(name: str, spec: dict) -> FastAPI:
    """Create a FastAPI app simulating one online API endpoint.

    Args:
        name: Endpoint name (e.g. "together-llama").
        spec: Dict with model, delay, provider, port.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title=f"Mock {spec['provider']} — {spec['model']}")
    model = spec["model"]
    provider = spec["provider"]
    delay_min, delay_max = spec["delay"]

    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check."""
        return JSONResponse({"status": "healthy", "model": model, "provider": provider})

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        """Simulate chat completion with provider-specific latency.

        Validates Bearer token and returns an OpenAI-compatible
        response after a simulated delay.
        """
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid Authorization header",
            )

        body = await request.json()
        messages = body.get("messages", [])
        prompt_text = " ".join(
            m.get("content", "")
            for m in messages
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        )
        prompt_tokens = max(1, len(prompt_text.split()))

        # Delay scales slightly with input length
        scale = 1.0 + len(prompt_text) / 5000.0
        delay = random.uniform(delay_min, delay_max) * scale
        await asyncio.sleep(delay)

        completion_tokens = random.randint(20, 150)

        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                f"[{provider}] Response to: " f"{prompt_text[:80]}..."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    return app


def run_server(name: str, spec: dict) -> None:
    """Run a single mock API server (blocking).

    Args:
        name: Endpoint name.
        spec: Endpoint specification dict.
    """
    app = create_app(name, spec)
    uvicorn.run(app, host="0.0.0.0", port=spec["port"], log_level="warning")


def main() -> None:
    """Start mock servers for the requested model group(s).

    Usage:
        python mock_online_api.py              # all 9 endpoints
        python mock_online_api.py qwen3_8b    # Qwen3-8B only (3 providers)
        python mock_online_api.py qwen3_next  # Qwen3-Next only
        python mock_online_api.py gemma4      # Gemma4 only
    """
    group_filter = sys.argv[1] if len(sys.argv) > 1 else None

    targets = {
        name: spec
        for name, spec in ENDPOINTS.items()
        if group_filter is None or spec["group"] == group_filter
    }

    if not targets:
        print(f"Unknown group: {group_filter}")
        print("Available: qwen3_8b, qwen3_next, gemma4 (or omit for all)")
        sys.exit(1)

    print(f"Starting {len(targets)} mock online API server(s):\n")
    print(f"  {'NAME':<22s} {'PORT':>5s}  {'PROVIDER':<14s} {'MODEL':<42s} DELAY")
    print(f"  {'─'*22} {'─'*5}  {'─'*14} {'─'*42} {'─'*12}")
    for name, spec in targets.items():
        print(
            f"  {name:<22s} {spec['port']:>5d}  {spec['provider']:<14s} "
            f"{spec['model']:<42s} "
            f"{spec['delay'][0]:.2f}-{spec['delay'][1]:.2f}s"
        )
    print()

    processes = []
    for name, spec in targets.items():
        p = multiprocessing.Process(target=run_server, args=(name, spec), daemon=True)
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutting down mock servers...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
