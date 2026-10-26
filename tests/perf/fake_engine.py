"""Fake inference engine for benchmarks and end-to-end tests.

Serves just enough of the OpenAI-compatible surface that vLLM and SGLang
expose, plus an images route for a diffusion-style workload, so that the
same harness can measure SwarmPilot without a GPU:

- ``POST /v1/chat/completions`` and ``POST /v1/completions``: JSON, or SSE
  when the body has ``"stream": true``.
- ``POST /v1/images/generations``: latency scales with ``num_inference_steps``.
- ``GET /health``, ``GET /v1/models``, ``GET /version``.

Latency is controlled per process with ``--latency-ms`` (time to the full
response) and ``--tokens`` (number of SSE chunks, spread over that latency).
A body may override them with ``fake_latency_ms`` and ``fake_tokens``.

Usage:
    uv run python tests/perf/fake_engine.py --port 18001 --latency-ms 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

LATENCY_MS = float(os.getenv("FAKE_ENGINE_LATENCY_MS", "0"))
TOKENS = int(os.getenv("FAKE_ENGINE_TOKENS", "16"))
MODEL = os.getenv("FAKE_ENGINE_MODEL", "fake-model")
VERSION = os.getenv("FAKE_ENGINE_VERSION", "0.0.0")
MS_PER_STEP = float(os.getenv("FAKE_ENGINE_MS_PER_STEP", "0"))


async def _read_json(request: Request) -> dict:
    raw = await request.body()
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _completion(body: dict, chat: bool, tokens: int) -> dict:
    text = "x" * tokens
    choice = (
        {"index": 0, "message": {"role": "assistant", "content": text}}
        if chat
        else {"index": 0, "text": text}
    )
    choice["finish_reason"] = "length"
    return {
        "id": "fake-0",
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()),
        "model": body.get("model", MODEL),
        "choices": [choice],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": tokens,
            "total_tokens": tokens + 1,
        },
    }


async def _stream(body: dict, chat: bool, tokens: int, latency_ms: float):
    gap = latency_ms / 1000 / max(tokens, 1)
    model = body.get("model", MODEL)
    for i in range(tokens):
        if gap:
            await asyncio.sleep(gap)
        delta = {"content": "x"} if chat else None
        choice = (
            {"index": 0, "delta": delta, "finish_reason": None}
            if chat
            else {"index": 0, "text": "x", "finish_reason": None}
        )
        chunk = {"id": f"fake-{i}", "model": model, "choices": [choice]}
        yield f"data: {json.dumps(chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def completions(request: Request) -> Response:
    """Serve chat and text completions, streaming when asked."""
    body = await _read_json(request)
    chat = request.url.path.endswith("chat/completions")
    latency_ms = float(body.get("fake_latency_ms", LATENCY_MS))
    tokens = max(1, int(body.get("fake_tokens", TOKENS)))
    if body.get("stream"):
        return StreamingResponse(
            _stream(body, chat, tokens, latency_ms),
            media_type="text/event-stream",
        )
    if latency_ms:
        await asyncio.sleep(latency_ms / 1000)
    return JSONResponse(_completion(body, chat, tokens))


async def images(request: Request) -> Response:
    """Serve a diffusion-style image request; latency scales with steps."""
    body = await _read_json(request)
    steps = int(body.get("num_inference_steps", 30))
    latency_ms = float(body.get("fake_latency_ms", MS_PER_STEP * steps))
    if latency_ms:
        await asyncio.sleep(latency_ms / 1000)
    n = int(body.get("n", 1))
    return JSONResponse(
        {"created": int(time.time()), "data": [{"b64_json": ""}] * n}
    )


async def health(_: Request) -> Response:
    """Report liveness."""
    return JSONResponse({"status": "ok"})


async def models(_: Request) -> Response:
    """List the single fake model."""
    return JSONResponse(
        {"object": "list", "data": [{"id": MODEL, "object": "model"}]}
    )


async def version(_: Request) -> Response:
    """Report the fake engine version."""
    return JSONResponse({"version": VERSION})


app = Starlette(
    routes=[
        Route("/v1/chat/completions", completions, methods=["POST"]),
        Route("/v1/completions", completions, methods=["POST"]),
        Route("/v1/images/generations", images, methods=["POST"]),
        Route("/health", health),
        Route("/v1/models", models),
        Route("/version", version),
    ]
)


def main() -> None:
    """Run the fake engine under uvicorn."""
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--latency-ms", type=float, default=LATENCY_MS)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--ms-per-step", type=float, default=MS_PER_STEP)
    args = parser.parse_args()

    global LATENCY_MS, TOKENS, MS_PER_STEP
    LATENCY_MS, TOKENS, MS_PER_STEP = (
        args.latency_ms,
        args.tokens,
        args.ms_per_step,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
