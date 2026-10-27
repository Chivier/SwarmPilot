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
A body may override them with ``fake_latency_ms`` and ``fake_tokens``, ask
for an error with ``fake_status``, and keep a stream open for
``fake_hold_open_ms`` after its ``[DONE]`` (or ``[[hold_open_ms=N]]`` in a
message, which survives API style translation).

For black-box tests of whatever sits in front of it, the engine counts
what it sees: ``GET /_stats`` returns requests per path and the most
requests it ever had in flight at once, and ``POST /_reset`` clears that.

Usage:
    uv run python tests/perf/fake_engine.py --port 18001 --latency-ms 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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


_STATS: dict = {"in_flight": 0, "max_in_flight": 0, "requests": {}}


def _enter(path: str) -> None:
    _STATS["in_flight"] += 1
    _STATS["max_in_flight"] = max(_STATS["max_in_flight"], _STATS["in_flight"])
    _STATS["requests"][path] = _STATS["requests"].get(path, 0) + 1


def _leave() -> None:
    _STATS["in_flight"] -= 1


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
    try:
        async for chunk in _chunks(body, chat, tokens, latency_ms):
            yield chunk
    finally:
        _leave()


async def _chunks(body: dict, chat: bool, tokens: int, latency_ms: float):
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
    await asyncio.sleep(_hold_open_ms(body) / 1000)


def _hold_open_ms(body: dict) -> float:
    if "fake_hold_open_ms" in body:
        return float(body["fake_hold_open_ms"])
    m = re.search(r"\[\[hold_open_ms=(\d+)\]\]", json.dumps(body))
    return float(m.group(1)) if m else 0.0


async def completions(request: Request) -> Response:
    """Serve chat and text completions, streaming when asked."""
    body = await _read_json(request)
    chat = request.url.path.endswith("chat/completions")
    latency_ms = float(body.get("fake_latency_ms", LATENCY_MS))
    tokens = max(1, int(body.get("fake_tokens", TOKENS)))
    _enter(request.url.path)
    if body.get("stream") and "fake_status" not in body:
        return StreamingResponse(
            _stream(body, chat, tokens, latency_ms),
            media_type="text/event-stream",
        )
    try:
        if latency_ms:
            await asyncio.sleep(latency_ms / 1000)
    finally:
        _leave()
    if "fake_status" in body:
        return JSONResponse(
            {"error": {"message": "fake error", "type": "fake"}},
            status_code=int(body["fake_status"]),
        )
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


async def stats(_: Request) -> Response:
    """Report requests per path and peak concurrency."""
    return JSONResponse(_STATS)


async def reset(_: Request) -> Response:
    """Clear the counters (in-flight requests keep counting down)."""
    _STATS["max_in_flight"] = _STATS["in_flight"]
    _STATS["requests"] = {}
    return JSONResponse({"ok": True})


async def models(request: Request) -> Response:
    """List the single fake model."""
    _enter(request.url.path)
    _leave()
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
        Route("/_stats", stats),
        Route("/_reset", reset, methods=["POST"]),
    ]
)


def main() -> None:
    """Run the fake engine under uvicorn."""
    global LATENCY_MS, TOKENS, MS_PER_STEP
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--latency-ms", type=float, default=LATENCY_MS)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--ms-per-step", type=float, default=MS_PER_STEP)
    args = parser.parse_args()

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
