"""Minimal OpenAI-style image server around a ``diffusers`` pipeline.

Launched by :class:`swarmpilot.engines.diffusion.DiffusersAdapter`:

    python -m swarmpilot.engines.servers.diffusers_server --model M --port P

Routes:
    POST /v1/images/generations   OpenAI images request; answers b64 PNGs.
    GET  /health                  200 once the pipeline is loaded.
    GET  /version                 {"version": <diffusers version>}

The pipeline runs one request at a time in a worker thread, so the event
loop keeps answering health checks during generation. ``--dry-run`` skips
the model and returns blank images after a delay proportional to steps and
pixels, for wiring tests on machines without a GPU.

Needs ``pip install "swarmpilot[diffusion]"`` unless ``--dry-run``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from swarmpilot.engines.diffusion import (
    DEFAULT_STEPS,
    ImageGenerationRequest,
    diffusion_features,
    parse_size,
)

# 1x1 transparent PNG, returned in dry-run mode.
_BLANK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

Generate = Callable[[ImageGenerationRequest, int], list[bytes]]


def _load_pipeline(model: str, device: str | None) -> tuple[Generate, str]:
    import diffusers
    import torch
    from diffusers import AutoPipelineForText2Image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype)
    pipe = pipe.to(device)

    def generate(req: ImageGenerationRequest, steps: int) -> list[bytes]:
        width, height = parse_size(req.model_dump())
        kwargs: dict[str, Any] = {
            "prompt": req.prompt,
            "num_inference_steps": steps,
            "num_images_per_prompt": max(1, req.n),
            "width": width,
            "height": height,
        }
        if req.guidance_scale is not None:
            kwargs["guidance_scale"] = req.guidance_scale
        if req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt
        if req.seed is not None:
            kwargs["generator"] = torch.Generator(device).manual_seed(req.seed)
        images = pipe(**kwargs).images
        out = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out

    return generate, diffusers.__version__


def _dry_run_pipeline(ms_per_step_mp: float) -> tuple[Generate, str]:
    def generate(req: ImageGenerationRequest, steps: int) -> list[bytes]:
        f = diffusion_features(req.model_dump(), steps)
        cost = f["steps"] * f["megapixels"] * f["n_images"]
        time.sleep(ms_per_step_mp * cost / 1000)
        return [_BLANK_PNG] * int(f["n_images"])

    return generate, "0.0.0-dryrun"


def create_app(
    model: str,
    default_steps: int = DEFAULT_STEPS,
    loader: Callable[[], tuple[Generate, str]] | None = None,
) -> FastAPI:
    """Build the server app.

    Args:
        model: Model id reported in responses.
        default_steps: Steps when a request sets none.
        loader: Returns the generate function and version; defaults to
            loading ``model`` with diffusers.

    Returns:
        The FastAPI app.
    """
    state: dict[str, Any] = {}
    load = loader or (lambda: _load_pipeline(model, None))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["generate"], state["version"] = await asyncio.to_thread(load)
        state["lock"] = asyncio.Lock()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        if "generate" not in state:
            raise HTTPException(503, "loading")
        return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"version": state.get("version", "unknown")}

    @app.post("/v1/images/generations")
    async def generations(body: dict[str, Any]) -> dict[str, Any]:
        try:
            req = ImageGenerationRequest.model_validate(body)
        except ValidationError as e:
            raise HTTPException(400, str(e)) from e
        steps = req.num_inference_steps or default_steps
        async with state["lock"]:
            images = await asyncio.to_thread(state["generate"], req, steps)
        return {
            "created": int(time.time()),
            "model": req.model or model,
            "data": [
                {"b64_json": base64.b64encode(png).decode()} for png in images
            ],
        }

    return app


def main() -> None:
    """Serve one model until interrupted."""
    import uvicorn

    parser = argparse.ArgumentParser(description="diffusers image server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--default-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No model; sleep in proportion to steps x megapixels.",
    )
    parser.add_argument(
        "--dry-run-ms",
        type=float,
        default=20.0,
        help="Dry-run milliseconds per step per megapixel.",
    )
    args = parser.parse_args()

    if args.dry_run:
        loader = lambda: _dry_run_pipeline(args.dry_run_ms)  # noqa: E731
    else:
        loader = lambda: _load_pipeline(args.model, args.device)  # noqa: E731
    app = create_app(args.model, args.default_steps, loader)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
