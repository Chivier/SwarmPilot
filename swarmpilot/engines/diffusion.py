"""Reference adapter for text-to-image diffusion models.

It serves any Hugging Face ``diffusers`` text-to-image pipeline through
:mod:`swarmpilot.engines.servers.diffusers_server`, an OpenAI-style
``POST /v1/images/generations`` endpoint. Install the server's dependencies
with ``pip install "swarmpilot[diffusion]"``.

Diffusion runtime is close to linear in denoising steps times pixels times
images, so those are the predictor features. Requests carry them either as
the OpenAI ``size`` string (``"1024x768"``) or as ``width`` and ``height``.

Any other server that speaks the same images API can reuse this adapter by
overriding :meth:`DiffusersAdapter.launch_spec`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from swarmpilot.engines.base import (
    PORT,
    Capacity,
    EngineAdapter,
    LaunchSpec,
    RouteSpec,
    WorkloadSpec,
    WorkloadType,
)


class ImageGenerationRequest(BaseModel):
    """OpenAI-style image request, plus the usual diffusers knobs."""

    model_config = ConfigDict(extra="ignore")

    prompt: str
    model: str | None = None
    n: int = 1
    size: str | None = None
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    response_format: str = "b64_json"


DIFFUSION_WORKLOAD = WorkloadSpec(
    type=WorkloadType.DIFFUSION_IMAGE,
    feature_names=("steps", "megapixels", "n_images"),
    latency_metric="e2e",
)

DEFAULT_SIZE = (1024, 1024)
DEFAULT_STEPS = 30


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _area(size: Any) -> int | None:
    if type(size) is str:
        w, _, h = size.lower().partition("x")
        if w.isdigit() and h.isdigit():
            return int(w) * int(h)
    return None


def parse_size(body: Mapping[str, Any]) -> tuple[int, int]:
    """Return (width, height) from ``width``/``height`` or ``size``.

    Args:
        body: Image request body.

    Returns:
        Width and height in pixels, or the default size when absent or
        malformed.
    """
    width, height = _number(body.get("width")), _number(body.get("height"))
    if width is not None and height is not None:
        return int(width), int(height)
    size = body.get("size")
    if _area(size) is not None:
        w, _, h = size.lower().partition("x")
        return int(w), int(h)
    return DEFAULT_SIZE


def feature_exprs(default_steps: int) -> tuple[tuple[str, str], ...]:
    """Declarative form of :func:`diffusion_features`."""
    pixels = DEFAULT_SIZE[0] * DEFAULT_SIZE[1]
    return (
        ("steps", f"coalesce(num_inference_steps, {default_steps})"),
        (
            "megapixels",
            "product(coalesce(product(width, height), area(size), "
            f"{pixels}), 0.000001)",
        ),
        ("n_images", "coalesce(n, 1)"),
    )


def diffusion_features(
    body: Mapping[str, Any], default_steps: int = DEFAULT_STEPS
) -> dict[str, float]:
    """Return steps, megapixels and image count for an image request.

    A hand-written copy of :func:`feature_exprs`, kept for speed; the
    contract suite checks that the two agree.

    Args:
        body: Image request body.
        default_steps: Steps the server uses when the request sets none.

    Returns:
        Feature dict with every feature present.
    """
    steps = _number(body.get("num_inference_steps"))
    n = _number(body.get("n"))
    width, height = _number(body.get("width")), _number(body.get("height"))
    if width is not None and height is not None:
        pixels: float = width * height
    else:
        area = _area(body.get("size"))
        pixels = (
            float(area)
            if area is not None
            else float(DEFAULT_SIZE[0] * DEFAULT_SIZE[1])
        )
    return {
        "steps": float(default_steps) if steps is None else steps,
        "megapixels": pixels * 0.000001,
        "n_images": 1.0 if n is None else n,
    }


class DiffusersAdapter(EngineAdapter):
    """Text-to-image models served by the bundled diffusers server."""

    name: ClassVar[str] = "diffusers"
    workload: ClassVar[WorkloadSpec] = DIFFUSION_WORKLOAD
    supported_versions: ClassVar[tuple[str, ...]] = (">=0.30",)
    default_steps: ClassVar[int] = DEFAULT_STEPS

    def launch_spec(
        self,
        model_id: str,
        *,
        gpus: int = 1,
        extra_args: tuple[str, ...] = (),
    ) -> LaunchSpec:
        """Return ``python -m swarmpilot.engines.servers.diffusers_server``."""
        argv = (
            "python",
            "-m",
            "swarmpilot.engines.servers.diffusers_server",
            "--model",
            model_id,
            "--host",
            "0.0.0.0",
            "--port",
            PORT,
            "--default-steps",
            str(self.default_steps),
        )
        return LaunchSpec(
            argv=(*argv, *extra_args), gpus=gpus, ready_timeout_s=1200.0
        )

    def routes(self) -> tuple[RouteSpec, ...]:
        """Expose image generation."""
        return (
            RouteSpec(
                path="v1/images/generations",
                request_model=ImageGenerationRequest,
                rerun_safe=True,
            ),
        )

    def feature_extractors(self) -> tuple[tuple[str, str], ...]:
        """Steps, megapixels and image count as extractor expressions."""
        return feature_exprs(self.default_steps)

    def features(self, path: str, body: Mapping[str, Any]) -> dict[str, float]:
        """Return steps, megapixels and image count."""
        return diffusion_features(body, self.default_steps)

    def capacity(self) -> Capacity:
        """A diffusion pipeline runs one request at a time."""
        return Capacity(max_in_flight=1, batching="none")

    def example_requests(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Return one small image request."""
        return (
            (
                "v1/images/generations",
                {"prompt": "a red fox", "size": "512x512", "n": 1},
            ),
        )
