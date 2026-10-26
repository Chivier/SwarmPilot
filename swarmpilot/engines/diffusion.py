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


def parse_size(body: Mapping[str, Any]) -> tuple[int, int]:
    """Return (width, height) from ``size`` or ``width``/``height``.

    Args:
        body: Image request body.

    Returns:
        Width and height in pixels, or the default size when absent or
        malformed.
    """
    width, height = body.get("width"), body.get("height")
    if isinstance(width, int) and isinstance(height, int):
        return width, height
    size = body.get("size")
    if isinstance(size, str):
        w, _, h = size.lower().partition("x")
        if w.isdigit() and h.isdigit():
            return int(w), int(h)
    return DEFAULT_SIZE


def diffusion_features(
    body: Mapping[str, Any], default_steps: int = DEFAULT_STEPS
) -> dict[str, float]:
    """Return steps, megapixels and image count for an image request.

    Args:
        body: Image request body.
        default_steps: Steps the server uses when the request sets none.

    Returns:
        Feature dict with every feature present.
    """
    steps = body.get("num_inference_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        steps = default_steps
    n = body.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        n = 1
    width, height = parse_size(body)
    return {
        "steps": float(steps),
        "megapixels": width * height / 1e6,
        "n_images": float(n),
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
            ),
        )

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
