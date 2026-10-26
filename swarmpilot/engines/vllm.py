"""vLLM engine adapter."""

from __future__ import annotations

from typing import ClassVar

from swarmpilot.engines.base import PORT, LaunchSpec
from swarmpilot.engines.openai_compat import OpenAICompatibleAdapter


class VLLMAdapter(OpenAICompatibleAdapter):
    """vLLM's OpenAI-compatible server (``vllm serve``)."""

    name: ClassVar[str] = "vllm"
    supported_versions: ClassVar[tuple[str, ...]] = (">=0.11",)
    metric_names: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("vllm:num_requests_running",),
        ("vllm:num_requests_waiting",),
        ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    )

    def launch_spec(
        self,
        model_id: str,
        *,
        gpus: int = 1,
        extra_args: tuple[str, ...] = (),
    ) -> LaunchSpec:
        """Return ``vllm serve <model> --host 0.0.0.0 --port <port>``."""
        argv = ["vllm", "serve", model_id, "--host", "0.0.0.0", "--port", PORT]
        if gpus > 1:
            argv += ["--tensor-parallel-size", str(gpus)]
        return LaunchSpec(argv=(*argv, *extra_args), gpus=gpus)
