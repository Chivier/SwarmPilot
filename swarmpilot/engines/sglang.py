"""SGLang engine adapter."""

from __future__ import annotations

from typing import ClassVar

import httpx

from swarmpilot.engines.base import PORT, LaunchSpec
from swarmpilot.engines.openai_compat import OpenAICompatibleAdapter


class SGLangAdapter(OpenAICompatibleAdapter):
    """SGLang's server (``python -m sglang.launch_server``)."""

    name: ClassVar[str] = "sglang"
    supported_versions: ClassVar[tuple[str, ...]] = (">=0.4",)
    metric_names: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("sglang:num_running_reqs",),
        ("sglang:num_queue_reqs",),
        ("sglang:token_usage",),
    )

    def launch_spec(
        self,
        model_id: str,
        *,
        gpus: int = 1,
        extra_args: tuple[str, ...] = (),
    ) -> LaunchSpec:
        """Return ``python -m sglang.launch_server --model-path <model> ...``."""
        argv = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_id,
            "--host",
            "0.0.0.0",
            "--port",
            PORT,
            "--enable-metrics",
        ]
        if gpus > 1:
            argv += ["--tp-size", str(gpus)]
        return LaunchSpec(argv=(*argv, *extra_args), gpus=gpus)

    async def version(self, client: httpx.AsyncClient, endpoint: str) -> str:
        """Read the version from ``/get_server_info``, then ``/version``."""
        base = endpoint.rstrip("/")
        for path in ("/get_server_info", "/server_info"):
            try:
                r = await client.get(f"{base}{path}", timeout=5)
                if r.is_success and r.json().get("version"):
                    return str(r.json()["version"])
            except (httpx.HTTPError, ValueError, AttributeError):
                continue
        return await super().version(client, endpoint)
