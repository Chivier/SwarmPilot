"""Proxy overhead benchmark (plan item 0.1).

Starts a fake engine that answers instantly and a Scheduler in front of it,
then drives both with a closed loop at several concurrency levels, with and
without streaming. The difference between "direct" and "proxy" is what
SwarmPilot adds per request.

One Scheduler instance is registered per fake engine, because in production
one engine process is one instance.

``--data-plane gateway`` measures ``swarmpilot-gateway`` (phase R3) in
front of the same Scheduler instead of the Scheduler's Python proxy.

Usage:
    uv run python tests/perf/bench_proxy.py
    uv run python tests/perf/bench_proxy.py --data-plane gateway
    uv run python tests/perf/bench_proxy.py --concurrency 1 64 512 \
        --duration 10 --out tests/perf/baselines/proxy-<commit>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import aiohttp
import httpx
from _harness import (
    closed_loop,
    fake_engine_cmd,
    free_port,
    gateway_binary,
    gateway_cmd,
    git_commit,
    scheduler_cmd,
    spawn,
    wait_gateway_instances,
    write_json,
)

BODY = {
    "model": "fake-model",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 16,
}
# The same request in each client API style; the fake engine speaks
# OpenAI Chat, so the other styles measure translation as well.
STYLE_REQUESTS = {
    "openai_chat": ("v1/chat/completions", BODY),
    "anthropic_messages": ("v1/messages", BODY),
    "openai_responses": (
        "v1/responses",
        {"model": "fake-model", "input": "Say hello.", "max_output_tokens": 16},
    ),
}


def make_request(
    session: aiohttp.ClientSession,
    base_url: str,
    stream: bool,
    style: str = "openai_chat",
):
    """Return a coroutine factory that sends one request in ``style``."""
    path, base = STYLE_REQUESTS[style]
    url = f"{base_url}/{path}"
    body = json.dumps({**base, "stream": stream}).encode()
    headers = {"content-type": "application/json"}

    async def one() -> tuple[float, float]:
        t0 = time.perf_counter()
        async with session.post(url, data=body, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text[:120]}")
            ttft = None
            async for _ in resp.content.iter_any():
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
            if stream and ttft is None:
                raise RuntimeError("empty stream")
        lat = (time.perf_counter() - t0) * 1000
        return (ttft if ttft is not None else lat), lat

    return one


async def run_matrix(
    engine_url: str,
    proxy_url: str,
    concurrencies: list[int],
    duration: float,
    warmup: float,
    streams: list[bool],
    style: str = "openai_chat",
) -> list[dict]:
    """Measure direct and proxied calls for every (stream, concurrency)."""
    rows = []
    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector
    ) as session:
        for stream in streams:
            for c in concurrencies:
                row: dict = {"stream": stream, "concurrency": c}
                for target, url, req_style in (
                    ("direct", engine_url, "openai_chat"),
                    ("proxy", proxy_url, style),
                ):
                    res = await closed_loop(
                        make_request(session, url, stream, req_style),
                        c,
                        duration,
                        warmup,
                    )
                    row[target] = res.summary()
                d, p = row["direct"], row["proxy"]
                if p.get("requests") and d.get("requests"):
                    row["added_p50_ms"] = round(
                        p["latency_p50_ms"] - d["latency_p50_ms"], 3
                    )
                    row["added_p99_ms"] = round(
                        p["latency_p99_ms"] - d["latency_p99_ms"], 3
                    )
                    row["throughput_ratio"] = round(p["rps"] / d["rps"], 4)
                print(json.dumps(row), flush=True)
                rows.append(row)
    return rows


def main() -> None:
    """Run the benchmark and print (and optionally save) the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 64, 512]
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument(
        "--strategy",
        default="round_robin",
        help="Scheduler strategy; round_robin isolates proxy cost.",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Skip the streaming runs."
    )
    parser.add_argument(
        "--log-level", default="INFO", help="SCHEDULER_LOGURU_LEVEL"
    )
    parser.add_argument(
        "--engine",
        default="vllm",
        help="Platform software_name the fake engine registers as; picks "
        "the engine adapter (and its in-flight limit) in the Scheduler.",
    )
    parser.add_argument(
        "--data-plane",
        choices=["python", "gateway"],
        default="python",
        help="Serve requests with the Scheduler's proxy or swarmpilot-gateway.",
    )
    parser.add_argument(
        "--client-style",
        choices=sorted(STYLE_REQUESTS),
        default="openai_chat",
        help="API style clients call the proxy in; other than openai_chat "
        "it is translated for the (OpenAI Chat) fake engine.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    binary = None
    if args.data_plane == "gateway":
        binary = gateway_binary()
        if binary is None:
            parser.error("swarmpilot-gateway is not built")

    eport, sport = free_port(), free_port()
    engine_url = f"http://127.0.0.1:{eport}"
    scheduler_url = f"http://127.0.0.1:{sport}"
    tmp = Path(tempfile.mkdtemp(prefix="sp-bench-"))
    control = tmp / "gateway.sock"
    env = {
        "SCHEDULING_STRATEGY": args.strategy,
        "SCHEDULER_LOGURU_LEVEL": args.log_level,
        "SCHEDULER_LOG_DIR": str(tmp / "logs"),
        "PREDICTOR_STORAGE_DIR": str(tmp / "models"),
        "SCHEDULER_MODEL_ID": "fake-model",
        "TRAINING_ENABLE_AUTO": "false",
    }
    if binary is not None:
        env["SCHEDULER_GATEWAY_SOCKET"] = str(control)

    with (
        spawn(
            fake_engine_cmd(eport, latency_ms=0),
            f"{engine_url}/health",
            log_path=tmp / "engine.log",
        ),
        spawn(
            scheduler_cmd(sport),
            f"{scheduler_url}/v1/health",
            env=env,
            log_path=tmp / "scheduler.log",
        ),
        ExitStack() as stack,
    ):
        for i in range(args.instances):
            r = httpx.post(
                f"{scheduler_url}/v1/instance/register",
                json={
                    "instance_id": f"fake-{i}",
                    "model_id": "fake-model",
                    "endpoint": engine_url,
                    "platform_info": {
                        "software_name": args.engine,
                        "software_version": "0.0.0",
                        "hardware_name": "cpu",
                    },
                },
            )
            r.raise_for_status()
        proxy_url = scheduler_url
        if binary is not None:
            gport = free_port()
            proxy_url = f"http://127.0.0.1:{gport}"
            stack.enter_context(
                spawn(
                    gateway_cmd(binary, gport, control),
                    f"{proxy_url}/_gateway/health",
                    env={"GATEWAY_LOG": "warn"},
                    log_path=tmp / "gateway.log",
                )
            )
            wait_gateway_instances(proxy_url, args.instances)
        streams = [False] if args.no_stream else [False, True]
        rows = asyncio.run(
            run_matrix(
                engine_url,
                proxy_url,
                args.concurrency,
                args.duration,
                args.warmup,
                streams,
                args.client_style,
            )
        )

    write_json(
        args.out,
        {
            "benchmark": "proxy_overhead",
            "commit": git_commit(),
            "date": time.strftime("%Y-%m-%d"),
            "host": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "cpus": os.cpu_count(),
            },
            "config": {
                "data_plane": args.data_plane,
                "client_style": args.client_style,
                "instances": args.instances,
                "engine": args.engine,
                "strategy": args.strategy,
                "log_level": args.log_level,
                "duration_s": args.duration,
                "warmup_s": args.warmup,
            },
            "results": rows,
        },
    )


if __name__ == "__main__":
    main()
