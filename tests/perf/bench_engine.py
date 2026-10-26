"""Engine-agnostic benchmark: direct engine calls vs. through SwarmPilot.

Plan item 0.2. Generalizes scripts_deploy/benchmark.py so that the same
numbers can be taken for any engine and workload:

- ``llm-chat``: OpenAI chat completions. With streaming it records TTFT,
  TPOT and end-to-end latency; without it only end-to-end latency.
- ``diffusion-image``: ``/v1/images/generations`` end-to-end latency.

The engine and the Scheduler are started by you (see tests/perf/README.md),
and this script drives them with the same requests at each concurrency
level, first directly and then through the Scheduler proxy.

Usage:
    uv run python tests/perf/bench_engine.py \
        --workload llm-chat --model Qwen/Qwen3-8B \
        --direct http://gpu-node:8100 --proxy http://scheduler:8000 \
        --engine vllm --gpu H100 --concurrency 1 16 64 \
        --requests 200 --out tests/perf/baselines/engine-vllm-H100.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import aiohttp
import numpy as np
from _harness import git_commit, write_json

WORDS = [
    "the",
    "scheduler",
    "routes",
    "every",
    "request",
    "to",
    "the",
    "replica",
    "with",
    "the",
    "lowest",
    "predicted",
    "tail",
    "latency",
    "while",
    "the",
    "planner",
    "scales",
    "replicas",
    "to",
    "demand",
]


def synthetic_prompts(n: int, seed: int, max_tokens: int) -> list[dict]:
    """Return ``n`` chat prompts with lognormal lengths (about 50-2000 words).

    Args:
        n: Number of prompts.
        seed: RNG seed so every run sends the same prompts.
        max_tokens: Output length requested for every prompt.

    Returns:
        List of dicts with ``prompt`` and ``max_tokens``.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        words = int(min(2000, max(50, rng.lognormvariate(5.5, 0.8))))
        out.append(
            {
                "prompt": " ".join(rng.choice(WORDS) for _ in range(words)),
                "max_tokens": max_tokens,
            }
        )
    return out


def load_dataset(path: Path, max_tokens: int) -> list[dict]:
    """Load prompts from a JSONL file with ``prompt`` and optional ``max_tokens``."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row.setdefault("max_tokens", max_tokens)
                rows.append(row)
    return rows


def detect_gpu() -> str:
    """Return the first GPU name from nvidia-smi, or ``"none"``."""
    if not shutil.which("nvidia-smi"):
        return "none"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


async def detect_engine_version(
    session: aiohttp.ClientSession, base: str
) -> str:
    """Ask the engine for its version (vLLM ``/version``, SGLang server info)."""
    for path, key in (("/version", "version"), ("/get_server_info", "version")):
        try:
            async with session.get(f"{base}{path}") as r:
                if r.status == 200:
                    data = await r.json(content_type=None)
                    if isinstance(data, dict) and data.get(key):
                        return str(data[key])
        except (aiohttp.ClientError, ValueError):
            continue
    return "unknown"


class Workload:
    """Builds requests for one workload type and parses its responses."""

    def __init__(self, args: argparse.Namespace):
        self.kind = args.workload
        self.model = args.model
        self.stream = not args.no_stream and self.kind == "llm-chat"
        self.extra = json.loads(args.extra_body) if args.extra_body else {}
        if self.kind == "llm-chat":
            self.items = (
                load_dataset(args.dataset, args.max_tokens)
                if args.dataset
                else synthetic_prompts(
                    args.requests, args.seed, args.max_tokens
                )
            )
        else:
            rng = random.Random(args.seed)
            self.items = [
                {
                    "prompt": "a lighthouse on a cliff at dusk",
                    "num_inference_steps": rng.choice(args.steps),
                    "size": rng.choice(args.sizes),
                }
                for _ in range(args.requests)
            ]

    def request(self, i: int) -> tuple[str, dict]:
        """Return (path, JSON body) for item ``i``."""
        item = self.items[i % len(self.items)]
        if self.kind == "llm-chat":
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": item["prompt"]}],
                "max_tokens": item["max_tokens"],
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": self.stream,
            }
            if self.stream:
                body["stream_options"] = {"include_usage": True}
            return "/v1/chat/completions", {**body, **self.extra}
        body = {"model": self.model, "n": 1, **item}
        return "/v1/images/generations", {**body, **self.extra}


async def one_request(
    session: aiohttp.ClientSession, base: str, wl: Workload, i: int
) -> dict:
    """Send one request and return its timings."""
    path, body = wl.request(i)
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage_tokens = None
    async with session.post(f"{base}{path}", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(
                f"HTTP {resp.status}: {(await resp.text())[:120]}"
            )
        if wl.stream:
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                data = json.loads(line[5:])
                if data.get("usage"):
                    usage_tokens = data["usage"].get("completion_tokens")
                choices = data.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
        else:
            data = await resp.json(content_type=None)
            usage = data.get("usage") or {}
            usage_tokens = usage.get("completion_tokens")
    e2e = time.perf_counter() - t0
    tokens = usage_tokens or chunks
    out = {"e2e_ms": e2e * 1000, "tokens": tokens}
    if ttft is not None:
        out["ttft_ms"] = ttft * 1000
        if tokens > 1:
            out["tpot_ms"] = (e2e - ttft) * 1000 / (tokens - 1)
    return out


async def run_level(
    session: aiohttp.ClientSession,
    base: str,
    wl: Workload,
    concurrency: int,
    n_requests: int,
) -> dict:
    """Send ``n_requests`` with ``concurrency`` in flight and summarize them."""
    next_i = 0
    samples: list[dict] = []
    errors: list[str] = []

    async def worker() -> None:
        nonlocal next_i
        while next_i < n_requests:
            i = next_i
            next_i += 1
            try:
                samples.append(await one_request(session, base, wl, i))
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}"[:200])

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - t0

    out: dict = {
        "concurrency": concurrency,
        "requests": len(samples),
        "errors": len(errors),
        "wall_s": round(wall, 3),
        "req_per_s": round(len(samples) / wall, 3) if wall else 0.0,
    }
    tokens = sum(s["tokens"] for s in samples)
    if wl.kind == "llm-chat":
        out["output_tok_per_s"] = round(tokens / wall, 1) if wall else 0.0
    for key in ("e2e_ms", "ttft_ms", "tpot_ms"):
        xs = [s[key] for s in samples if key in s]
        if xs:
            name = key.removesuffix("_ms")
            out[f"{name}_p50_ms"] = round(float(np.percentile(xs, 50)), 3)
            out[f"{name}_p99_ms"] = round(float(np.percentile(xs, 99)), 3)
    if errors:
        out["error_samples"] = errors[:5]
    return out


async def main_async(args: argparse.Namespace) -> dict:
    """Run every concurrency level against every target."""
    wl = Workload(args)
    targets = [("direct", args.direct)]
    if args.proxy:
        targets.append(("proxy", args.proxy))
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        version = args.engine_version or await detect_engine_version(
            session, args.direct
        )
        results = []
        for c in args.concurrency:
            row: dict = {"concurrency": c}
            for name, base in targets:
                # Warm up so first-request compilation is not measured.
                await run_level(session, base, wl, min(c, 4), min(c, 4))
                row[name] = await run_level(session, base, wl, c, args.requests)
            if "proxy" in row and row["direct"]["req_per_s"]:
                row["throughput_ratio"] = round(
                    row["proxy"]["req_per_s"] / row["direct"]["req_per_s"], 4
                )
            print(json.dumps(row), flush=True)
            results.append(row)
    return {
        "benchmark": "engine",
        "commit": git_commit(),
        "date": time.strftime("%Y-%m-%d"),
        "engine": args.engine,
        "engine_version": version,
        "gpu": args.gpu or detect_gpu(),
        "workload": args.workload,
        "model": args.model,
        "stream": wl.stream,
        "host": {"python": platform.python_version(), "cpus": os.cpu_count()},
        "results": results,
    }


def main() -> None:
    """Parse arguments, run the benchmark and write the JSON report."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--workload",
        choices=["llm-chat", "diffusion-image"],
        default="llm-chat",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--direct", required=True, help="Engine base URL")
    p.add_argument("--proxy", help="Scheduler base URL (omit for direct only)")
    p.add_argument("--engine", default="unknown", help="vllm, sglang, ...")
    p.add_argument("--engine-version", default="")
    p.add_argument("--gpu", default="", help="GPU type; detected if empty")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--dataset", type=Path, help="JSONL with prompt[, max_tokens]"
    )
    p.add_argument("--steps", type=int, nargs="+", default=[20, 30, 50])
    p.add_argument("--sizes", nargs="+", default=["512x512", "1024x1024"])
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--extra-body", default="", help="JSON merged into bodies")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    write_json(args.out, asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
