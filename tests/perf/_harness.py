"""Shared helpers for the perf scripts: processes, ports and load loops."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    """Return a TCP port that is free on localhost right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def git_commit() -> str:
    """Return the short commit hash of the working tree, if any."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "swarmpilot"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def wait_http(
    url: str, timeout: float = 60.0, proc: subprocess.Popen | None = None
) -> None:
    """Poll ``url`` until it answers 2xx or ``timeout`` passes."""
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"process for {url} exited with {proc.returncode}"
            )
        try:
            if httpx.get(url, timeout=1.0).is_success:
                return
        except httpx.HTTPError as e:
            last = e
        time.sleep(0.2)
    raise TimeoutError(f"{url} not ready after {timeout}s: {last}")


@contextlib.contextmanager
def spawn(
    args: list[str],
    ready_url: str,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> Iterator[subprocess.Popen]:
    """Start a process, wait for ``ready_url``, and stop it on exit."""
    full_env = {**os.environ, **(env or {})}
    log = open(log_path, "wb") if log_path else subprocess.DEVNULL  # noqa: SIM115
    proc = subprocess.Popen(
        args, env=full_env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT
    )
    try:
        wait_http(ready_url, proc=proc)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_path:
            log.close()


def fake_engine_cmd(port: int, **opts: float) -> list[str]:
    """Return the argv that starts the fake engine on ``port``."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tests" / "perf" / "fake_engine.py"),
        "--port",
        str(port),
    ]
    for key, value in opts.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    return cmd


def scheduler_cmd(port: int) -> list[str]:
    """Return the argv that starts the Scheduler app on ``port``."""
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "swarmpilot.scheduler.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
        "--no-access-log",
    ]


@dataclass
class LoadResult:
    """Latency samples from one closed-loop load run.

    Attributes:
        latencies_ms: End-to-end latency of each successful request.
        ttft_ms: Time to first body chunk of each successful request.
        errors: Number of failed requests.
        elapsed_s: Measured wall time, excluding warm-up.
        error_samples: Up to five error descriptions.
    """

    latencies_ms: list[float] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    errors: int = 0
    elapsed_s: float = 0.0
    error_samples: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """Return P50/P99 latency, TTFT, throughput and error count."""
        out: dict = {
            "requests": len(self.latencies_ms),
            "errors": self.errors,
            "rps": round(len(self.latencies_ms) / self.elapsed_s, 1)
            if self.elapsed_s
            else 0.0,
        }
        for name, xs in (
            ("latency", self.latencies_ms),
            ("ttft", self.ttft_ms),
        ):
            if xs:
                arr = np.asarray(xs)
                out[f"{name}_p50_ms"] = round(float(np.percentile(arr, 50)), 3)
                out[f"{name}_p99_ms"] = round(float(np.percentile(arr, 99)), 3)
        if self.error_samples:
            out["error_samples"] = self.error_samples
        return out


RequestFn = Callable[[], Awaitable[tuple[float, float]]]


async def closed_loop(
    request: RequestFn,
    concurrency: int,
    duration_s: float,
    warmup_s: float = 1.0,
) -> LoadResult:
    """Run ``concurrency`` workers that call ``request`` back to back.

    Args:
        request: Coroutine factory returning (ttft_ms, latency_ms).
        concurrency: Number of concurrent workers.
        duration_s: Measured duration after warm-up.
        warmup_s: Warm-up time whose samples are dropped.

    Returns:
        LoadResult with the samples recorded after warm-up.
    """
    res = LoadResult()
    start = time.perf_counter()
    measure_from = start + warmup_s
    stop_at = measure_from + duration_s

    async def worker() -> None:
        while True:
            t0 = time.perf_counter()
            if t0 >= stop_at:
                return
            try:
                ttft, lat = await request()
            except Exception as e:
                if t0 >= measure_from:
                    res.errors += 1
                    if len(res.error_samples) < 5:
                        res.error_samples.append(
                            f"{type(e).__name__}: {e}"[:200]
                        )
                continue
            if t0 >= measure_from:
                res.latencies_ms.append(lat)
                res.ttft_ms.append(ttft)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    res.elapsed_s = max(time.perf_counter(), stop_at) - measure_from
    return res


def write_json(path: Path | None, payload: dict) -> None:
    """Print ``payload`` and optionally write it to ``path``."""
    text = json.dumps(payload, indent=2)
    print(text)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
