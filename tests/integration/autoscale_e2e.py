"""End-to-end check of the Phase 3 loop on a local PyLet cluster.

Starts a Planner in PyLet local mode with the auto-optimizer on, and one
Scheduler for an LLM and one for a diffusion model, each reporting demand
to the Planner. Both models run the fake engine (no GPU): chat completions
take ``LLM_MS``, image generations ``STEPS * MS_PER_STEP``. ``splanner
serve`` deploys two replicas of each through PyLet.

Agent sessions then call the LLM through its Scheduler's proxy and, for
some sessions, the diffusion model with ``X-SwarmPilot-Parent`` set to the
LLM. Phase A is LLM-heavy (one image per ten LLM steps), phase B
image-heavy (two images per LLM step). The check passes when the
auto-optimizer moves machines toward the LLM in phase A and toward the
diffusion model in phase B and keeps it there, through PyLet, while every
request is answered with 200, and the Planner has learned the
LLM-to-diffusion call ratio.

Usage:
    uv run python tests/integration/autoscale_e2e.py
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
PLANNER = 18002
SCHED = {"agent-llm": 18010, "sdxl-diffusion": 18020}
LLM, DIFF = "agent-llm", "sdxl-diffusion"
LLM_MS, STEPS, MS_PER_STEP = 300, 20, 40
INTERVAL = 10.0


def _spawn(cmd: list[str], env: dict, log: Path) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        env={**os.environ, **env},
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_healthy(url: str, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError(url)


def _counts() -> dict[str, int]:
    r = httpx.get(f"http://127.0.0.1:{PLANNER}/v1/instances", timeout=10)
    r.raise_for_status()
    out: dict[str, int] = {}
    for inst in r.json():
        out[inst["model_id"]] = out.get(inst["model_id"], 0) + 1
    return out


async def _session(client: httpx.AsyncClient, images: float, stats: dict) -> None:
    llm = f"http://127.0.0.1:{SCHED[LLM]}/v1/chat/completions"
    diff = f"http://127.0.0.1:{SCHED[DIFF]}/v1/images/generations"
    r = await client.post(
        llm,
        json={
            "model": LLM,
            "messages": [{"role": "user", "content": "plan"}],
            "fake_latency_ms": LLM_MS,
        },
    )
    stats[r.status_code] = stats.get(r.status_code, 0) + 1
    n = int(images) + (1 if (time.monotonic() * 7919) % 1 < images % 1 else 0)
    for _ in range(n):
        r = await client.post(
            diff,
            json={"prompt": "x", "num_inference_steps": STEPS},
            headers={"X-SwarmPilot-Parent": LLM},
        )
        stats[r.status_code] = stats.get(r.status_code, 0) + 1


async def _load(rate: float, images: float, seconds: float) -> dict:
    stats: dict = {}
    tasks = []
    async with httpx.AsyncClient(timeout=120) as client:
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            tasks.append(asyncio.create_task(_session(client, images, stats)))
            await asyncio.sleep(1 / rate)
        await asyncio.gather(*tasks)
    return stats


def _wait_for(pred, timeout: float) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    counts = _counts()
    while time.monotonic() < deadline and not pred(counts):
        time.sleep(2)
        counts = _counts()
    return counts


def main() -> int:
    logs = Path(tempfile.mkdtemp(prefix="autoscale_e2e-"))
    print(f"logs: {logs}")
    engine = (
        f"{PY} {ROOT}/tests/perf/fake_engine.py --host 127.0.0.1 "
        f"--port $PORT --ms-per-step {MS_PER_STEP}"
    )
    procs: list[subprocess.Popen] = []
    try:
        procs.append(
            _spawn(
                [PY, "-m", "swarmpilot.planner.cli", "start", "--port", str(PLANNER)],
                {
                    "PYLET_LOCAL_MODE": "true",
                    "PYLET_LOCAL_PORT": "15100",
                    "PYLET_LOCAL_WORKER_PORT_START": "15300",
                    "PYLET_LOCAL_NUM_WORKERS": "1",
                    "PYLET_LOCAL_GPU_PER_WORKER": "4",
                    "PYLET_CUSTOM_COMMAND": engine,
                    "PYLET_DRAIN_TIMEOUT": "10",
                    "AUTO_OPTIMIZE_ENABLED": "true",
                    "AUTO_OPTIMIZE_INTERVAL": str(INTERVAL),
                    "AUTO_OPTIMIZE_CHANGE_FACTOR": "0.5",
                },
                logs / "planner.log",
            )
        )
        _wait_healthy(f"http://127.0.0.1:{PLANNER}/v1/health", 120)
        for model, port in SCHED.items():
            procs.append(
                _spawn(
                    [
                        PY,
                        "-m",
                        "swarmpilot.scheduler.cli",
                        "start",
                        "--port",
                        str(port),
                    ],
                    {
                        "SCHEDULER_MODEL_ID": model,
                        "PLANNER_URL": f"http://127.0.0.1:{PLANNER}",
                        "SCHEDULER_AUTO_REPORT": "2",
                        "PLANNER_REGISTRATION_URL": (f"http://127.0.0.1:{PLANNER}"),
                        "SCHEDULER_SELF_URL": f"http://127.0.0.1:{port}",
                        "SCHEDULING_STRATEGY": "min_time",
                    },
                    logs / f"scheduler-{port}.log",
                )
            )
            _wait_healthy(f"http://127.0.0.1:{port}/v1/health")
        time.sleep(2)
        for model in SCHED:
            r = httpx.post(
                f"http://127.0.0.1:{PLANNER}/v1/serve",
                json={"model_or_command": model, "gpu_count": 1, "replicas": 2},
                timeout=300,
            )
            print("serve", model, r.status_code, r.text[:200])
            r.raise_for_status()
        start = _wait_for(lambda c: c.get(LLM) == 2 and c.get(DIFF) == 2, 120)
        print("start", start)

        # Phase A: LLM-heavy.
        stats_a = asyncio.run(_load(rate=6.0, images=0.1, seconds=45))
        a = _wait_for(lambda c: c.get(LLM, 0) >= 3, 3 * INTERVAL)
        print("phase A", stats_a, a)
        # Phase B: image-heavy. Sample the deployment while it runs.
        seen: list[dict[str, int]] = []
        stop = threading.Event()

        def sample() -> None:
            while not stop.wait(2):
                with contextlib.suppress(httpx.HTTPError):
                    seen.append(_counts())

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        stats_b = asyncio.run(_load(rate=2.0, images=2.0, seconds=60))
        stop.set()
        sampler.join()
        b = _wait_for(lambda c: c.get(DIFF, 0) >= 3, 3 * INTERVAL)
        # Once the diffusion model has its machines, it keeps them.
        first = next((i for i, c in enumerate(seen) if c.get(DIFF, 0) >= 3), len(seen))
        flips = sum(c.get(DIFF, 0) < 3 for c in seen[first:])
        print(
            "phase B",
            stats_b,
            b,
            f"samples after converging={len(seen) - first} flips={flips}",
        )
        demand = httpx.get(f"http://127.0.0.1:{PLANNER}/v1/demand").json()
        print("demand", demand)

        checks = {
            "phase A moved toward the LLM": a.get(LLM, 0) >= 3,
            "phase B moved toward diffusion": b.get(DIFF, 0) >= 3,
            "phase B converged while loaded": first < len(seen),
            "phase B held its deployment": flips == 0,
            "every request answered 200": set(stats_a) | set(stats_b) == {200},
            "call ratio learned": LLM in demand["call_ratios"],
        }
        for name, passed in checks.items():
            print("ok  " if passed else "FAIL", name)
        ok = all(checks.values())
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(
                f"http://127.0.0.1:{PLANNER}/v1/terminate",
                json={"all": True},
                timeout=60,
            )
        for p in reversed(procs):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())
