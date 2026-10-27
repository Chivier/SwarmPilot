"""Data plane A/B on the 256-instance simulated cluster (phase R3c).

Runs the cluster of ``swarmpilot/sim/specs/default-256.yaml`` (9 hardware
configurations, 6 models, 256 virtual engines from ``swarmpilot-sim
serve``) with one Scheduler per model, and replays the same open-loop
request stream through each data plane:

1. ``pretrain``: Python proxy, round_robin, online training on. Gives
   every platform trained quantile and expect-error models.
2. ``python``: Python proxy, adaptive_bootstrap on the pretrained models,
   training off. A gateway started with ``--shadow`` receives every
   decision (``SCHEDULER_GATEWAY_SHADOW_URL``): the decision mismatch
   rate at cluster scale.
3. ``python2`` (with ``--noise``): the same again, for the run-to-run
   noise floor of the distributions below.
4. ``gateway``: ``swarmpilot-gateway`` serves the same stream, same
   models, same strategy.

Per model it reports the share of requests each instance and each
hardware configuration served (from the engines' completion counters),
end-to-end latency percentiles in simulated time, and the shadow
comparison; then compares ``gateway`` with ``python`` against the bars.

Each model's engines run on their own time scale so that the wall-clock
request rate stays near ``--wall-rps`` while the simulated cluster runs
at ``--util`` of its capacity. Latencies are reported in simulated time,
so data plane overhead is divided by the time scale: this A/B is about
routing at scale; ``bench_proxy.py`` measures overhead.

Usage:
    uv run python tests/perf/ab_cluster.py --out /tmp/ab
    uv run python tests/perf/ab_cluster.py --out /tmp/ab --noise \
        --save tests/perf/baselines/ab-256-r3c.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import aiohttp
import httpx
import numpy as np
import yaml
from _harness import (
    REPO_ROOT,
    gateway_binary,
    git_commit,
    wait_gateway_instances,
    wait_http,
)

SIM = REPO_ROOT / "rust" / "target" / "release" / "swarmpilot-sim"
SPEC = REPO_ROOT / "swarmpilot" / "sim" / "specs" / "default-256.yaml"
TOY = REPO_ROOT / "examples" / "9.engine_plugin" / "toy-ranker-http.yaml"
# Engine ports stay below the Linux ephemeral range (32768).
ENGINE_PORT0 = 20000
SCHED_PORT0 = 19000
GATEWAY_PORT0 = 19100

# Saturated throughput of one engine, simulated req/s, from
# `swarmpilot-sim bench --requests 400` (ShareGPT-like lengths, SDXL at
# 30 steps).
CAPACITY = {
    ("vllm", "llama-3.1-8b", "H100-SXM-80GB"): 31.8,
    ("sglang", "llama-3.1-70b", "H100-SXM-80GB"): 12.6,
    ("vllm", "llama-3.1-70b", "H200-141GB"): 9.2,
    ("vllm", "qwen2.5-7b", "A100-SXM-80GB"): 16.3,
    ("sglang", "qwen2.5-7b", "A100-40GB"): 14.5,
    ("vllm", "qwen2.5-7b", "L40S-48GB"): 10.8,
    ("vllm", "llama-3.2-3b", "A10G-24GB"): 12.5,
    ("diffusers", "sdxl", "L4-24GB"): 0.094,
    ("generic", "generic-small", "RTX4090-24GB"): 15.8,
}
# What the Scheduler registers each engine kind as.
SOFTWARE = {
    "vllm": "vllm",
    "sglang": "sglang",
    "diffusers": "diffusers",
    "generic": "toy-ranker-http",
}


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def instances() -> list[dict]:
    """Every engine instance of the spec, in spec order."""
    spec = yaml.safe_load(SPEC.read_text())
    out = []
    for group in spec["node_groups"]:
        for d in group["deployments"]:
            for i in range(d["instances"]):
                out.append(
                    {
                        "id": f"{group['name']}-{i}",
                        "group": group["name"],
                        "gpu": group["gpu"],
                        "engine": d["engine"],
                        "model": d["model"],
                        "tp": d["tp"],
                    }
                )
    for k, inst in enumerate(out):
        inst["port"] = ENGINE_PORT0 + k
    return out


def plan_models(
    insts: list[dict], util: float, wall_rps: float, max_scale: float
) -> dict[str, dict]:
    """Per model: instances, simulated request rate and time scale."""
    models: dict[str, dict] = {}
    for inst in insts:
        m = models.setdefault(inst["model"], {"instances": [], "capacity": 0.0})
        m["instances"].append(inst)
        m["capacity"] += CAPACITY[(inst["engine"], inst["model"], inst["gpu"])]
    for m in models.values():
        # A request's wall latency is its simulated latency times the time
        # scale, so the scale is capped; the big groups then run below
        # ``util``.
        m["time_scale"] = min(max_scale, util * m["capacity"] / wall_rps)
        m["sim_rps"] = wall_rps * m["time_scale"]
        m["util"] = m["sim_rps"] / m["capacity"]
    return models


def request(model: str, rng: random.Random) -> tuple[str, dict]:
    """One request for ``model``: its path and body."""
    if model == "sdxl":
        return "v1/images/generations", {
            "model": model,
            "prompt": "a lighthouse " * rng.randint(2, 20),
            "num_inference_steps": rng.choice([20, 30, 30, 50]),
            "size": rng.choice(["512x512", "768x768", "1024x1024"]),
        }
    if model == "generic-small":
        n = rng.randint(1, 64)
        return "v1/rank", {
            "query": "q" * rng.randint(4, 200),
            "documents": ["d"] * n,
        }
    prompt = int(min(2000, max(8, rng.lognormvariate(math.log(180), 1.0))))
    # Shortish answers: a request's wall latency is its simulated one
    # times the time scale.
    out = int(min(256, max(4, rng.lognormvariate(math.log(48), 0.8))))
    cap = next(c for c in (64, 128, 256) if c >= out)
    return "v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "word " * prompt}],
        "max_tokens": cap,
        "sim_output_tokens": out,
    }


def start_engines(es: ExitStack, models: dict[str, dict], logdir: Path) -> None:
    """Start every virtual engine (warm) and wait for all of them."""
    procs = []
    for m in models.values():
        for inst in m["instances"]:
            procs.append(
                subprocess.Popen(
                    [
                        str(SIM),
                        "serve",
                        "--engine",
                        inst["engine"],
                        "--model",
                        inst["model"],
                        "--gpu",
                        inst["gpu"],
                        "--tp",
                        str(inst["tp"]),
                        "--port",
                        str(inst["port"]),
                        "--time-scale",
                        f"{m['time_scale']:.6g}",
                        "--warm",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=open(logdir / "engines.log", "ab"),  # noqa: SIM115
                    start_new_session=True,
                )
            )
    es.callback(_stop, procs)
    for m in models.values():
        for inst in m["instances"]:
            wait_http(f"http://127.0.0.1:{inst['port']}/health", 60)


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGTERM)
    for p in procs:
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def _spawn(
    es: ExitStack, argv: list[str], env: dict, log: Path, ready: str
) -> None:
    proc = subprocess.Popen(
        argv,
        env={**os.environ, **env},
        stdout=open(log, "ab"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=REPO_ROOT,
    )
    es.callback(_stop, [proc])
    wait_http(ready, 90, proc=proc)


def start_plane(
    es: ExitStack,
    phase: str,
    models: dict[str, dict],
    storage: Path,
    logdir: Path,
) -> dict[str, str]:
    """Start one Scheduler (and gateway) per model; return client URLs."""
    binary = gateway_binary()
    urls = {}
    for k, (name, m) in enumerate(models.items()):
        sport = SCHED_PORT0 + k
        gport = GATEWAY_PORT0 + k
        sched = f"http://127.0.0.1:{sport}"
        gw = f"http://127.0.0.1:{gport}"
        # Unix socket paths are limited to 108 bytes.
        control = Path(tempfile.gettempdir()) / f"sp-ab-{phase}-{k}.sock"
        control.unlink(missing_ok=True)
        env = {
            "SCHEDULER_MODEL_ID": name,
            "SCHEDULER_LOGURU_LEVEL": "WARNING",
            "SCHEDULER_LOG_DIR": str(logdir / f"{phase}-{name}"),
            "PREDICTOR_STORAGE_DIR": str(storage / name),
            "SWARMPILOT_ENGINE_CONFIGS": str(TOY),
            "SCHEDULING_RANDOM_SEED": "7",
            "PROXY_TIMEOUT": "900",
            "WORKER_HTTP_TIMEOUT": "900",
            "PLANNER_REGISTRATION_URL": "",
            "PLANNER_URL": "",
        }
        if phase == "pretrain":
            env |= {
                "SCHEDULING_STRATEGY": "round_robin",
                "TRAINING_ENABLE_AUTO": "true",
                "TRAINING_PREDICTION_TYPES": "quantile,expect_error",
                "TRAINING_BATCH_SIZE": "200",
            }
        else:
            env |= {
                "SCHEDULING_STRATEGY": "adaptive_bootstrap",
                "TRAINING_ENABLE_AUTO": "false",
                "SCHEDULER_GATEWAY_SOCKET": str(control),
            }
        if phase.startswith("python"):
            env["SCHEDULER_GATEWAY_SHADOW_URL"] = gw
        _spawn(
            es,
            [
                sys.executable,
                "-m",
                "uvicorn",
                "swarmpilot.scheduler.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(sport),
                "--log-level",
                "warning",
                "--no-access-log",
            ],
            env,
            logdir / f"{phase}-{name}-scheduler.log",
            f"{sched}/v1/health",
        )
        for inst in m["instances"]:
            httpx.post(
                f"{sched}/v1/instance/register",
                json={
                    "instance_id": inst["id"],
                    "model_id": name,
                    "endpoint": f"http://127.0.0.1:{inst['port']}",
                    "platform_info": {
                        "software_name": SOFTWARE[inst["engine"]],
                        "software_version": "sim",
                        "hardware_name": inst["gpu"],
                    },
                },
                timeout=10,
            ).raise_for_status()
        urls[name] = sched
        if phase == "pretrain":
            continue
        argv = [
            str(binary),
            "--listen",
            f"127.0.0.1:{gport}",
            "--control",
            str(control),
            "--seed",
            "7",
        ]
        if phase.startswith("python"):
            argv.append("--shadow")
        else:
            urls[name] = gw
        _spawn(
            es,
            argv,
            {"GATEWAY_LOG": "warn"},
            logdir / f"{phase}-{name}-gateway.log",
            f"{gw}/_gateway/health",
        )
        wait_gateway_instances(gw, len(m["instances"]), timeout=60)
    return urls


def completed(models: dict[str, dict]) -> dict[str, float]:
    """Each engine's completion counter."""
    out = {}
    for m in models.values():
        for inst in m["instances"]:
            text = httpx.get(
                f"http://127.0.0.1:{inst['port']}/metrics", timeout=10
            ).text
            for line in text.splitlines():
                if line.startswith("swarmpilot_sim_completed_total"):
                    out[inst["id"]] = float(line.rsplit(" ", 1)[1])
    return out


async def drive(
    models: dict[str, dict],
    urls: dict[str, str],
    duration: float,
    wall_rps: float,
    seed: int,
) -> list[dict]:
    """Replay the seeded open-loop stream; one record per request."""
    plan = []
    for name in models:
        rng = random.Random(f"{seed}-{name}")
        t = 0.0
        while True:
            t += rng.expovariate(wall_rps)
            if t >= duration:
                break
            plan.append((t, name, *request(name, rng)))
    plan.sort(key=lambda x: x[0])
    records: list[dict] = []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:

        async def one(at: float, name: str, path: str, body: dict) -> None:
            await asyncio.sleep(max(0.0, t0 + at - time.monotonic()))
            sent = time.monotonic()
            status, err = 0, None
            try:
                async with s.post(f"{urls[name]}/{path}", json=body) as r:
                    status = r.status
                    await r.read()
            except Exception as e:
                err = repr(e)
            records.append(
                {
                    "model": name,
                    "at": at,
                    "lag_ms": (sent - t0 - at) * 1000,
                    "e2e_sim_ms": (time.monotonic() - sent)
                    * 1000
                    / models[name]["time_scale"],
                    "status": status,
                    "error": err,
                }
            )

        t0 = time.monotonic() + 1.0
        await asyncio.gather(*(one(*p) for p in plan))
    return records


def pct(xs: list[float], q: float) -> float | None:
    """Percentile ``q`` of ``xs``, or None if empty."""
    return round(float(np.percentile(xs, q)), 1) if xs else None


def ks(a: list[float], b: list[float]) -> float | None:
    """Two-sample Kolmogorov-Smirnov statistic."""
    if not a or not b:
        return None
    a_s, b_s = np.sort(a), np.sort(b)
    grid = np.concatenate([a_s, b_s])
    fa = np.searchsorted(a_s, grid, side="right") / len(a_s)
    fb = np.searchsorted(b_s, grid, side="right") / len(b_s)
    return round(float(np.max(np.abs(fa - fb))), 4)


def ks_slower(a: list[float], b: list[float]) -> float | None:
    """How much slower ``b`` is than ``a``: max over x of F_a(x) - F_b(x).

    Zero when ``b`` is nowhere slower; a faster ``b`` does not count.
    """
    if not a or not b:
        return None
    a_s, b_s = np.sort(a), np.sort(b)
    grid = np.concatenate([a_s, b_s])
    fa = np.searchsorted(a_s, grid, side="right") / len(a_s)
    fb = np.searchsorted(b_s, grid, side="right") / len(b_s)
    return round(float(max(0.0, np.max(fa - fb))), 4)


def tvd(a: dict[str, float], b: dict[str, float]) -> float:
    """Total variation distance between two share maps."""
    keys = set(a) | set(b)
    return round(0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys), 4)


def summarize(
    models: dict[str, dict],
    records: list[dict],
    served: dict[str, float],
    shadow: dict[str, dict],
) -> dict:
    """Per model: success, latency and where requests went."""
    out = {}
    for name, m in models.items():
        rs = [r for r in records if r["model"] == name]
        ok = [r["e2e_sim_ms"] for r in rs if r["status"] == 200]
        by_inst = {i["id"]: served.get(i["id"], 0.0) for i in m["instances"]}
        total = sum(by_inst.values()) or 1.0
        by_group: Counter = Counter()
        for inst in m["instances"]:
            by_group[inst["group"]] += by_inst[inst["id"]]
        out[name] = {
            "requests": len(rs),
            "ok": len(ok),
            "ok_rate": round(len(ok) / len(rs), 4) if rs else None,
            "statuses": dict(Counter(str(r["status"]) for r in rs)),
            "e2e_sim_ms": {
                "p50": pct(ok, 50),
                "p90": pct(ok, 90),
                "p99": pct(ok, 99),
            },
            "send_lag_p99_ms": pct([r["lag_ms"] for r in rs], 99),
            "share_by_instance": {
                k: round(v / total, 5) for k, v in by_inst.items()
            },
            "share_by_group": {
                k: round(v / total, 4) for k, v in by_group.items()
            },
            "latencies": ok,
            "shadow": shadow.get(name),
        }
    return out


def run_phase(
    phase: str,
    models: dict[str, dict],
    storage: Path,
    logdir: Path,
    args: argparse.Namespace,
) -> dict:
    """Run one phase on fresh Schedulers; return its summary."""
    print(f"[ab] phase {phase}", flush=True)
    with ExitStack() as es:
        urls = start_plane(es, phase, models, storage, logdir)
        before = completed(models)
        duration = args.pretrain if phase == "pretrain" else args.duration
        records = asyncio.run(
            drive(models, urls, duration, args.wall_rps, args.seed)
        )
        after = completed(models)
        shadow = {}
        if phase.startswith("python"):
            for name, url in urls.items():
                # Let the last releases and decisions reach the gateway.
                time.sleep(0.5)
                s = httpx.get(f"{url}/v1/gateway/shadow", timeout=10).json()
                s.pop("recent_mismatches", None)
                shadow[name] = s
    served = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    return summarize(models, records, served, shadow)


def compare(a: dict, b: dict, noise: dict | None) -> dict:
    """``b`` (gateway) against ``a`` (python), per model, with bars."""
    out = {}
    for name in a:
        pa, pb = a[name], b[name]
        row = {
            "ok_rate": [pa["ok_rate"], pb["ok_rate"]],
            "p50": [pa["e2e_sim_ms"]["p50"], pb["e2e_sim_ms"]["p50"]],
            "p99": [pa["e2e_sim_ms"]["p99"], pb["e2e_sim_ms"]["p99"]],
            "latency_ks": ks(pa["latencies"], pb["latencies"]),
            "latency_ks_slower": ks_slower(pa["latencies"], pb["latencies"]),
            "group_tvd": tvd(pa["share_by_group"], pb["share_by_group"]),
            "instance_tvd": tvd(
                pa["share_by_instance"], pb["share_by_instance"]
            ),
            "shadow_mismatch_rate": (pa["shadow"] or {}).get("mismatch_rate"),
        }
        if noise is not None:
            pn = noise[name]
            row["noise"] = {
                "latency_ks": ks(pa["latencies"], pn["latencies"]),
                "group_tvd": tvd(pa["share_by_group"], pn["share_by_group"]),
                "instance_tvd": tvd(
                    pa["share_by_instance"], pn["share_by_instance"]
                ),
            }
        out[name] = row
    return out


def bars(cmp: dict) -> dict[str, bool]:
    """The R3c bars, per model."""
    out = {}
    for name, row in cmp.items():
        noise = row.get("noise", {})
        ok = (
            (row["shadow_mismatch_rate"] or 0.0) < 0.01
            and row["ok_rate"][1] >= row["ok_rate"][0] - 0.005
            # Latency no slower, and placement no further from Python's,
            # than Python's own run-to-run noise allows (with a floor).
            # Faster does not fail: on time scales below 1 the gateway's
            # lower overhead shows in simulated time.
            and row["latency_ks_slower"]
            <= max(0.05, 1.5 * (noise.get("latency_ks") or 0.0))
            and row["group_tvd"]
            <= max(0.02, 1.5 * (noise.get("group_tvd") or 0.0))
        )
        out[name] = ok
    return out


def main() -> None:
    """Run the phases and print (and optionally save) the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--util", type=float, default=0.3)
    parser.add_argument("--wall-rps", type=float, default=12.0)
    parser.add_argument("--max-time-scale", type=float, default=15.0)
    parser.add_argument("--pretrain", type=float, default=150.0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--noise", action="store_true", help="Also run python twice."
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Recompute the report from the phases saved in --out.",
    )
    parser.add_argument(
        "--models", default="", help="Comma-separated subset of models."
    )
    args = parser.parse_args()
    if gateway_binary() is None or not SIM.exists():
        sys.exit("build swarmpilot-gateway and swarmpilot-sim first")
    args.out.mkdir(parents=True, exist_ok=True)
    logdir = args.out / "logs"
    logdir.mkdir(exist_ok=True)
    models = plan_models(
        instances(), args.util, args.wall_rps, args.max_time_scale
    )
    if args.models:
        keep = args.models.split(",")
        models = {k: v for k, v in models.items() if k in keep}
    for name, m in models.items():
        print(
            f"[ab] {name}: {len(m['instances'])} instances, "
            f"{m['sim_rps']:.1f} sim req/s ({m['util']:.0%} of capacity), "
            f"time scale {m['time_scale']:.3g}",
            flush=True,
        )
    results: dict = {}
    if args.report_only:
        for phase in ("python", "gateway", "python2"):
            f = args.out / f"{phase}.json"
            if f.exists():
                results[phase] = {
                    k: v
                    for k, v in json.loads(f.read_text()).items()
                    if k in models
                }
    if not args.report_only:
        with ExitStack() as es:
            start_engines(es, models, logdir)
            trained = args.out / "models"
            if not trained.exists():
                run_phase("pretrain", models, trained, logdir, args)
            phases = ["python", "gateway"]
            if args.noise:
                phases.append("python2")
            for phase in phases:
                storage = args.out / f"models-{phase}"
                shutil.rmtree(storage, ignore_errors=True)
                shutil.copytree(trained, storage)
                results[phase] = run_phase(phase, models, storage, logdir, args)
                _save(args.out / f"{phase}.json", results[phase])
    cmp = compare(results["python"], results["gateway"], results.get("python2"))
    verdict = bars(cmp)
    report = {
        "benchmark": "ab_cluster",
        "commit": git_commit(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "spec": str(SPEC.relative_to(REPO_ROOT)),
            "util": args.util,
            "wall_rps_per_model": args.wall_rps,
            "duration_s": args.duration,
            "pretrain_s": args.pretrain,
            "seed": args.seed,
            "strategy": "adaptive_bootstrap",
            "time_scales": {
                k: round(m["time_scale"], 4) for k, m in models.items()
            },
            "utilization": {k: round(m["util"], 3) for k, m in models.items()},
        },
        "comparison": cmp,
        "bars_met": verdict,
        "phases": {
            p: {
                name: {k: v for k, v in row.items() if k != "latencies"}
                for name, row in r.items()
            }
            for p, r in results.items()
        },
    }
    print(json.dumps({"comparison": cmp, "bars_met": verdict}, indent=2))
    _save(args.out / "report.json", report)
    if args.save:
        _save(args.save, report)


if __name__ == "__main__":
    main()
