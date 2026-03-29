#!/usr/bin/env python3
"""Benchmark script: download SWE-bench Pro, run inference on both models,
collect timing data, train predictor, and report statistics.

Uses the Scheduler's transparent proxy (synchronous API) to send requests.
Each request goes through: Scheduler → scheduling algorithm → instance →
response returned synchronously. The Scheduler auto-collects runtime
samples for predictor training.

Usage:
    python3 scripts_deploy/benchmark.py [--count 100] [--max-tokens 512]

Requires: datasets, huggingface-hub, pyyaml, httpx
    uv pip install datasets huggingface-hub pyyaml httpx
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Error: httpx not installed. Run: uv pip install httpx")

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML not installed. Run: uv pip install pyyaml")

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "cluster.yaml"
CACHE_PATH = Path("/tmp/swebench_pro_benchmark.json")

# ── Config ────────────────────────────────────────────────────


def load_cluster_config() -> dict:
    """Load cluster.yaml and return the cluster dict."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["cluster"]


# ── Dataset ───────────────────────────────────────────────────


def download_dataset(output: Path) -> list[dict]:
    """Download SWE-bench Pro from HuggingFace, cache locally."""
    if output.exists():
        print(f"  Using cached dataset: {output}")
        with open(output) as f:
            return json.load(f)

    print("  Downloading SWE-bench Pro from HuggingFace...")
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        sys.exit(
            "Error: 'datasets' package required. "
            "Run: uv pip install datasets huggingface-hub"
        )

    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    records = []
    for row in ds:
        records.append(
            {
                "case_id": row.get("instance_id", ""),
                "repo": row.get("repo", ""),
                "problem_statement": row.get("problem_statement", ""),
                "requirements": row.get("requirements") or "",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Saved {len(records)} cases to {output}")
    return records


def build_prompt(case: dict) -> str:
    """Build a prompt string from a benchmark case."""
    repo = case.get("repo", "").strip()
    problem = case.get("problem_statement", "").strip()
    reqs = case.get("requirements", "").strip()
    return (
        f"## Repository\n{repo}\n\n"
        f"## Problem Statement\n{problem}\n\n"
        f"## Requirements\n{reqs}"
    )


# ── Synchronous proxy inference ────────────────────────────────


def run_inference(
    client: httpx.Client,
    scheduler_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Send a single synchronous inference request via the Scheduler proxy.

    The Scheduler's catch-all transparent proxy forwards the request to
    an instance selected by the scheduling algorithm, waits for the
    response, and returns it synchronously. Runtime data is auto-collected
    by the Scheduler for predictor training.

    Args:
        client: httpx Client.
        scheduler_url: Base URL of the Scheduler.
        model_id: Model identifier.
        prompt: User prompt text.
        max_tokens: Maximum tokens to generate.

    Returns:
        Dict with 'status', 'latency_ms', and optionally 'error'.
    """
    t0 = time.time()
    try:
        resp = client.post(
            f"{scheduler_url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
        )
        latency_ms = (time.time() - t0) * 1000.0

        if resp.status_code >= 500:
            return {
                "status": "FAILED",
                "latency_ms": latency_ms,
                "error": resp.text[:200],
            }

        resp.raise_for_status()
        return {
            "status": "COMPLETED",
            "latency_ms": latency_ms,
            "response": resp.json(),
        }
    except httpx.TimeoutException:
        return {
            "status": "TIMEOUT",
            "latency_ms": (time.time() - t0) * 1000.0,
            "error": "Request timed out",
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "latency_ms": (time.time() - t0) * 1000.0,
            "error": str(e),
        }


# ── Predictor training ────────────────────────────────────────


def train_predictor(
    client: httpx.Client,
    scheduler_url: str,
    model_id: str,
) -> dict:
    """Trigger predictor training on a scheduler."""
    resp = client.post(
        f"{scheduler_url}/v1/predictor/train",
        json={"model_id": model_id},
    )
    if resp.status_code == 503:
        return {"success": False, "message": "Training not enabled"}
    resp.raise_for_status()
    return resp.json()


# ── Statistics ────────────────────────────────────────────────


def compute_stats(times: list[float]) -> dict:
    """Compute P50/P90/P99/mean/min/max from a list of times in ms."""
    if not times:
        return {}
    s = sorted(times)
    n = len(s)
    return {
        "count": n,
        "mean_ms": round(statistics.mean(s), 1),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
        "p50_ms": round(s[int(n * 0.50)], 1),
        "p90_ms": round(s[int(n * 0.90)], 1),
        "p99_ms": round(s[min(int(n * 0.99), n - 1)], 1),
    }


def print_table(rows: list[list[str]], header: list[str]) -> None:
    """Print a formatted table."""
    all_rows = [header] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    """Run the benchmark pipeline."""
    parser = argparse.ArgumentParser(
        description="Benchmark models with SWE-bench Pro prompts"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of prompts to use (default: 100)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per completion (default: 512)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-task timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────
    cfg = load_cluster_config()
    head = cfg["head_node"]
    models = cfg["models"]

    print("=" * 60)
    print("  SwarmPilot Benchmark")
    print("=" * 60)
    print(f"  Prompts:    {args.count}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Models:     {len(models)}")
    for m in models:
        print(f"    - {m['model_id']} (scheduler :{m['scheduler_port']})")
    print()

    # ── Download dataset ──────────────────────────────────────
    print("[1/4] Downloading dataset...")
    cases = download_dataset(CACHE_PATH)
    cases = cases[: args.count]
    print(f"  Using {len(cases)} prompts")
    print()

    # ── Build prompts ─────────────────────────────────────────
    prompts = [build_prompt(c) for c in cases]

    # ── Run inference serially per model ──────────────────────
    print("[2/4] Running inference (serial, via Scheduler proxy)...")
    print()

    # results[model_id] = list of result dicts with status/latency_ms
    results: dict[str, list[dict]] = {}

    with httpx.Client(timeout=args.timeout) as client:
        for model_cfg in models:
            model_id = model_cfg["model_id"]
            sched_port = model_cfg["scheduler_port"]
            sched_url = f"http://{head}:{sched_port}"

            print(f"  Model: {model_id}")
            print(f"  Scheduler: {sched_url}")

            # Check scheduler health
            try:
                health = client.get(f"{sched_url}/v1/health")
                health.raise_for_status()
            except Exception as e:
                print(f"  ERROR: Scheduler not reachable: {e}")
                results[model_id] = []
                print()
                continue

            model_results = []
            completed = 0
            failed = 0

            for idx, prompt in enumerate(prompts):
                progress = f"[{idx + 1}/{len(prompts)}]"

                result = run_inference(
                    client,
                    sched_url,
                    model_id,
                    prompt,
                    args.max_tokens,
                )
                model_results.append(result)

                st = result["status"]
                t = result["latency_ms"]
                if st == "COMPLETED":
                    completed += 1
                    if (idx + 1) % 10 == 0 or idx == 0:
                        print(f"  {progress} {t:.0f}ms")
                else:
                    failed += 1
                    err = result.get("error", st)
                    print(f"  {progress} {st}: {err}")

            results[model_id] = model_results
            print(
                f"  Done: {completed} completed, {failed} failed"
            )
            print()

        # ── Train predictor ───────────────────────────────────
        print("[3/4] Training predictor...")
        for model_cfg in models:
            model_id = model_cfg["model_id"]
            sched_port = model_cfg["scheduler_port"]
            sched_url = f"http://{head}:{sched_port}"

            train_resp = train_predictor(client, sched_url, model_id)
            success = train_resp.get("success", False)
            samples = train_resp.get("samples_trained", 0)
            msg = train_resp.get("message", "")
            strategy = train_resp.get("strategy", "")

            if success:
                print(
                    f"  {model_id}: trained ({samples} samples)"
                )
                if strategy:
                    print(f"    Strategy auto-switched to: {strategy}")
            else:
                print(f"  {model_id}: {msg}")
        print()

    # ── Compute and display statistics ────────────────────────
    print("[4/4] Results")
    print("=" * 60)
    print()

    output_data = {}

    for model_cfg in models:
        model_id = model_cfg["model_id"]
        model_tasks = results.get(model_id, [])

        completed_tasks = [
            t
            for t in model_tasks
            if t.get("status") == "COMPLETED"
        ]
        failed_tasks = [
            t
            for t in model_tasks
            if t.get("status") != "COMPLETED"
        ]

        times = [t["latency_ms"] for t in completed_tasks]
        stats = compute_stats(times)

        output_data[model_id] = {
            "completed": len(completed_tasks),
            "failed": len(failed_tasks),
            "stats": stats,
        }

        print(f"  {model_id}")
        print(
            f"    Completed: {len(completed_tasks)}/{len(model_tasks)}"
            f"  Failed: {len(failed_tasks)}"
        )
        if stats:
            print(
                f"    Mean:  {stats['mean_ms']:.0f}ms"
                f"    Min: {stats['min_ms']:.0f}ms"
                f"    Max: {stats['max_ms']:.0f}ms"
            )
            print(
                f"    P50:   {stats['p50_ms']:.0f}ms"
                f"    P90: {stats['p90_ms']:.0f}ms"
                f"    P99: {stats['p99_ms']:.0f}ms"
            )
        else:
            print("    No completed tasks — no timing data.")
        print()

    # ── Summary table ─────────────────────────────────────────
    print("Summary")
    print("-" * 60)
    header = ["Model", "OK", "Fail", "Mean", "P50", "P90", "P99"]
    rows = []
    for model_cfg in models:
        mid = model_cfg["model_id"]
        d = output_data.get(mid, {})
        s = d.get("stats", {})
        rows.append(
            [
                mid.split("/")[-1],
                str(d.get("completed", 0)),
                str(d.get("failed", 0)),
                f"{s['mean_ms']:.0f}ms" if s else "-",
                f"{s['p50_ms']:.0f}ms" if s else "-",
                f"{s['p90_ms']:.0f}ms" if s else "-",
                f"{s['p99_ms']:.0f}ms" if s else "-",
            ]
        )
    print_table(rows, header)
    print()

    # ── Save raw results ──────────────────────────────────────
    output_file = SCRIPT_DIR / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Raw results saved to: {output_file}")


if __name__ == "__main__":
    main()
