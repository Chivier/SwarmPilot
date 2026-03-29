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
import asyncio
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


# ── Async proxy inference with concurrency control ────────────

DEFAULT_CONCURRENCY = 16


async def run_inference(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    scheduler_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    idx: int,
) -> dict:
    """Send a single inference request via the Scheduler proxy.

    Concurrency is bounded by the semaphore so at most N requests
    are in-flight at the same time.

    Args:
        client: httpx AsyncClient.
        semaphore: Concurrency-limiting semaphore.
        scheduler_url: Base URL of the Scheduler.
        model_id: Model identifier.
        prompt: User prompt text.
        max_tokens: Maximum tokens to generate.
        idx: Prompt index (for result ordering).

    Returns:
        Dict with 'idx', 'status', 'latency_ms', and optionally 'error'.
    """
    async with semaphore:
        t0 = time.time()
        try:
            resp = await client.post(
                f"{scheduler_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                },
            )
            latency_ms = (time.time() - t0) * 1000.0

            if resp.status_code >= 500:
                return {
                    "idx": idx,
                    "status": "FAILED",
                    "latency_ms": latency_ms,
                    "error": resp.text[:200],
                }

            resp.raise_for_status()
            return {
                "idx": idx,
                "status": "COMPLETED",
                "latency_ms": latency_ms,
            }
        except httpx.TimeoutException:
            return {
                "idx": idx,
                "status": "TIMEOUT",
                "latency_ms": (time.time() - t0) * 1000.0,
                "error": "Request timed out",
            }
        except Exception as e:
            return {
                "idx": idx,
                "status": "FAILED",
                "latency_ms": (time.time() - t0) * 1000.0,
                "error": str(e),
            }



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


async def main() -> None:
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent requests per model (default: {DEFAULT_CONCURRENCY})",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────
    cfg = load_cluster_config()
    head = cfg["head_node"]
    models = cfg["models"]

    print("=" * 60)
    print("  SwarmPilot Benchmark")
    print("=" * 60)
    print(f"  Prompts:     {args.count}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Models:      {len(models)}")
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

    # ── Run inference on all models in parallel ─────────────
    print(
        f"[2/4] Running inference "
        f"(concurrency={args.concurrency} per model, "
        f"{len(models)} models in parallel)..."
    )
    print()

    # results[model_id] = list of result dicts with status/latency_ms
    results: dict[str, list[dict]] = {}

    async def bench_one_model(
        client: httpx.AsyncClient,
        model_cfg: dict,
    ) -> tuple[str, list[dict], float]:
        """Benchmark a single model. Returns (model_id, results, wall_time)."""
        model_id = model_cfg["model_id"]
        sched_port = model_cfg["scheduler_port"]
        sched_url = f"http://{head}:{sched_port}"
        # Per-model semaphore so each model gets its own concurrency limit
        sem = asyncio.Semaphore(args.concurrency)

        print(f"  Model: {model_id}")
        print(f"  Scheduler: {sched_url}")

        # Check scheduler health
        try:
            health = await client.get(f"{sched_url}/v1/health")
            health.raise_for_status()
        except Exception as e:
            print(f"  ERROR: Scheduler ({model_id}) not reachable: {e}")
            return model_id, [], 0.0

        # Launch all prompts as concurrent tasks
        tasks = [
            run_inference(
                client,
                sem,
                sched_url,
                model_id,
                prompt,
                args.max_tokens,
                idx,
            )
            for idx, prompt in enumerate(prompts)
        ]

        t0 = time.time()
        model_results = list(await asyncio.gather(*tasks))
        wall_time = time.time() - t0

        return model_id, model_results, wall_time

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        # Run all models concurrently
        model_outputs = await asyncio.gather(
            *(bench_one_model(client, m) for m in models)
        )

        for model_id, model_results, wall_time in model_outputs:
            if not model_results:
                results[model_id] = []
                print()
                continue

            # Sort by original index
            model_results = sorted(
                model_results, key=lambda r: r["idx"]
            )

            completed = sum(
                1 for r in model_results if r["status"] == "COMPLETED"
            )
            failed = len(model_results) - completed

            # Print failures
            short = model_id.split("/")[-1]
            for r in model_results:
                if r["status"] != "COMPLETED":
                    print(
                        f"  [{short} #{r['idx'] + 1}] "
                        f"{r['status']}: {r.get('error', '')}"
                    )

            results[model_id] = model_results
            throughput = (
                completed / wall_time if wall_time > 0 else 0
            )
            print(
                f"  {model_id}: {completed} ok, {failed} fail "
                f"in {wall_time:.1f}s ({throughput:.1f} req/s)"
            )

        print()

        # ── Train predictor ───────────────────────────────────
        # Use a sync client for the simple training calls
        print("[3/4] Training predictor...")
        for model_cfg in models:
            model_id = model_cfg["model_id"]
            sched_port = model_cfg["scheduler_port"]
            sched_url = f"http://{head}:{sched_port}"

            resp = await client.post(
                f"{sched_url}/v1/predictor/train",
                json={"model_id": model_id},
            )
            if resp.status_code == 503:
                print(f"  {model_id}: Training not enabled")
                continue
            train_resp = resp.json()
            success = train_resp.get("success", False)
            samples = train_resp.get("samples_trained", 0)
            msg = train_resp.get("message", "")
            strategy = train_resp.get("strategy", "")

            if success:
                print(
                    f"  {model_id}: trained ({samples} samples)"
                )
                if strategy:
                    print(
                        f"    Strategy auto-switched to: {strategy}"
                    )
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
    asyncio.run(main())
