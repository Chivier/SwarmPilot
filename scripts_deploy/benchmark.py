#!/usr/bin/env python3
"""Benchmark script: download SWE-bench Pro, run inference on both models,
collect timing data, train predictor, and report statistics.

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


# ── Task submission ───────────────────────────────────────────


def submit_task(
    client: httpx.Client,
    scheduler_url: str,
    task_id: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Submit a single task and return the response."""
    resp = client.post(
        f"{scheduler_url}/v1/task/submit",
        json={
            "task_id": task_id,
            "model_id": model_id,
            "task_input": {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            "metadata": {
                "path": "v1/chat/completions",
                "method": "POST",
            },
        },
    )
    resp.raise_for_status()
    return resp.json()


def wait_task(
    client: httpx.Client,
    scheduler_url: str,
    task_id: str,
    timeout: float = 600.0,
    poll_interval: float = 2.0,
) -> dict:
    """Poll until task completes or fails. Returns task info dict."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(
            f"{scheduler_url}/v1/task/info",
            params={"task_id": task_id},
        )
        resp.raise_for_status()
        task = resp.json().get("task", {})
        status = task.get("status", "")
        if status in ("COMPLETED", "FAILED"):
            return task
        time.sleep(poll_interval)
    return {"task_id": task_id, "status": "TIMEOUT"}


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
    print("[2/4] Running inference (serial)...")
    print()

    # results[model_id] = list of task info dicts
    results: dict[str, list[dict]] = {}

    with httpx.Client(timeout=max(args.timeout + 30, 60.0)) as client:
        for model_cfg in models:
            model_id = model_cfg["model_id"]
            sched_port = model_cfg["scheduler_port"]
            sched_url = f"http://{head}:{sched_port}"
            short_name = model_id.split("/")[-1]

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
                task_id = f"bench-{short_name}-{idx:04d}"
                progress = f"[{idx + 1}/{len(prompts)}]"

                try:
                    submit_task(
                        client,
                        sched_url,
                        task_id,
                        model_id,
                        prompt,
                        args.max_tokens,
                    )
                except Exception as e:
                    print(f"  {progress} SUBMIT ERROR: {e}")
                    failed += 1
                    model_results.append(
                        {
                            "task_id": task_id,
                            "status": "SUBMIT_FAILED",
                            "error": str(e),
                        }
                    )
                    continue

                # Wait for result
                task_info = wait_task(
                    client,
                    sched_url,
                    task_id,
                    timeout=args.timeout,
                )
                model_results.append(task_info)

                st = task_info.get("status", "?")
                t = task_info.get("execution_time_ms")
                if st == "COMPLETED" and t is not None:
                    completed += 1
                    if (idx + 1) % 10 == 0 or idx == 0:
                        print(
                            f"  {progress} {t:.0f}ms"
                        )
                else:
                    failed += 1
                    err = task_info.get("error", st)
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
            and t.get("execution_time_ms") is not None
        ]
        failed_tasks = [
            t
            for t in model_tasks
            if t.get("status") != "COMPLETED"
        ]

        times = [t["execution_time_ms"] for t in completed_tasks]
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
