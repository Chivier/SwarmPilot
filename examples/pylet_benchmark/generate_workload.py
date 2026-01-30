#!/usr/bin/env python3
"""Generate workload for PyLet Benchmark example (Direct Registration).

This script submits sleep model tasks to the scheduler at a specified QPS rate,
selects models using round-robin distribution, and tracks completion statistics.

Usage:
    python examples/pylet_benchmark/generate_workload.py [options]

Options:
    --scheduler-url      Scheduler URL (default: http://localhost:8000)
    --qps                Tasks per second (default: 5)
    --duration           Duration in seconds (default: 60)
    --sleep-time-min     Minimum sleep time in seconds (default: 0.1)
    --sleep-time-max     Maximum sleep time in seconds (default: 1.0)
    --models             Comma-separated model IDs (default: sleep_model_a,sleep_model_b,sleep_model_c)

Example:
    python examples/pylet_benchmark/generate_workload.py --qps 10 --duration 120

PYLET-025: Direct Scheduler Registration Example
"""

import argparse
import asyncio
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


@dataclass
class TaskStats:
    """Statistics for a completed task."""

    task_id: str
    model_id: str
    submit_time: float
    complete_time: float | None = None
    status: str = "pending"
    error: str | None = None

    @property
    def latency_ms(self) -> float | None:
        """Task latency in milliseconds."""
        if self.complete_time is None:
            return None
        return (self.complete_time - self.submit_time) * 1000


@dataclass
class WorkloadStats:
    """Aggregate statistics for workload run."""

    tasks: dict[str, TaskStats] = field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None

    def add_task(self, task: TaskStats) -> None:
        """Add a task to statistics."""
        self.tasks[task.task_id] = task

    def mark_complete(
        self, task_id: str, status: str, error: str | None = None
    ) -> None:
        """Mark a task as complete."""
        if task_id in self.tasks:
            self.tasks[task_id].complete_time = time.time()
            self.tasks[task_id].status = status
            self.tasks[task_id].error = error

    def get_model_stats(self, model_id: str) -> dict[str, Any]:
        """Get statistics for a specific model."""
        model_tasks = [t for t in self.tasks.values() if t.model_id == model_id]
        completed = [t for t in model_tasks if t.status == "completed"]
        latencies = [t.latency_ms for t in completed if t.latency_ms is not None]

        return {
            "total": len(model_tasks),
            "completed": len(completed),
            "failed": len([t for t in model_tasks if t.status == "failed"]),
            "pending": len([t for t in model_tasks if t.status == "pending"]),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
        }

    def summary(self) -> dict[str, Any]:
        """Get overall summary statistics."""
        total = len(self.tasks)
        completed = len([t for t in self.tasks.values() if t.status == "completed"])
        failed = len([t for t in self.tasks.values() if t.status == "failed"])
        pending = len([t for t in self.tasks.values() if t.status == "pending"])

        all_latencies = [
            t.latency_ms
            for t in self.tasks.values()
            if t.status == "completed" and t.latency_ms is not None
        ]

        duration = (self.end_time - self.start_time) if self.end_time else 0
        actual_qps = completed / duration if duration > 0 else 0

        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "duration_s": duration,
            "actual_qps": actual_qps,
            "avg_latency_ms": (
                sum(all_latencies) / len(all_latencies) if all_latencies else 0
            ),
        }


async def submit_task(
    client: httpx.AsyncClient,
    scheduler_url: str,
    task_id: str,
    model_id: str,
    sleep_time: float,
) -> bool:
    """Submit a single task to the scheduler.

    Args:
        client: HTTP client.
        scheduler_url: Scheduler base URL.
        task_id: Unique task identifier.
        model_id: Model to route the task to.
        sleep_time: Sleep duration in seconds.

    Returns:
        True if task was accepted, False otherwise.
    """
    task_input = {"sleep_time": sleep_time}

    try:
        response = await client.post(
            f"{scheduler_url}/v1/task/submit",
            json={
                "task_id": task_id,
                "model_id": model_id,
                "task_input": task_input,
                "metadata": {"sleep_time": sleep_time},
            },
            timeout=10.0,
        )
        return response.status_code == 200

    except Exception as e:
        console.print(f"[red]Failed to submit {task_id}: {e}[/red]")
        return False


async def poll_task_status(
    client: httpx.AsyncClient,
    scheduler_url: str,
    task_id: str,
) -> tuple[str, str | None]:
    """Poll task status.

    Args:
        client: HTTP client.
        scheduler_url: Scheduler base URL.
        task_id: Task identifier.

    Returns:
        Tuple of (status, error_message).
    """
    try:
        response = await client.get(
            f"{scheduler_url}/v1/task/info",
            params={"task_id": task_id},
            timeout=5.0,
        )
        if response.status_code == 200:
            data = response.json()
            task = data.get("task", {})
            return task.get("status", "unknown"), task.get("error")
        return "unknown", None
    except Exception:
        return "unknown", None


async def run_workload(
    scheduler_url: str,
    model_ids: list[str],
    total_tasks: int,
    target_qps: float,
    sleep_time_range: tuple[float, float],
) -> WorkloadStats:
    """Run the workload generator.

    Args:
        scheduler_url: Scheduler URL.
        model_ids: List of model IDs to use (round-robin).
        total_tasks: Total number of tasks to generate.
        target_qps: Target tasks per second.
        sleep_time_range: Tuple of (min, max) sleep time in seconds.

    Returns:
        WorkloadStats with results.
    """
    stats = WorkloadStats()
    stats.start_time = time.time()

    # Generate unique run prefix based on timestamp
    run_prefix = f"run{int(time.time()) % 100000}"

    console.print(
        f"\n[bold cyan]Workload Configuration:[/bold cyan]"
        f"\n  Scheduler URL: {scheduler_url}"
        f"\n  Total Tasks: {total_tasks}"
        f"\n  Models (round-robin): {', '.join(model_ids)}"
        f"\n  Target QPS: {target_qps}"
        f"\n  Sleep Time Range: {sleep_time_range[0]:.2f}s - {sleep_time_range[1]:.2f}s"
        f"\n  Estimated Duration: {total_tasks / target_qps:.1f}s"
        f"\n"
    )

    # Create progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )

    submit_task_id = progress.add_task("Submitting tasks", total=total_tasks)
    complete_task_id = progress.add_task("Completed tasks", total=total_tasks)

    interval = 1.0 / target_qps

    async with httpx.AsyncClient() as client:
        with Live(progress, console=console, refresh_per_second=10):
            # Phase 1: Submit tasks with round-robin model selection
            for i in range(total_tasks):
                model_id = model_ids[i % len(model_ids)]
                sleep_time = random.uniform(*sleep_time_range)
                task_id = f"{run_prefix}-{i:06d}"

                task_stat = TaskStats(
                    task_id=task_id,
                    model_id=model_id,
                    submit_time=time.time(),
                )
                stats.add_task(task_stat)

                success = await submit_task(
                    client, scheduler_url, task_id, model_id, sleep_time
                )
                if not success:
                    stats.mark_complete(task_id, "failed", "Submit failed")
                    progress.update(complete_task_id, advance=1)

                progress.update(submit_task_id, advance=1)
                await asyncio.sleep(interval)

            progress.update(submit_task_id, description="[green]Tasks submitted")

            # Phase 2: Poll for completion
            pending_tasks = [
                t.task_id
                for t in stats.tasks.values()
                if t.status == "pending"
            ]

            poll_timeout = 300  # 5 minutes max wait
            poll_start = time.time()

            while pending_tasks and (time.time() - poll_start) < poll_timeout:
                still_pending = []

                for task_id in pending_tasks:
                    poll_status, error = await poll_task_status(
                        client, scheduler_url, task_id
                    )

                    if poll_status in ("completed", "failed"):
                        stats.mark_complete(task_id, poll_status, error)
                        progress.update(complete_task_id, advance=1)
                    else:
                        still_pending.append(task_id)

                pending_tasks = still_pending
                if pending_tasks:
                    await asyncio.sleep(0.5)

            # Mark remaining as timeout
            for task_id in pending_tasks:
                stats.mark_complete(task_id, "failed", "Timeout")
                progress.update(complete_task_id, advance=1)

    stats.end_time = time.time()
    return stats


def print_results(stats: WorkloadStats) -> None:
    """Print formatted results."""
    summary = stats.summary()

    # Overall summary table
    console.print("\n[bold cyan]Overall Results:[/bold cyan]")
    overall_table = Table(show_header=False, box=None)
    overall_table.add_column("Metric", style="dim")
    overall_table.add_column("Value", style="bold")

    overall_table.add_row("Total Tasks", str(summary["total_tasks"]))
    overall_table.add_row("Completed", f"[green]{summary['completed']}[/green]")
    overall_table.add_row("Failed", f"[red]{summary['failed']}[/red]")
    overall_table.add_row("Pending", f"[yellow]{summary['pending']}[/yellow]")
    overall_table.add_row("Duration", f"{summary['duration_s']:.2f}s")
    overall_table.add_row("Actual QPS", f"{summary['actual_qps']:.2f}")
    overall_table.add_row("Avg Latency", f"{summary['avg_latency_ms']:.2f}ms")

    console.print(overall_table)

    # Per-model statistics
    console.print("\n[bold cyan]Per-Model Statistics:[/bold cyan]")
    model_table = Table()
    model_table.add_column("Model")
    model_table.add_column("Total", justify="right")
    model_table.add_column("Completed", justify="right")
    model_table.add_column("Failed", justify="right")
    model_table.add_column("Avg Latency", justify="right")
    model_table.add_column("Min Latency", justify="right")
    model_table.add_column("Max Latency", justify="right")

    for model_id in sorted(set(t.model_id for t in stats.tasks.values())):
        model_stats = stats.get_model_stats(model_id)
        model_table.add_row(
            model_id,
            str(model_stats["total"]),
            f"[green]{model_stats['completed']}[/green]",
            f"[red]{model_stats['failed']}[/red]" if model_stats["failed"] > 0 else "0",
            f"{model_stats['avg_latency_ms']:.0f}ms",
            f"{model_stats['min_latency_ms']:.0f}ms",
            f"{model_stats['max_latency_ms']:.0f}ms",
        )

    console.print(model_table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate workload for PyLet Benchmark example"
    )
    parser.add_argument(
        "--scheduler-url",
        type=str,
        default="http://localhost:8000",
        help="Scheduler URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=5.0,
        help="Tasks per second (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--sleep-time-min",
        type=float,
        default=0.1,
        help="Minimum sleep time in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--sleep-time-max",
        type=float,
        default=1.0,
        help="Maximum sleep time in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="sleep_model_a,sleep_model_b,sleep_model_c",
        help="Comma-separated model IDs (default: sleep_model_a,sleep_model_b,sleep_model_c)",
    )

    args = parser.parse_args()

    model_ids = [m.strip() for m in args.models.split(",")]
    total_tasks = int(args.qps * args.duration)
    sleep_time_range = (args.sleep_time_min, args.sleep_time_max)

    console.print(
        "[bold blue]╔════════════════════════════════════════════════════════╗[/bold blue]"
    )
    console.print(
        "[bold blue]║     PyLet Benchmark - Workload Generator              ║[/bold blue]"
    )
    console.print(
        "[bold blue]╚════════════════════════════════════════════════════════╝[/bold blue]"
    )

    # Health-check scheduler
    console.print("\nChecking scheduler health...")
    try:
        response = httpx.get(f"{args.scheduler_url}/v1/health", timeout=5.0)
        if response.status_code != 200:
            console.print(f"[red]Scheduler not healthy at {args.scheduler_url}[/red]")
            sys.exit(1)
        console.print(f"[green]✓ Scheduler is healthy[/green]")
    except Exception as e:
        console.print(f"[red]Cannot connect to scheduler: {e}[/red]")
        sys.exit(1)

    # Check instances are registered
    console.print("Checking registered instances...")
    try:
        response = httpx.get(
            f"{args.scheduler_url}/v1/instance/list", timeout=5.0
        )
        if response.status_code == 200:
            instances = response.json().get("instances", [])
            console.print(f"[green]✓ {len(instances)} instances registered[/green]")
            if len(instances) == 0:
                console.print(
                    "[red]No instances registered. "
                    "Run deploy_model.sh first.[/red]"
                )
                sys.exit(1)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not check instances: {e}[/yellow]")

    # Run workload
    stats = asyncio.run(
        run_workload(
            scheduler_url=args.scheduler_url,
            model_ids=model_ids,
            total_tasks=total_tasks,
            target_qps=args.qps,
            sleep_time_range=sleep_time_range,
        )
    )

    # Print results
    print_results(stats)


if __name__ == "__main__":
    main()
