#!/usr/bin/env python3
"""Generate workload for Multi-Scheduler example (sleep models).

This script generates tasks with equal QPS distribution across 3 sleep models,
queries the planner for the scheduler map, and submits tasks directly to the correct
per-model scheduler.

Usage:
    python examples/multi_scheduler/generate_workload.py [options]

Options:
    --total-tasks       Total number of tasks to submit (default: 120)
    --target-qps        Target tasks per second (default: 6)
    --duration          Duration in seconds (default: 20)
    --sleep-time-min    Minimum sleep time in seconds (default: 0.1)
    --sleep-time-max    Maximum sleep time in seconds (default: 1.0)
    --planner-url       Planner URL for scheduler discovery (default: http://localhost:8003)

Example:
    python examples/multi_scheduler/generate_workload.py --total-tasks 300 --target-qps 10
"""

import argparse
import asyncio
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


class SchedulerRouter:
    """Routes tasks to the correct per-model scheduler.

    Queries the planner's /v1/scheduler/list endpoint to build a mapping
    from model_id to scheduler_url.

    Attributes:
        planner_url: Base URL of the planner service.
        scheduler_map: Mapping from model_id to scheduler URL.
    """

    def __init__(self, planner_url: str) -> None:
        """Initialize the scheduler router.

        Args:
            planner_url: Base URL of the planner service.
        """
        self.planner_url = planner_url.rstrip("/")
        self.scheduler_map: dict[str, str] = {}

    def refresh(self) -> None:
        """Refresh the scheduler map from the planner.

        Raises:
            RuntimeError: If the planner is unreachable or returns no schedulers.
        """
        try:
            response = httpx.get(
                f"{self.planner_url}/v1/scheduler/list", timeout=5.0
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get scheduler list: HTTP {response.status_code}"
                )
            data = response.json()
            self.scheduler_map = {
                s["model_id"]: s["scheduler_url"]
                for s in data.get("schedulers", [])
            }
            if not self.scheduler_map:
                raise RuntimeError("No schedulers registered with planner")
        except httpx.RequestError as e:
            raise RuntimeError(f"Cannot reach planner: {e}") from e

    def get_scheduler_url(self, model_id: str) -> str:
        """Get the scheduler URL for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Scheduler URL.

        Raises:
            KeyError: If no scheduler is registered for the model.
        """
        if model_id not in self.scheduler_map:
            raise KeyError(
                f"No scheduler registered for model: {model_id}. "
                f"Available: {list(self.scheduler_map.keys())}"
            )
        return self.scheduler_map[model_id]


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


import random


async def submit_task(
    client: httpx.AsyncClient,
    scheduler_url: str,
    task_id: str,
    model_id: str,
    sleep_time: float,
) -> bool:
    """Submit a single task to the model's scheduler.

    Args:
        client: HTTP client.
        scheduler_url: Scheduler base URL for this model.
        task_id: Unique task identifier.
        model_id: Model to route the task to.
        sleep_time: Sleep duration in seconds.

    Returns:
        True if task was accepted, False otherwise.
    """
    payload = {
        "task_id": task_id,
        "model_id": model_id,
        "task_input": {"sleep_time": sleep_time},
        "metadata": {
            "sleep_time": sleep_time,
            "exp_runtime": sleep_time * 1000,  # Convert to ms
            "path": "inference",  # Route to sleep model inference endpoint
        },
    }

    try:
        response = await client.post(
            f"{scheduler_url}/v1/task/submit",
            json=payload,
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
    router: SchedulerRouter,
    model_ids: list[str],
    total_tasks: int,
    target_qps: float,
    sleep_time_range: tuple[float, float],
) -> WorkloadStats:
    """Run the workload generator.

    Args:
        router: Scheduler router for per-model task routing.
        model_ids: List of model IDs to distribute tasks across.
        total_tasks: Total number of tasks to generate.
        target_qps: Target tasks per second.
        sleep_time_range: (min, max) sleep time in seconds.

    Returns:
        WorkloadStats with results.
    """
    stats = WorkloadStats()
    stats.start_time = time.time()

    # Generate unique run prefix based on timestamp
    run_prefix = f"run{int(time.time()) % 100000}"

    # Calculate tasks per model (equal distribution)
    tasks_per_model = total_tasks // len(model_ids)
    remainder = total_tasks % len(model_ids)

    # Build task queue with interleaved order
    task_queue: list[tuple[str, str]] = []

    for model_idx, model_id in enumerate(model_ids):
        # Distribute remainder tasks to first N models
        tasks_for_this_model = tasks_per_model + (1 if model_idx < remainder else 0)

        for i in range(tasks_for_this_model):
            task_idx = sum(
                tasks_per_model + (1 if j < remainder else 0)
                for j in range(model_idx)
            ) + i
            task_queue.append((f"{run_prefix}-{task_idx:04d}", model_id))

    console.print(
        f"\n[bold cyan]Workload Configuration:[/bold cyan]"
        f"\n  Total Tasks: {total_tasks}"
        f"\n  Models: {', '.join(model_ids)}"
        f"\n  Distribution: Equal ({tasks_per_model} tasks per model)"
        f"\n  Target QPS: {target_qps}"
        f"\n  Sleep Time: {sleep_time_range[0]:.2f}s - {sleep_time_range[1]:.2f}s"
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
            # Phase 1: Submit tasks to per-model schedulers
            for task_id, model_id in task_queue:
                sleep_time = random.uniform(*sleep_time_range)

                task_stat = TaskStats(
                    task_id=task_id,
                    model_id=model_id,
                    submit_time=time.time(),
                )
                stats.add_task(task_stat)

                scheduler_url = router.get_scheduler_url(model_id)
                success = await submit_task(
                    client, scheduler_url, task_id, model_id, sleep_time
                )
                if not success:
                    stats.mark_complete(task_id, "failed", "Submit failed")
                    progress.update(complete_task_id, advance=1)

                progress.update(submit_task_id, advance=1)
                await asyncio.sleep(interval)

            progress.update(submit_task_id, description="[green]Tasks submitted")

            # Phase 2: Poll for completion from per-model schedulers
            pending_tasks = [
                (t.task_id, t.model_id)
                for t in stats.tasks.values()
                if t.status == "pending"
            ]

            poll_timeout = 300  # 5 minutes max wait
            poll_start = time.time()

            while pending_tasks and (time.time() - poll_start) < poll_timeout:
                still_pending = []

                for task_id, model_id in pending_tasks:
                    scheduler_url = router.get_scheduler_url(model_id)
                    poll_status, error = await poll_task_status(
                        client, scheduler_url, task_id
                    )

                    if poll_status in ("completed", "failed"):
                        stats.mark_complete(task_id, poll_status, error)
                        progress.update(complete_task_id, advance=1)
                    else:
                        still_pending.append((task_id, model_id))

                pending_tasks = still_pending
                if pending_tasks:
                    await asyncio.sleep(0.5)

            # Mark remaining as timeout
            for task_id, _ in pending_tasks:
                stats.mark_complete(task_id, "failed", "Timeout")
                progress.update(complete_task_id, advance=1)

    stats.end_time = time.time()
    return stats


def print_results(stats: WorkloadStats, model_ids: list[str]) -> None:
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

    for model_id in model_ids:
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
        description="Generate workload for Multi-Scheduler (sleep models)"
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=120,
        help="Total number of tasks to submit (default: 120)",
    )
    parser.add_argument(
        "--target-qps",
        type=float,
        default=6.0,
        help="Target tasks per second (default: 6)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Duration in seconds (default: 20)",
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
        "--planner-url",
        type=str,
        default="http://localhost:8003",
        help="Planner URL for scheduler discovery (default: http://localhost:8003)",
    )

    args = parser.parse_args()

    console.print(
        "[bold blue]╔════════════════════════════════════════════════════════╗[/bold blue]"
    )
    console.print(
        "[bold blue]║    Multi-Scheduler - Workload Generator (Sleep)        ║[/bold blue]"
    )
    console.print(
        "[bold blue]╚════════════════════════════════════════════════════════╝[/bold blue]"
    )

    # Discover schedulers from planner
    console.print("\nDiscovering schedulers from planner...")
    router = SchedulerRouter(args.planner_url)
    try:
        router.refresh()
        console.print(
            f"[green]✓ Found {len(router.scheduler_map)} schedulers[/green]"
        )
        model_ids = sorted(router.scheduler_map.keys())
        for model_id in model_ids:
            console.print(f"  {model_id}: {router.scheduler_map[model_id]}")
    except RuntimeError as e:
        console.print(f"[red]Failed to discover schedulers: {e}[/red]")
        sys.exit(1)

    # Health-check each scheduler
    console.print("\nChecking scheduler health...")
    for model_id in model_ids:
        scheduler_url = router.scheduler_map[model_id]
        try:
            response = httpx.get(f"{scheduler_url}/v1/health", timeout=5.0)
            if response.status_code != 200:
                console.print(
                    f"[red]Scheduler for {model_id} not healthy "
                    f"at {scheduler_url}[/red]"
                )
                sys.exit(1)
            console.print(f"[green]✓ {model_id} scheduler is healthy[/green]")
        except Exception as e:
            console.print(
                f"[red]Cannot connect to {model_id} scheduler: {e}[/red]"
            )
            sys.exit(1)

    # Check instances are registered in each scheduler
    console.print("Checking registered instances...")
    total_instances = 0
    for model_id in model_ids:
        scheduler_url = router.scheduler_map[model_id]
        try:
            response = httpx.get(
                f"{scheduler_url}/v1/instance/list", timeout=5.0
            )
            if response.status_code == 200:
                instances = response.json().get("instances", [])
                total_instances += len(instances)
                console.print(
                    f"  {model_id}: {len(instances)} instances"
                )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not check {model_id} instances: "
                f"{e}[/yellow]"
            )

    if total_instances == 0:
        console.print(
            "[red]No instances registered. "
            "Run deploy_model.sh first.[/red]"
        )
        sys.exit(1)

    console.print(
        f"[green]✓ {total_instances} total instances registered[/green]"
    )

    # Run workload
    stats = asyncio.run(
        run_workload(
            router=router,
            model_ids=model_ids,
            total_tasks=args.total_tasks,
            target_qps=args.target_qps,
            sleep_time_range=(args.sleep_time_min, args.sleep_time_max),
        )
    )

    # Print results
    print_results(stats, model_ids)


if __name__ == "__main__":
    main()
