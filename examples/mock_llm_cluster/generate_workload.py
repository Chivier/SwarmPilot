#!/usr/bin/env python3
"""Generate workload for Mock LLM Cluster example (Multi-Scheduler).

This script generates tasks with a 1:5 QPS ratio between llm-7b and llm-32b models,
queries the planner for the scheduler map, and submits tasks directly to the correct
per-model scheduler.

Usage:
    python examples/mock_llm_cluster/generate_workload.py [options]

Options:
    --total-tasks   Total number of tasks to submit (default: 120)
    --target-qps    Target tasks per second (default: 6)
    --planner-url   Planner URL for scheduler discovery (default: http://localhost:8002)

Example:
    python examples/mock_llm_cluster/generate_workload.py --total-tasks 300 --target-qps 10

PYLET-024: Multi-Scheduler Architecture
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

    Queries the planner's /scheduler/list endpoint to build a mapping
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


async def submit_task(
    client: httpx.AsyncClient,
    scheduler_url: str,
    task_id: str,
    model_id: str,
) -> bool:
    """Submit a single task to the model's scheduler.

    Args:
        client: HTTP client.
        scheduler_url: Scheduler base URL for this model.
        task_id: Unique task identifier.
        model_id: Model to route the task to.

    Returns:
        True if task was accepted, False otherwise.
    """
    messages = [{"role": "user", "content": f"Test prompt for task {task_id}"}]

    # task_input is sent as-is to the instance endpoint by the worker.
    # The mock LLM server's /task/submit expects TaskSubmitRequest
    # fields (task_id, model_id, task_input) at the top level.
    task_input = {
        "task_id": task_id,
        "model_id": model_id,
        "task_input": {"messages": messages},
    }

    try:
        response = await client.post(
            f"{scheduler_url}/v1/task/submit",
            json={
                "task_id": task_id,
                "model_id": model_id,
                "task_input": task_input,
                "metadata": {"source": "workload_generator", "path": "task/submit"},
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
    router: SchedulerRouter,
    total_tasks: int,
    target_qps: float,
) -> WorkloadStats:
    """Run the workload generator.

    Args:
        router: Scheduler router for per-model task routing.
        total_tasks: Total number of tasks to generate.
        target_qps: Target tasks per second.

    Returns:
        WorkloadStats with results.
    """
    stats = WorkloadStats()
    stats.start_time = time.time()

    # Generate unique run prefix based on timestamp
    run_prefix = f"run{int(time.time()) % 100000}"

    # Calculate tasks per model based on 1:5 ratio
    # 1 part for 7B, 5 parts for 32B = 6 total parts
    tasks_7b = total_tasks // 6
    tasks_32b = total_tasks - tasks_7b

    # Build task queue with interleaved order
    task_queue: list[tuple[str, str]] = []
    idx_7b = 0
    idx_32b = 0

    for i in range(total_tasks):
        # Every 6th task goes to 7B (1:5 ratio)
        if i % 6 == 0 and idx_7b < tasks_7b:
            task_queue.append((f"{run_prefix}-{i:04d}", "llm-7b"))
            idx_7b += 1
        elif idx_32b < tasks_32b:
            task_queue.append((f"{run_prefix}-{i:04d}", "llm-32b"))
            idx_32b += 1
        elif idx_7b < tasks_7b:
            task_queue.append((f"{run_prefix}-{i:04d}", "llm-7b"))
            idx_7b += 1

    console.print(
        f"\n[bold cyan]Workload Configuration:[/bold cyan]"
        f"\n  Total Tasks: {total_tasks}"
        f"\n  Traffic Ratio: 1:5 (7B:32B)"
        f"\n  Tasks for llm-7b: {tasks_7b}"
        f"\n  Tasks for llm-32b: {tasks_32b}"
        f"\n  Target QPS: {target_qps}"
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
                task_stat = TaskStats(
                    task_id=task_id,
                    model_id=model_id,
                    submit_time=time.time(),
                )
                stats.add_task(task_stat)

                scheduler_url = router.get_scheduler_url(model_id)
                success = await submit_task(
                    client, scheduler_url, task_id, model_id
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

            poll_timeout = 120  # 2 minutes max wait
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

    for model_id in ["llm-7b", "llm-32b"]:
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

    # Traffic analysis
    console.print("\n[bold cyan]Traffic Analysis:[/bold cyan]")
    model_7b = stats.get_model_stats("llm-7b")
    model_32b = stats.get_model_stats("llm-32b")

    if model_7b["total"] > 0 and model_32b["total"] > 0:
        actual_ratio = model_32b["total"] / model_7b["total"]
        console.print("  Target Ratio:  1:5 (32B gets 5x traffic)")
        console.print(f"  Actual Ratio:  1:{actual_ratio:.1f}")
        console.print(
            f"  7B Tasks:      {model_7b['total']} ({100*model_7b['total']/(model_7b['total']+model_32b['total']):.1f}%)"
        )
        console.print(
            f"  32B Tasks:     {model_32b['total']} ({100*model_32b['total']/(model_7b['total']+model_32b['total']):.1f}%)"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate workload for Mock LLM Cluster"
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
        "--planner-url",
        type=str,
        default="http://localhost:8002",
        help="Planner URL for scheduler discovery (default: http://localhost:8002)",
    )

    args = parser.parse_args()

    console.print(
        "[bold blue]╔════════════════════════════════════════════════════════╗[/bold blue]"
    )
    console.print(
        "[bold blue]║     Mock LLM Cluster - Workload Generator (Multi)     ║[/bold blue]"
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
        for model_id, url in router.scheduler_map.items():
            console.print(f"  {model_id}: {url}")
    except RuntimeError as e:
        console.print(f"[red]Failed to discover schedulers: {e}[/red]")
        sys.exit(1)

    # Health-check each scheduler
    console.print("\nChecking scheduler health...")
    for model_id, scheduler_url in router.scheduler_map.items():
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
    for model_id, scheduler_url in router.scheduler_map.items():
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
            "Run deploy_models.sh first.[/red]"
        )
        sys.exit(1)

    console.print(
        f"[green]✓ {total_instances} total instances registered[/green]"
    )

    # Run workload
    stats = asyncio.run(
        run_workload(
            router=router,
            total_tasks=args.total_tasks,
            target_qps=args.target_qps,
        )
    )

    # Print results
    print_results(stats)


if __name__ == "__main__":
    main()
