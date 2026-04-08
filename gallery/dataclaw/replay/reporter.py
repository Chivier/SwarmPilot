"""Terminal progress display and JSON report generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from replay.models import GroupMetrics, RequestMetrics


def _percentile_stats(values: list[float]) -> dict[str, float]:
    """Compute p50, p90, p99, mean, min, max for a list of values.

    Args:
        values: Numeric values to summarize.

    Returns:
        Dict with p50, p90, p99, mean, min, max keys.
    """
    if not values:
        return {"p50": 0, "p90": 0, "p99": 0, "mean": 0, "min": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    return {
        "p50": s[int(n * 0.50)],
        "p90": s[min(int(n * 0.90), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
        "mean": round(sum(s) / n, 2),
        "min": s[0],
        "max": s[-1],
    }


class ReplayReporter:
    """Handles terminal progress output and JSON report writing."""

    def print_progress(
        self, completed: int, total: int, metric: RequestMetrics
    ) -> None:
        """Print a real-time progress line to the terminal.

        Args:
            completed: Number of requests completed so far.
            total: Total expected request count.
            metric: The just-completed request metric.
        """
        pct = completed / total * 100 if total else 0
        status_icon = "OK" if metric.status == "success" else "FAIL"
        line = (
            f"\r[{completed}/{total}] ({pct:.1f}%) "
            f"group={metric.group_id[:16]} step={metric.step_index} "
            f"{metric.model_size} {metric.latency_ms:.0f}ms [{status_icon}]"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def print_summary(self, group_metrics: list[GroupMetrics]) -> None:
        """Print final summary statistics to the terminal.

        Args:
            group_metrics: All completed group metrics.
        """
        all_latencies = [
            m.latency_ms
            for gm in group_metrics
            for m in gm.request_metrics
            if m.status == "success"
        ]
        large_latencies = [
            m.latency_ms
            for gm in group_metrics
            for m in gm.request_metrics
            if m.status == "success" and m.model_size == "large"
        ]
        small_latencies = [
            m.latency_ms
            for gm in group_metrics
            for m in gm.request_metrics
            if m.status == "success" and m.model_size == "small"
        ]
        group_latencies = [gm.total_latency_ms for gm in group_metrics]

        total_reqs = sum(gm.total_steps for gm in group_metrics)
        ok_reqs = sum(gm.completed_steps for gm in group_metrics)
        fail_reqs = sum(gm.failed_steps for gm in group_metrics)

        print("\n" + "=" * 60)
        print("REPLAY SUMMARY")
        print("=" * 60)
        print(f"Groups: {len(group_metrics)}")
        print(f"Requests: {total_reqs} total, {ok_reqs} success, {fail_reqs} failed")
        print()
        self._print_latency_table("All requests", all_latencies)
        self._print_latency_table("Large model", large_latencies)
        self._print_latency_table("Small model", small_latencies)
        self._print_latency_table("Group e2e", group_latencies)

    @staticmethod
    def _print_latency_table(label: str, values: list[float]) -> None:
        """Print one latency summary row.

        Args:
            label: Human-readable label for this metric category.
            values: Latency values in milliseconds.
        """
        stats = _percentile_stats(values)
        if not values:
            print(f"  {label}: (no data)")
            return
        print(
            f"  {label}: "
            f"p50={stats['p50']:.0f}ms "
            f"p90={stats['p90']:.0f}ms "
            f"p99={stats['p99']:.0f}ms "
            f"mean={stats['mean']:.0f}ms "
            f"min={stats['min']:.0f}ms "
            f"max={stats['max']:.0f}ms"
        )

    def write_json(
        self, group_metrics: list[GroupMetrics], output_path: str
    ) -> None:
        """Write full results to a JSON file.

        Args:
            group_metrics: All completed group metrics.
            output_path: Path for the output JSON file.
        """
        all_latencies = [
            m.latency_ms
            for gm in group_metrics
            for m in gm.request_metrics
            if m.status == "success"
        ]
        large_latencies = [
            m.latency_ms
            for gm in group_metrics
            for m in gm.request_metrics
            if m.status == "success" and m.model_size == "large"
        ]
        small_latencies = [
            m.latency_ms
            for gm in group_metrics
            for m in gm.request_metrics
            if m.status == "success" and m.model_size == "small"
        ]

        report = {
            "summary": {
                "total_groups": len(group_metrics),
                "total_requests": sum(gm.total_steps for gm in group_metrics),
                "successful_requests": sum(
                    gm.completed_steps for gm in group_metrics
                ),
                "failed_requests": sum(gm.failed_steps for gm in group_metrics),
                "latency_all_ms": _percentile_stats(all_latencies),
                "latency_large_ms": _percentile_stats(large_latencies),
                "latency_small_ms": _percentile_stats(small_latencies),
                "group_latency_ms": _percentile_stats(
                    [gm.total_latency_ms for gm in group_metrics]
                ),
            },
            "groups": [gm.model_dump() for gm in group_metrics],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
