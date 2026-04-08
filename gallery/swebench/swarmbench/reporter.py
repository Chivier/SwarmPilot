from __future__ import annotations

import json
from pathlib import Path

from swarmbench.models import EvalResult


class Reporter:
    """Generate machine-readable and console evaluation reports."""

    def _build_summary(
        self, results: list[EvalResult]
    ) -> dict[str, dict[str, float | int]]:
        """Aggregate per-dataset evaluation metrics.

        Args:
            results: Evaluation results to aggregate.

        Returns:
            Mapping of dataset names to summary statistics.
        """
        summary: dict[str, dict[str, float | int]] = {}
        for result in results:
            dataset_stats = summary.setdefault(
                result.dataset_name,
                {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "pass_rate": 0.0,
                    "avg_score": 0.0,
                    "scored": 0,
                },
            )
            dataset_stats["total"] += 1
            if result.passed:
                dataset_stats["passed"] += 1
            else:
                dataset_stats["failed"] += 1

            if result.score is not None:
                dataset_stats["scored"] += 1
                dataset_stats["avg_score"] += result.score

        for dataset_stats in summary.values():
            total = int(dataset_stats["total"])
            scored = int(dataset_stats["scored"])
            passed = int(dataset_stats["passed"])
            dataset_stats["pass_rate"] = (passed / total) if total else 0.0
            dataset_stats["avg_score"] = (
                dataset_stats["avg_score"] / scored if scored else 0.0
            )

        return summary

    def to_json(self, results: list[EvalResult], output_path: str) -> None:
        """Write results and summary report as JSON.

        Args:
            results: Evaluation results to serialize.
            output_path: Destination report file path.

        Returns:
            None.
        """
        report_payload = {
            "summary": self._build_summary(results),
            "results": [result.model_dump() for result in results],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, indent=2)

    def to_console(self, results: list[EvalResult]) -> None:
        """Print a formatted report summary to stdout.

        Args:
            results: Evaluation results to summarize.

        Returns:
            None.
        """
        summary = self._build_summary(results)
        print("=== Swarmbench Report ===")
        if not summary:
            print("No results to report.")
            return

        for dataset_name in sorted(summary):
            stats = summary[dataset_name]
            print(
                (
                    f"{dataset_name}: total={int(stats['total'])} "
                    f"passed={int(stats['passed'])} "
                    f"failed={int(stats['failed'])} "
                    f"pass_rate={float(stats['pass_rate']):.2%} "
                    f"avg_score={float(stats['avg_score']):.3f}"
                )
            )

    def report(self, results: list[EvalResult], output_path: str | None = None) -> None:
        """Emit console output and optionally write JSON report.

        Args:
            results: Evaluation results to report.
            output_path: Optional JSON output file path.

        Returns:
            None.
        """
        self.to_console(results)
        if output_path is not None:
            self.to_json(results, output_path)
