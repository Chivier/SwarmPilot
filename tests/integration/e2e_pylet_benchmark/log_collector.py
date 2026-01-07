"""Log Collector for E2E Testing.

This module provides utilities to collect and organize logs from all
components of the E2E test (scheduler, planner, predictor, PyLet workers).

Features:
- Collect logs from file paths or directories
- Combine multiple log files
- Add timestamps to collected logs
- Generate log summaries

Usage:
    from log_collector import LogCollector

    collector = LogCollector(output_dir=Path("/tmp/test_logs"))
    log_paths = collector.collect_logs({
        "scheduler": "/tmp/scheduler.log",
        "planner": "/tmp/planner.log",
        "predictor": "/tmp/predictor.log",
        "pylet_workers": "/tmp/workers/",
    })
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class LogCollector:
    """Collect and organize logs from test components."""

    def __init__(self, output_dir: Path | str):
        """Initialize log collector.

        Args:
            output_dir: Directory to store collected logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def collect_logs(
        self, sources: dict[str, str | Path]
    ) -> dict[str, Path]:
        """Collect logs from multiple sources.

        Args:
            sources: Dict of component_name -> log_source
                     log_source can be a file path or directory

        Returns:
            Dict of component_name -> collected log path
        """
        collected = {}

        for component, source in sources.items():
            source_path = Path(source)

            if not source_path.exists():
                logger.warning(f"Log source not found: {source}")
                continue

            try:
                if source_path.is_file():
                    collected[component] = self._collect_file(component, source_path)
                elif source_path.is_dir():
                    collected[component] = self._collect_directory(component, source_path)
            except Exception as e:
                logger.error(f"Failed to collect logs for {component}: {e}")

        return collected

    def _collect_file(self, component: str, source: Path) -> Path:
        """Collect a single log file.

        Args:
            component: Component name
            source: Source file path

        Returns:
            Path to collected log file
        """
        dest = self.output_dir / f"{component}_{self._timestamp}.log"
        shutil.copy(source, dest)
        logger.debug(f"Collected log: {source} -> {dest}")
        return dest

    def _collect_directory(self, component: str, source: Path) -> Path:
        """Collect logs from a directory (combines all .log files).

        Args:
            component: Component name
            source: Source directory path

        Returns:
            Path to combined log file
        """
        dest = self.output_dir / f"{component}_{self._timestamp}.log"

        log_files = sorted(source.glob("*.log"))
        if not log_files:
            logger.warning(f"No .log files found in {source}")
            return dest

        with open(dest, "w") as outfile:
            for log_file in log_files:
                outfile.write(f"\n{'=' * 60}\n")
                outfile.write(f"=== {log_file.name} ===\n")
                outfile.write(f"{'=' * 60}\n\n")
                try:
                    outfile.write(log_file.read_text())
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")

        logger.debug(
            f"Collected {len(log_files)} log files from {source} -> {dest}"
        )
        return dest

    def collect_process_output(
        self,
        component: str,
        stdout: str | None,
        stderr: str | None,
    ) -> Path:
        """Collect process stdout/stderr output.

        Args:
            component: Component name
            stdout: Standard output content
            stderr: Standard error content

        Returns:
            Path to collected log file
        """
        dest = self.output_dir / f"{component}_{self._timestamp}.log"

        with open(dest, "w") as outfile:
            if stdout:
                outfile.write("=== STDOUT ===\n")
                outfile.write(stdout)
                outfile.write("\n\n")

            if stderr:
                outfile.write("=== STDERR ===\n")
                outfile.write(stderr)
                outfile.write("\n")

        logger.debug(f"Collected process output for {component} -> {dest}")
        return dest

    def generate_summary(self, log_paths: dict[str, Path]) -> dict[str, Any]:
        """Generate a summary of collected logs.

        Args:
            log_paths: Dict of component_name -> log file path

        Returns:
            Summary dict with file info and error counts
        """
        summary = {
            "timestamp": self._timestamp,
            "output_dir": str(self.output_dir),
            "components": {},
        }

        for component, log_path in log_paths.items():
            if not log_path.exists():
                continue

            log_stat = log_path.stat()
            log_content = log_path.read_text()

            # Count errors and warnings
            error_count = log_content.lower().count("error")
            warning_count = log_content.lower().count("warning")
            exception_count = log_content.lower().count("exception")

            summary["components"][component] = {
                "path": str(log_path),
                "size_bytes": log_stat.st_size,
                "lines": log_content.count("\n"),
                "error_count": error_count,
                "warning_count": warning_count,
                "exception_count": exception_count,
            }

        return summary

    def get_log_tail(self, log_path: Path, lines: int = 50) -> str:
        """Get the last N lines of a log file.

        Args:
            log_path: Path to log file
            lines: Number of lines to return

        Returns:
            Last N lines of the log file
        """
        if not log_path.exists():
            return ""

        try:
            content = log_path.read_text()
            log_lines = content.split("\n")
            return "\n".join(log_lines[-lines:])
        except Exception as e:
            return f"[Error reading log: {e}]"

    def search_logs(
        self,
        log_paths: dict[str, Path],
        pattern: str,
        case_sensitive: bool = False,
    ) -> dict[str, list[str]]:
        """Search for a pattern in collected logs.

        Args:
            log_paths: Dict of component_name -> log file path
            pattern: Pattern to search for
            case_sensitive: Whether search is case-sensitive

        Returns:
            Dict of component_name -> list of matching lines
        """
        import re

        results = {}
        flags = 0 if case_sensitive else re.IGNORECASE

        for component, log_path in log_paths.items():
            if not log_path.exists():
                continue

            try:
                content = log_path.read_text()
                matches = []
                for line in content.split("\n"):
                    if re.search(pattern, line, flags):
                        matches.append(line)
                if matches:
                    results[component] = matches
            except Exception as e:
                logger.debug(f"Error searching {component} log: {e}")

        return results
