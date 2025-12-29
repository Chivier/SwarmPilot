"""Instance count timeline tracker for redeploy events.

This module tracks instance counts per model during Planner redeployment events.
It provides a timeline of how instance allocations change over time when
auto-optimization or explicit migration calls occur.
"""

import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


@dataclass
class TimelineEntry:
    """Single timeline entry recording instance counts at a point in time."""

    timestamp: float
    timestamp_iso: str
    event_type: str  # "deploy_migration" or "auto_optimize"
    instance_counts: dict[str, int]
    total_instances: int
    changes_count: int
    success: bool
    target_distribution: list[float] | None = None
    score: float | None = None


class InstanceTimelineTracker:
    """Thread-safe tracker for instance count timeline with JSON persistence."""

    def __init__(
        self, output_path: str = "./logs/instance_count_timeline.json"
    ):
        """Initialize the timeline tracker.

        Args:
            output_path: Path to the JSON file for persisting timeline data.
        """
        self._output_path = Path(output_path)
        self._entries: list[TimelineEntry] = []
        self._lock = threading.Lock()
        self._created_at = datetime.now(UTC).isoformat()

        # Ensure output directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def record_migration(
        self,
        event_type: str,
        instance_counts: dict[str, int],
        changes_count: int,
        success: bool,
        target_distribution: list[float] | None = None,
        score: float | None = None,
    ) -> None:
        """Record a migration event to the timeline.

        Args:
            event_type: Type of event ("deploy_migration" or "auto_optimize")
            instance_counts: Dict mapping model_id to instance count
            changes_count: Number of instance model changes in this migration
            success: Whether the migration was successful
            target_distribution: Target distribution used for optimization
            score: Optimization score achieved
        """
        now = time.time()

        # Convert numpy types to Python native types for JSON serialization
        # This prevents "Object of type int64 is not JSON serializable" errors
        instance_counts_native = {k: int(v) for k, v in instance_counts.items()}
        changes_count_native = int(changes_count)
        target_distribution_native = (
            [float(x) for x in target_distribution]
            if target_distribution
            else None
        )

        # Handle special float values (inf, -inf, nan) which are not valid JSON
        if score is not None:
            score_float = float(score)
            # Convert inf/-inf/nan to None for valid JSON
            if math.isinf(score_float) or math.isnan(score_float):
                score_native = None
            else:
                score_native = score_float
        else:
            score_native = None

        entry = TimelineEntry(
            timestamp=now,
            timestamp_iso=datetime.fromtimestamp(now, tz=UTC).isoformat(),
            event_type=event_type,
            instance_counts=instance_counts_native,
            total_instances=sum(instance_counts_native.values()),
            changes_count=changes_count_native,
            success=success,
            target_distribution=target_distribution_native,
            score=score_native,
        )

        with self._lock:
            self._entries.append(entry)
            self._persist()

        logger.info(
            f"Timeline: recorded {event_type} - {instance_counts}, "
            f"changes={changes_count}, success={success}"
        )

    def _persist(self) -> None:
        """Persist timeline to JSON file (called with lock held)."""
        data = {
            "version": "1.0",
            "created_at": self._created_at,
            "entries": [asdict(e) for e in self._entries],
        }
        with open(self._output_path, "w") as f:
            json.dump(data, f, indent=2)

    def clear(self) -> None:
        """Clear timeline for new experiment run."""
        with self._lock:
            self._entries.clear()
            self._created_at = datetime.now(UTC).isoformat()
            self._persist()
        logger.info("Timeline cleared")

    def get_entries(self) -> list[dict]:
        """Get all timeline entries as dictionaries."""
        with self._lock:
            return [asdict(e) for e in self._entries]

    def get_entry_count(self) -> int:
        """Get the number of timeline entries."""
        with self._lock:
            return len(self._entries)


# Global singleton
_tracker: InstanceTimelineTracker | None = None
_tracker_lock = threading.Lock()


def get_timeline_tracker(
    output_path: str = "./logs/instance_count_timeline.json",
) -> InstanceTimelineTracker:
    """Get or create the global timeline tracker.

    Args:
        output_path: Path to the JSON file for persisting timeline data.
                     Only used when creating a new tracker instance.

    Returns:
        The global InstanceTimelineTracker instance.
    """
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = InstanceTimelineTracker(output_path)
        return _tracker


def compute_instance_counts(instances) -> dict[str, int]:
    """Compute instance counts per model from instances list.

    Args:
        instances: List of InstanceInfo objects with current_model attribute.

    Returns:
        Dict mapping model_id to count of instances running that model.
    """
    counts: dict[str, int] = {}
    for inst in instances:
        model = inst.current_model
        counts[model] = counts.get(model, 0) + 1
    return counts
