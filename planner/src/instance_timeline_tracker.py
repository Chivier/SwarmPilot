"""Instance count timeline tracker for redeploy events.

This module tracks instance counts per model during Planner redeployment events.
It provides a timeline of how instance allocations change over time when
auto-optimization or explicit migration calls occur.
"""

import json
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class TimelineEntry:
    """Single timeline entry recording instance counts at a point in time."""

    timestamp: float
    timestamp_iso: str
    event_type: str  # "deploy_migration" or "auto_optimize"
    instance_counts: Dict[str, int]
    total_instances: int
    changes_count: int
    success: bool
    target_distribution: Optional[List[float]] = None
    score: Optional[float] = None


class InstanceTimelineTracker:
    """Thread-safe tracker for instance count timeline with JSON persistence."""

    def __init__(self, output_path: str = "./logs/instance_count_timeline.json"):
        """Initialize the timeline tracker.

        Args:
            output_path: Path to the JSON file for persisting timeline data.
        """
        self._output_path = Path(output_path)
        self._entries: List[TimelineEntry] = []
        self._lock = threading.Lock()
        self._created_at = datetime.now(timezone.utc).isoformat()

        # Ensure output directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def record_migration(
        self,
        event_type: str,
        instance_counts: Dict[str, int],
        changes_count: int,
        success: bool,
        target_distribution: Optional[List[float]] = None,
        score: Optional[float] = None,
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
        entry = TimelineEntry(
            timestamp=now,
            timestamp_iso=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            event_type=event_type,
            instance_counts=instance_counts,
            total_instances=sum(instance_counts.values()),
            changes_count=changes_count,
            success=success,
            target_distribution=target_distribution,
            score=score,
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
            self._created_at = datetime.now(timezone.utc).isoformat()
            self._persist()
        logger.info("Timeline cleared")

    def get_entries(self) -> List[dict]:
        """Get all timeline entries as dictionaries."""
        with self._lock:
            return [asdict(e) for e in self._entries]

    def get_entry_count(self) -> int:
        """Get the number of timeline entries."""
        with self._lock:
            return len(self._entries)


# Global singleton
_tracker: Optional[InstanceTimelineTracker] = None
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


def compute_instance_counts(instances) -> Dict[str, int]:
    """Compute instance counts per model from instances list.

    Args:
        instances: List of InstanceInfo objects with current_model attribute.

    Returns:
        Dict mapping model_id to count of instances running that model.
    """
    counts: Dict[str, int] = {}
    for inst in instances:
        model = inst.current_model
        counts[model] = counts.get(model, 0) + 1
    return counts
