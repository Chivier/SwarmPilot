"""Throughput tracker for tracking per-instance execution times.

Uses a sliding window to maintain recent execution times per instance
and provides average execution time calculations for planner reporting.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field


@dataclass
class InstanceThroughputData:
    """Sliding window data for a single instance."""

    instance_endpoint: str
    window_size: int = 20
    execution_times_ms: deque = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self):
        """Initialize deque with correct maxlen."""
        self.execution_times_ms = deque(maxlen=self.window_size)

    def add_execution_time(self, time_ms: float) -> None:
        """Add an execution time to the sliding window."""
        self.execution_times_ms.append(time_ms)

    def get_average_ms(self) -> float | None:
        """Get average execution time in milliseconds."""
        if not self.execution_times_ms:
            return None
        return sum(self.execution_times_ms) / len(self.execution_times_ms)

    def get_sample_count(self) -> int:
        """Get number of samples in the window."""
        return len(self.execution_times_ms)


class ThroughputTracker:
    """Thread-safe tracker for per-instance execution times."""

    def __init__(self, window_size: int = 20):
        """Initialize the throughput tracker.

        Args:
            window_size: Size of the sliding window for each instance
        """
        self._window_size = window_size
        self._instances: dict[str, InstanceThroughputData] = {}
        self._instances_with_new_data: set = (
            set()
        )  # Track instances with data since last report
        self._lock = asyncio.Lock()

    async def record_execution_time(
        self, instance_endpoint: str, execution_time_ms: float
    ) -> None:
        """Record an execution time for an instance.

        Args:
            instance_endpoint: The instance's endpoint URL
            execution_time_ms: Execution time in milliseconds
        """
        async with self._lock:
            if instance_endpoint not in self._instances:
                self._instances[instance_endpoint] = InstanceThroughputData(
                    instance_endpoint=instance_endpoint,
                    window_size=self._window_size,
                )
            self._instances[instance_endpoint].add_execution_time(execution_time_ms)
            # Mark this instance as having new data since last report
            self._instances_with_new_data.add(instance_endpoint)

    async def get_average_execution_time_seconds(
        self, instance_endpoint: str
    ) -> float | None:
        """Get average execution time for an instance in seconds.

        Args:
            instance_endpoint: The instance's endpoint URL

        Returns:
            Average execution time in seconds, or None if no data
        """
        async with self._lock:
            if instance_endpoint not in self._instances:
                return None
            avg_ms = self._instances[instance_endpoint].get_average_ms()
            if avg_ms is None:
                return None
            return avg_ms / 1000.0  # Convert ms to seconds

    async def get_all_averages_seconds(self) -> dict[str, float]:
        """Get average execution times for all instances in seconds.

        Returns:
            Dict mapping instance endpoints to average execution time in seconds
        """
        async with self._lock:
            result = {}
            for endpoint, data in self._instances.items():
                avg_ms = data.get_average_ms()
                if avg_ms is not None:
                    result[endpoint] = avg_ms / 1000.0
            return result

    async def get_averages_for_recent_instances_seconds(
        self,
    ) -> dict[str, float]:
        """Get average execution times only for instances with new data since last report.

        This method returns averages for instances that have recorded at least one
        execution time since the previous call to this method. After returning,
        the "new data" tracking is cleared.

        Returns:
            Dict mapping instance endpoints to average execution time in seconds,
            only for instances with recent data
        """
        async with self._lock:
            result = {}
            for endpoint in self._instances_with_new_data:
                if endpoint in self._instances:
                    avg_ms = self._instances[endpoint].get_average_ms()
                    if avg_ms is not None:
                        result[endpoint] = avg_ms / 1000.0
            # Clear the tracking set after report
            self._instances_with_new_data.clear()
            return result

    async def remove_instance(self, instance_endpoint: str) -> None:
        """Remove tracking data for an instance.

        Args:
            instance_endpoint: The instance's endpoint URL
        """
        async with self._lock:
            if instance_endpoint in self._instances:
                del self._instances[instance_endpoint]
            self._instances_with_new_data.discard(instance_endpoint)

    async def clear_all(self) -> int:
        """Clear all tracking data.

        Returns:
            Number of instances cleared
        """
        async with self._lock:
            count = len(self._instances)
            self._instances.clear()
            self._instances_with_new_data.clear()
            return count
