"""Instance throughput tracking.

This module provides backward compatibility by re-exporting
from src.utils.throughput_tracker.
"""

from src.utils.throughput_tracker import InstanceThroughputData, ThroughputTracker

__all__ = ["InstanceThroughputData", "ThroughputTracker"]
