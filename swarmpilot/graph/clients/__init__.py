"""
Client modules for interacting with SwarmPilot services.

This package provides clients for:
- Scheduler Service: Task routing and instance management
- Predictor Service: Inference time prediction
"""

from .scheduler_client import SchedulerClient

__all__ = [
    "SchedulerClient",
]
