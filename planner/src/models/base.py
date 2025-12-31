"""Base models and enums for the Planner service."""

from enum import Enum


class InstanceStatus(str, Enum):
    """Enumeration of possible instance statuses (compatible with Scheduler)."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    DRAINING = "draining"
    REMOVING = "removing"
    REDEPLOYING = "redeploying"
