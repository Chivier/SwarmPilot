"""Data models for the Planner service.

This package provides domain-specific Pydantic models organized by function.
All models are re-exported here for backward compatibility.
"""

from .base import InstanceStatus
from .instance import (
    InstanceInfo,
    InstanceRegisterRequest,
    InstanceRegisterResponse,
)
from .planner import PlannerInput, PlannerOutput
from .pylet import (
    PyLetDeploymentInput,
    PyLetDeploymentOutput,
    PyLetInstanceStatus,
    PyLetMigrateInput,
    PyLetMigrateOutput,
    PyLetOptimizeInput,
    PyLetOptimizeOutput,
    PyLetScaleInput,
    PyLetScaleOutput,
    PyLetStatusOutput,
)
from .scheduler_compat import (
    InstanceDrainRequest,
    InstanceDrainResponse,
    InstanceDrainStatusResponse,
    InstanceRemoveRequest,
    InstanceRemoveResponse,
    TaskResubmitRequest,
    TaskResubmitResponse,
)

__all__ = [
    # Base
    "InstanceStatus",
    # Planner
    "PlannerInput",
    "PlannerOutput",
    # Instance
    "InstanceInfo",
    "InstanceRegisterRequest",
    "InstanceRegisterResponse",
    # Scheduler Compat
    "InstanceDrainRequest",
    "InstanceDrainResponse",
    "InstanceDrainStatusResponse",
    "InstanceRemoveRequest",
    "InstanceRemoveResponse",
    "TaskResubmitRequest",
    "TaskResubmitResponse",
    # PyLet
    "PyLetDeploymentInput",
    "PyLetDeploymentOutput",
    "PyLetInstanceStatus",
    "PyLetMigrateInput",
    "PyLetMigrateOutput",
    "PyLetOptimizeInput",
    "PyLetOptimizeOutput",
    "PyLetScaleInput",
    "PyLetScaleOutput",
    "PyLetStatusOutput",
]
