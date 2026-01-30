"""Data models for the Planner service.

This package provides domain-specific Pydantic models organized by function.
All models are re-exported here for backward compatibility.
"""

from .base import InstanceStatus
from .instance import (
    InstanceRegisterRequest,
    InstanceRegisterResponse,
)
from .planner import PlannerInput, PlannerOutput
from .pylet import (
    PyLetDeploymentInput,
    PyLetDeploymentOutput,
    PyLetDeployWithPlanInput,
    PyLetInstanceStatus,
    PyLetMigrateInput,
    PyLetMigrateOutput,
    PyLetOptimizeInput,
    PyLetOptimizeOutput,
    PyLetScaleInput,
    PyLetScaleOutput,
    PyLetStatusOutput,
)
from .scheduler import (
    SchedulerDeregisterRequest,
    SchedulerDeregisterResponse,
    SchedulerInfo,
    SchedulerListResponse,
    SchedulerRegisterRequest,
    SchedulerRegisterResponse,
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
    # Scheduler Registry
    "SchedulerRegisterRequest",
    "SchedulerRegisterResponse",
    "SchedulerDeregisterRequest",
    "SchedulerDeregisterResponse",
    "SchedulerInfo",
    "SchedulerListResponse",
    # PyLet
    "PyLetDeploymentInput",
    "PyLetDeploymentOutput",
    "PyLetDeployWithPlanInput",
    "PyLetInstanceStatus",
    "PyLetMigrateInput",
    "PyLetMigrateOutput",
    "PyLetOptimizeInput",
    "PyLetOptimizeOutput",
    "PyLetScaleInput",
    "PyLetScaleOutput",
    "PyLetStatusOutput",
]
