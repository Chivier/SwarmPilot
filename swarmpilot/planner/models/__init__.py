"""Data models for the Planner service.

This package provides domain-specific Pydantic models organized by function.
All models are re-exported here for backward compatibility.
"""

from swarmpilot.shared.models import (
    InstanceDrainRequest,
    InstanceDrainResponse,
    InstanceDrainStatusResponse,
    InstanceRemoveRequest,
    InstanceRemoveResponse,
    InstanceStatus,
    TaskResubmitRequest,
    TaskResubmitResponse,
)

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
from .sdk_api import (
    DeployResponse,
    InstanceDetailResponse,
    RegisteredModelsResponse,
    RegisterRequest,
    RunRequest,
    RunResponse,
    ScaleRequest,
    ScaleResponse,
    SchedulerMapResponse,
    ServeRequest,
    ServeResponse,
    TerminateRequest,
    TerminateResponse,
)

__all__ = [
    "DeployResponse",
    "InstanceDetailResponse",
    "InstanceDrainRequest",
    "InstanceDrainResponse",
    "InstanceDrainStatusResponse",
    "InstanceRegisterRequest",
    "InstanceRegisterResponse",
    "InstanceRemoveRequest",
    "InstanceRemoveResponse",
    "InstanceStatus",
    "PlannerInput",
    "PlannerOutput",
    "PyLetDeployWithPlanInput",
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
    "RegisterRequest",
    "RegisteredModelsResponse",
    "RunRequest",
    "RunResponse",
    "ScaleRequest",
    "ScaleResponse",
    "SchedulerDeregisterRequest",
    "SchedulerDeregisterResponse",
    "SchedulerInfo",
    "SchedulerListResponse",
    "SchedulerMapResponse",
    "SchedulerRegisterRequest",
    "SchedulerRegisterResponse",
    "ServeRequest",
    "ServeResponse",
    "TaskResubmitRequest",
    "TaskResubmitResponse",
    "TerminateRequest",
    "TerminateResponse",
]
