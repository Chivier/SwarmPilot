"""Data models for the Planner service.

This package provides domain-specific Pydantic models organized by function.
All models are re-exported here for backward compatibility.
"""

from .base import InstanceStatus
from .deployment import DeploymentInput, DeploymentOutput, DeploymentStatus
from .instance import (
    InstanceInfo,
    InstanceRegisterRequest,
    InstanceRegisterResponse,
)
from .migration import MigrationOutput, MigrationStatus
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
from .target import (
    SubmitTargetRequest,
    SubmitTargetResponse,
    SubmitThroughputRequest,
    SubmitThroughputResponse,
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
    # Deployment
    "DeploymentInput",
    "DeploymentOutput",
    "DeploymentStatus",
    # Migration
    "MigrationStatus",
    "MigrationOutput",
    # Target
    "SubmitTargetRequest",
    "SubmitTargetResponse",
    "SubmitThroughputRequest",
    "SubmitThroughputResponse",
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
