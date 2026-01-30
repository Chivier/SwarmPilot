"""Scheduler registry models for multi-scheduler architecture (PYLET-024).

This module provides Pydantic models for scheduler registration and
management, enabling per-model scheduler routing.
"""

from pydantic import BaseModel


class SchedulerRegisterRequest(BaseModel):
    """Request to register a scheduler with the planner.

    Attributes:
        model_id: Model identifier this scheduler handles.
        scheduler_url: Base URL of the scheduler service.
        metadata: Optional metadata about the scheduler.
    """

    model_id: str
    scheduler_url: str
    metadata: dict[str, str] = {}


class SchedulerRegisterResponse(BaseModel):
    """Response from scheduler registration.

    Attributes:
        success: Whether registration succeeded.
        message: Status message.
        replaced_previous: Whether a previous registration was replaced.
    """

    success: bool
    message: str
    replaced_previous: bool = False


class SchedulerDeregisterRequest(BaseModel):
    """Request to deregister a scheduler.

    Attributes:
        model_id: Model identifier to deregister.
    """

    model_id: str


class SchedulerDeregisterResponse(BaseModel):
    """Response from scheduler deregistration.

    Attributes:
        success: Whether deregistration succeeded.
        message: Status message.
    """

    success: bool
    message: str


class SchedulerInfo(BaseModel):
    """Information about a registered scheduler.

    Attributes:
        model_id: Model identifier this scheduler handles.
        scheduler_url: Base URL of the scheduler service.
        registered_at: ISO timestamp of registration.
        is_healthy: Whether the scheduler is currently healthy.
        metadata: Optional metadata about the scheduler.
    """

    model_id: str
    scheduler_url: str
    registered_at: str
    is_healthy: bool = True
    metadata: dict[str, str] = {}


class SchedulerListResponse(BaseModel):
    """Response listing all registered schedulers.

    Attributes:
        schedulers: List of registered scheduler info.
        total: Total number of registered schedulers.
    """

    schedulers: list[SchedulerInfo]
    total: int
