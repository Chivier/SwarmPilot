"""API request models for the scheduler.

This module defines all Pydantic models used for API request validation.
"""

from typing import Any

from pydantic import BaseModel, Field

from swarmpilot.scheduler.models.status import StrategyType

# ============================================================================
# Instance Management Requests
# ============================================================================


class InstanceRegisterRequest(BaseModel):
    """Request model for instance registration."""

    instance_id: str
    model_id: str
    endpoint: str
    platform_info: dict[
        str, str
    ]  # Required: software_name, software_version, hardware_name


class InstanceRemoveRequest(BaseModel):
    """Request model for instance removal."""

    instance_id: str


class InstanceDrainRequest(BaseModel):
    """Request model for starting instance draining."""

    instance_id: str


class InstanceRedeployRequest(BaseModel):
    """Request model for instance redeployment."""

    instance_id: str = Field(..., description="ID of the instance to redeploy")
    redeploy_reason: str | None = Field(None, description="Reason for redeployment")
    target_model_id: str | None = Field(
        None, description="Optional target model ID for redeployment"
    )


# ============================================================================
# Task Management Requests
# ============================================================================


class TaskSubmitRequest(BaseModel):
    """Request model for task submission."""

    task_id: str
    model_id: str
    task_input: dict[str, Any]
    metadata: dict[str, Any]



class TaskResubmitRequest(BaseModel):
    """Request model for task resubmission during instance migration."""

    task_id: str = Field(..., description="ID of the task to resubmit")
    original_instance_id: str = Field(
        ..., description="ID of the original instance for updating statistics"
    )


class TaskMetadataUpdate(BaseModel):
    """Single task metadata update."""

    task_id: str = Field(..., description="ID of the task to update")
    metadata: dict[str, Any] = Field(
        ..., description="New metadata for the task (replaces existing)"
    )


class TaskUpdateMetadataRequest(BaseModel):
    """Request model for batch task metadata update."""

    updates: list[TaskMetadataUpdate] = Field(
        ..., description="List of task metadata updates"
    )


# ============================================================================
# Strategy Management Requests
# ============================================================================


class StrategySetRequest(BaseModel):
    """Request model for setting scheduling strategy."""

    strategy_name: StrategyType = Field(
        ..., description="Name of the scheduling strategy to use"
    )
    target_quantile: float | None = Field(
        None,
        description="Target quantile for probabilistic strategy (0.0 < q < 1.0, default: 0.9)",
        gt=0.0,
        lt=1.0,
    )
    quantiles: list[float] | None = Field(
        None,
        description="Custom quantiles for probabilistic strategy (default: [0.5, 0.9, 0.95, 0.99])",
        min_length=1,
        max_length=100,
    )


# ============================================================================
# Predictor Management Requests
# ============================================================================


class PredictorTrainRequest(BaseModel):
    """Request model for triggering predictor training."""

    model_id: str = Field(
        ..., description="Model identifier to train"
    )
    platform_info: dict[str, str] | None = Field(
        None,
        description=(
            "Platform info (software_name, software_version,"
            " hardware_name)"
        ),
    )
    prediction_type: str = Field(
        "expect_error",
        description=(
            "Prediction type: 'expect_error' or 'quantile'"
        ),
    )


class PredictorPredictRequest(BaseModel):
    """Request model for manual prediction via the scheduler."""

    model_id: str = Field(
        ..., description="Model identifier"
    )
    platform_info: dict[str, str] = Field(
        ...,
        description=(
            "Platform info (software_name, software_version,"
            " hardware_name)"
        ),
    )
    features: dict[str, Any] = Field(
        ..., description="Feature values for prediction"
    )
    prediction_type: str = Field(
        "expect_error",
        description=(
            "Prediction type: 'expect_error' or 'quantile'"
        ),
    )


# ============================================================================
# Callback Requests (Instance -> Scheduler)
# ============================================================================


class TaskResultCallbackRequest(BaseModel):
    """Request model for task result callback from instance to scheduler."""

    task_id: str = Field(..., description="ID of the completed task")
    status: str = Field(..., description="Task status: 'completed' or 'failed'")
    result: dict[str, Any] | None = Field(
        None, description="Task result data (if completed)"
    )
    error: str | None = Field(None, description="Error message (if failed)")
    execution_time_ms: float | None = Field(
        None, description="Execution time in milliseconds"
    )
