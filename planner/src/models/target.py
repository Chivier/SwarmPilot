"""Target and throughput submission models for the Planner service."""

from pydantic import BaseModel, Field


class SubmitTargetRequest(BaseModel):
    """Request model for submitting target queue length from scheduler."""

    model_id: str = Field(..., description="Model identifier from scheduler")
    value: float = Field(..., description="Queue length from scheduler")


class SubmitTargetResponse(BaseModel):
    """Response model for submit_target endpoint."""

    success: bool = Field(
        ..., description="Whether target was updated successfully"
    )
    message: str = Field(..., description="Status message")
    current_target: list[float] | None = Field(
        None, description="Current accumulated target distribution"
    )


class SubmitThroughputRequest(BaseModel):
    """Request model for submitting instance throughput data."""

    instance_url: str = Field(..., description="Instance endpoint URL")
    avg_execution_time: float = Field(
        ..., gt=0, description="Average execution time in seconds (must be > 0)"
    )


class SubmitThroughputResponse(BaseModel):
    """Response model for submit_throughput endpoint."""

    success: bool = Field(
        ..., description="Whether throughput was recorded successfully"
    )
    message: str = Field(..., description="Status message")
    instance_url: str = Field(
        ..., description="Instance URL that was submitted"
    )
    model_id: str | None = Field(
        None,
        description="Model ID determined from instance's current deployment",
    )
    computed_capacity: float | None = Field(
        None, description="Computed processing capacity (1/avg_execution_time)"
    )
