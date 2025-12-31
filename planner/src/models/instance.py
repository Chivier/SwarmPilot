"""Instance-related models for the Planner service."""

from pydantic import BaseModel, Field


class InstanceInfo(BaseModel):
    """Information about a target instance."""

    endpoint: str = Field(..., description="Instance API endpoint")
    current_model: str = Field(..., description="Current model name")


class InstanceRegisterRequest(BaseModel):
    """Request model for instance registration (compatible with scheduler)."""

    instance_id: str = Field(..., description="Unique instance identifier")
    model_id: str = Field(
        ..., description="Model ID supported by this instance"
    )
    endpoint: str = Field(..., description="Instance HTTP endpoint URL")
    platform_info: dict[str, str] = Field(
        default_factory=dict,
        description="Platform info (software_name, software_version, hardware_name)",
    )


class InstanceRegisterResponse(BaseModel):
    """Response model for instance registration."""

    success: bool = Field(
        ..., description="Whether registration was successful"
    )
    message: str = Field(..., description="Status message")
