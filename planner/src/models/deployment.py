"""Deployment-related models for the Planner service."""

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from .instance import InstanceInfo
from .planner import PlannerInput, PlannerOutput


class DeploymentStatus(BaseModel):
    """Status of deployment to a single instance."""

    instance_index: int = Field(..., description="Instance index")
    endpoint: str = Field(..., description="Instance endpoint")
    target_model: str = Field(..., description="Target model name")
    previous_model: str | None = Field(None, description="Previous model name")
    success: bool = Field(..., description="Success flag")
    error_message: str | None = Field(None, description="Error details")
    deployment_time: float = Field(..., description="Deployment duration (sec)")


class DeploymentInput(BaseModel):
    """Input for deployment with optimization."""

    instances: list[InstanceInfo] = Field(..., description="Target instances")
    planner_input: PlannerInput = Field(..., description="Optimization config")
    scheduler_mapping: dict[str, str] = Field(
        None, description="mapping of model name to scheduler"
    )
    instance_scheduler_mapping: dict[str, str] | None = Field(
        None, description="mapping of instance endpoint to scheduler URL"
    )

    @model_validator(mode="after")
    def validate_instances_match(self):
        """Validate that number of instances matches M."""
        if len(self.instances) != self.planner_input.M:
            error_msg = f"Number of instances ({len(self.instances)}) must match M ({self.planner_input.M})"
            logger.error(f"DeploymentInput validation failed: {error_msg}")
            raise ValueError(error_msg)
        return self


class DeploymentOutput(PlannerOutput):
    """Output from deployment with execution status."""

    deployment_status: list[DeploymentStatus] = Field(
        ..., description="Per-instance results"
    )
    success: bool = Field(..., description="Overall success flag")
    failed_instances: list[int] = Field(
        ..., description="Failed instance indices"
    )
