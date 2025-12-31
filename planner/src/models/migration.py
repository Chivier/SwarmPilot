"""Migration-related models for the Planner service."""

from pydantic import BaseModel, Field

from .planner import PlannerOutput


class MigrationStatus(BaseModel):
    """Status of deployment to a single instance."""

    instance_index: int = Field(
        ..., description="Instance index in pending change list"
    )
    original_endpoint: str = Field(
        ..., description="Original instance endpoint before migration"
    )
    target_endpoint: str = Field(
        ..., description="Target instance endpoint after migration"
    )
    target_model: str = Field(..., description="Target model name")
    previous_model: str | None = Field(None, description="Previous model name")
    success: bool = Field(..., description="Success flag")
    error_message: str | None = Field(None, description="Error details")
    deployment_time: float = Field(..., description="Deployment duration (sec)")

    # Deprecated field for backward compatibility
    @property
    def endpoint(self) -> str:
        """Deprecated: use original_endpoint instead."""
        return self.original_endpoint


class MigrationOutput(PlannerOutput):
    """Output from deployment with execution status."""

    deployment_status: list[MigrationStatus] = Field(
        ..., description="Per-instance results"
    )
    success: bool = Field(..., description="Overall success flag")
    failed_instances: list[int] = Field(
        ..., description="Failed instance indices"
    )
