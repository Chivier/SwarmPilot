"""PyLet-related models for the Planner service."""

from pydantic import BaseModel, Field

from .planner import PlannerInput, PlannerOutput


class PyLetInstanceStatus(BaseModel):
    """Status of a PyLet-managed instance."""

    pylet_id: str = Field(..., description="PyLet instance ID")
    instance_id: str = Field(..., description="Scheduler instance ID")
    model_id: str = Field(..., description="Model identifier")
    endpoint: str | None = Field(None, description="Instance endpoint")
    status: str = Field(..., description="Instance status")
    error: str | None = Field(None, description="Error message if failed")


class PyLetDeploymentInput(BaseModel):
    """Input for PyLet-based deployment."""

    target_state: dict[str, int] = Field(
        ..., description="Target model counts {model_id: count}"
    )
    wait_for_ready: bool = Field(
        True, description="Wait for instances to be ready"
    )
    register_with_scheduler: bool = Field(
        True, description="Register instances with scheduler"
    )


class PyLetDeploymentOutput(BaseModel):
    """Output from PyLet-based deployment."""

    success: bool = Field(..., description="Overall success flag")
    added_count: int = Field(..., description="Number of instances added")
    removed_count: int = Field(..., description="Number of instances removed")
    active_instances: list[PyLetInstanceStatus] = Field(
        ..., description="Currently active instances"
    )
    failed_adds: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Failed add operations (model_id, error)",
    )
    failed_removes: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Failed remove operations (pylet_id, error)",
    )
    error: str | None = Field(None, description="Error message if failed")


class PyLetScaleInput(BaseModel):
    """Input for scaling a specific model via PyLet."""

    model_id: str = Field(..., description="Model to scale")
    target_count: int = Field(..., ge=0, description="Target instance count")
    wait_for_ready: bool = Field(
        True, description="Wait for instances to be ready"
    )


class PyLetScaleOutput(BaseModel):
    """Output from PyLet scaling operation."""

    success: bool = Field(..., description="Overall success flag")
    model_id: str = Field(..., description="Model that was scaled")
    previous_count: int = Field(..., description="Previous instance count")
    current_count: int = Field(..., description="Current instance count")
    added: int = Field(..., description="Number of instances added")
    removed: int = Field(..., description="Number of instances removed")
    active_instances: list[PyLetInstanceStatus] = Field(
        ..., description="Active instances for this model"
    )
    error: str | None = Field(None, description="Error message if failed")


class PyLetMigrateInput(BaseModel):
    """Input for migrating a PyLet instance."""

    pylet_id: str = Field(..., description="PyLet ID of instance to migrate")
    target_model_id: str | None = Field(
        None, description="New model ID (None to keep same)"
    )


class PyLetMigrateOutput(BaseModel):
    """Output from PyLet migration operation."""

    success: bool = Field(..., description="Overall success flag")
    old_pylet_id: str = Field(..., description="Original PyLet ID")
    new_pylet_id: str | None = Field(None, description="New PyLet ID")
    model_id: str = Field(..., description="Model ID")
    endpoint: str | None = Field(None, description="New endpoint")
    error: str | None = Field(None, description="Error message if failed")


class PyLetStatusOutput(BaseModel):
    """Output showing current PyLet deployment status."""

    enabled: bool = Field(..., description="Whether PyLet is enabled")
    initialized: bool = Field(..., description="Whether PyLet is initialized")
    current_state: dict[str, int] = Field(
        ..., description="Current model counts"
    )
    total_instances: int = Field(..., description="Total managed instances")
    active_instances: list[PyLetInstanceStatus] = Field(
        ..., description="All active instances"
    )


class PyLetOptimizeInput(BaseModel):
    """Input for PyLet-based optimization and deployment."""

    target: list[float] = Field(..., description="Target distribution")
    model_ids: list[str] = Field(..., description="Model IDs in order")
    B: list[list[float]] = Field(..., description="Capacity matrix")
    a: float = Field(0.3, ge=0, le=1, description="Change penalty factor")
    objective_method: str = Field(
        "ratio_difference", description="Objective function"
    )
    algorithm: str = Field(
        "simulated_annealing", description="Optimization algorithm"
    )
    wait_for_ready: bool = Field(True, description="Wait for instances")


class PyLetOptimizeOutput(PlannerOutput):
    """Output from PyLet-based optimization."""

    deployment_success: bool = Field(..., description="Deployment success")
    added_count: int = Field(..., description="Instances added")
    removed_count: int = Field(..., description="Instances removed")
    active_instances: list[PyLetInstanceStatus] = Field(
        ..., description="Active instances"
    )
    error: str | None = Field(None, description="Error if failed")


class PyLetDeployWithPlanInput(PlannerInput):
    """Input for /deploy endpoint - combines planning with PyLet deployment.

    Extends PlannerInput with model_ids mapping and PyLet deployment options.
    The optimizer runs first to compute optimal deployment, then PyLet deploys it.
    """

    model_ids: list[str] = Field(
        ...,
        description="Model IDs in order (maps indices 0..N-1 to model names)",
    )
    wait_for_ready: bool = Field(
        True, description="Wait for instances to be ready after deployment"
    )
