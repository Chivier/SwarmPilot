"""Data models for the Planner service."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from loguru import logger


class PlannerInput(BaseModel):
    """Input parameters for the optimization algorithm."""

    # Core parameters
    M: int = Field(..., description="Number of instances", gt=0)
    N: int = Field(..., description="Number of model types", gt=0)
    B: List[List[float]] = Field(..., description="Batch capacity matrix [M×N]")
    # When deploying the model, inital status will be computed from the instance informations
    initial: Optional[List[int]] = Field(None, description="Initial deployment [M], -1 = no model")
    a: float = Field(..., description="Change constraint (0 < a ≤ 1)", gt=0, le=1)
    target: List[float] = Field(..., description="Target request distribution [N]")

    # Algorithm configuration
    algorithm: Literal["simulated_annealing", "integer_programming"] = Field(
        default="simulated_annealing",
        description="Optimization algorithm to use"
    )
    objective_method: Literal["relative_error", "ratio_difference", "weighted_squared"] = Field(
        default="relative_error",
        description="Objective function method"
    )
    verbose: bool = Field(default=True, description="Enable logging")

    # Simulated Annealing parameters
    initial_temp: float = Field(default=100.0, description="Starting temperature", gt=0)
    final_temp: float = Field(default=0.01, description="Ending temperature", gt=0)
    cooling_rate: float = Field(default=0.95, description="Temperature decay", gt=0, lt=1)
    max_iterations: int = Field(default=5000, description="Max iterations", gt=0)
    iterations_per_temp: int = Field(default=100, description="Iterations per temperature", gt=0)

    # Integer Programming parameters
    solver_name: str = Field(default="PULP_CBC_CMD", description="Solver backend")
    time_limit: int = Field(default=300, description="Timeout (seconds)", gt=0)

    @field_validator("B")
    @classmethod
    def validate_batch_capacity(cls, v, info):
        """Validate batch capacity matrix dimensions."""
        M = info.data.get("M")
        N = info.data.get("N")

        if M is not None and len(v) != M:
            error_msg = f"B must have {M} rows (M instances), got {len(v)}"
            logger.error(f"PlannerInput validation failed: {error_msg}")
            raise ValueError(error_msg)

        if N is not None:
            for i, row in enumerate(v):
                if len(row) != N:
                    error_msg = f"B row {i} must have {N} columns (N models), got {len(row)}"
                    logger.error(f"PlannerInput validation failed: {error_msg}")
                    raise ValueError(error_msg)
                if any(val < 0 for val in row):
                    error_msg = f"B row {i} contains negative values"
                    logger.error(f"PlannerInput validation failed: {error_msg}")
                    raise ValueError(error_msg)

        return v

    @field_validator("initial")
    @classmethod
    def validate_initial(cls, v, info):
        """Validate initial deployment array."""
        if v is None:
            return v

        M = info.data.get("M")
        N = info.data.get("N")

        if M is not None and len(v) != M:
            error_msg = f"initial must have length {M}, got {len(v)}"
            logger.error(f"PlannerInput validation failed: {error_msg}")
            raise ValueError(error_msg)

        if N is not None:
            for i, model_id in enumerate(v):
                # Valid values: -1 (no model) or 0 to N-1 (valid model IDs)
                if model_id != -1 and (model_id < 0 or model_id >= N):
                    error_msg = f"initial[{i}] has invalid model ID {model_id}, must be -1 or in range [0, {N-1}]"
                    logger.error(f"PlannerInput validation failed: {error_msg}")
                    raise ValueError(error_msg)

        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v, info):
        """Validate target distribution."""
        N = info.data.get("N")

        if N is not None and len(v) != N:
            error_msg = f"target must have length {N}, got {len(v)}"
            logger.error(f"PlannerInput validation failed: {error_msg}")
            raise ValueError(error_msg)

        if any(val < 0 for val in v):
            error_msg = "target contains negative values"
            logger.error(f"PlannerInput validation failed: {error_msg}")
            raise ValueError(error_msg)

        return v

    @model_validator(mode="after")
    def validate_temperature_range(self):
        """Validate temperature parameters."""
        if self.final_temp >= self.initial_temp:
            error_msg = f"final_temp ({self.final_temp}) must be less than initial_temp ({self.initial_temp})"
            logger.error(f"PlannerInput validation failed: {error_msg}")
            raise ValueError(error_msg)
        return self


class PlannerOutput(BaseModel):
    """Output from the optimization algorithm."""

    deployment: List[int] = Field(..., description="Optimized assignment [M] (model IDs)")
    score: float = Field(..., description="Objective value (lower = better)")
    stats: Dict[str, Any] = Field(..., description="Algorithm statistics")
    service_capacity: List[float] = Field(..., description="Capacity per model [N]")
    changes_count: int = Field(..., description="Changes from initial state")


class InstanceInfo(BaseModel):
    """Information about a target instance."""

    endpoint: str = Field(..., description="Instance API endpoint")
    current_model: str = Field(..., description="Current model name")


class DeploymentInput(BaseModel):
    """Input for deployment with optimization."""

    instances: List[InstanceInfo] = Field(..., description="Target instances")
    planner_input: PlannerInput = Field(..., description="Optimization config")
    scheduler_mapping: Dict[str, str] = Field(None, description="mapping of model name to scheduler")
    instance_scheduler_mapping: Optional[Dict[str, str]] = Field(None, description="mapping of instance endpoint to scheduler URL")

    @model_validator(mode="after")
    def validate_instances_match(self):
        """Validate that number of instances matches M."""
        if len(self.instances) != self.planner_input.M:
            error_msg = f"Number of instances ({len(self.instances)}) must match M ({self.planner_input.M})"
            logger.error(f"DeploymentInput validation failed: {error_msg}")
            raise ValueError(error_msg)
        return self


class DeploymentStatus(BaseModel):
    """Status of deployment to a single instance."""

    instance_index: int = Field(..., description="Instance index")
    endpoint: str = Field(..., description="Instance endpoint")
    target_model: str = Field(..., description="Target model name")
    previous_model: Optional[str] = Field(None, description="Previous model name")
    success: bool = Field(..., description="Success flag")
    error_message: Optional[str] = Field(None, description="Error details")
    deployment_time: float = Field(..., description="Deployment duration (sec)")

class MigrationStatus(BaseModel):
    """Status of deployment to a single instance."""
    instance_index: int = Field(..., description="Instance index in pending change list")
    original_endpoint: str = Field(..., description="Original instance endpoint before migration")
    target_endpoint: str = Field(..., description="Target instance endpoint after migration")
    target_model: str = Field(..., description="Target model name")
    previous_model: Optional[str] = Field(None, description="Previous model name")
    success: bool = Field(..., description="Success flag")
    error_message: Optional[str] = Field(None, description="Error details")
    deployment_time: float = Field(..., description="Deployment duration (sec)")

    # Deprecated field for backward compatibility
    @property
    def endpoint(self) -> str:
        """Deprecated: use original_endpoint instead."""
        return self.original_endpoint



class DeploymentOutput(PlannerOutput):
    """Output from deployment with execution status."""

    deployment_status: List[DeploymentStatus] = Field(..., description="Per-instance results")
    success: bool = Field(..., description="Overall success flag")
    failed_instances: List[int] = Field(..., description="Failed instance indices")

class MigrationOutput(PlannerOutput):
    """Output from deployment with execution status."""

    deployment_status: List[MigrationStatus] = Field(..., description="Per-instance results")
    success: bool = Field(..., description="Overall success flag")
    failed_instances: List[int] = Field(..., description="Failed instance indices")


class InstanceRegisterRequest(BaseModel):
    """Request model for instance registration (compatible with scheduler)."""
    instance_id: str = Field(..., description="Unique instance identifier")
    model_id: str = Field(..., description="Model ID supported by this instance")
    endpoint: str = Field(..., description="Instance HTTP endpoint URL")
    platform_info: Dict[str, str] = Field(
        default_factory=dict,
        description="Platform info (software_name, software_version, hardware_name)"
    )


class InstanceRegisterResponse(BaseModel):
    """Response model for instance registration."""
    success: bool = Field(..., description="Whether registration was successful")
    message: str = Field(..., description="Status message")


class SubmitTargetRequest(BaseModel):
    """Request model for submitting target queue length from scheduler."""
    model_id: str = Field(..., description="Model identifier from scheduler")
    value: float = Field(..., description="Queue length from scheduler")


class SubmitTargetResponse(BaseModel):
    """Response model for submit_target endpoint."""
    success: bool = Field(..., description="Whether target was updated successfully")
    message: str = Field(..., description="Status message")
    current_target: Optional[List[float]] = Field(None, description="Current accumulated target distribution")
