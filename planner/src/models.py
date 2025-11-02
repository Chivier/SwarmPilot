"""Data models for the Planner service."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class PlannerInput(BaseModel):
    """Input parameters for the optimization algorithm."""

    # Core parameters
    M: int = Field(..., description="Number of instances", gt=0)
    N: int = Field(..., description="Number of model types", gt=0)
    B: List[List[float]] = Field(..., description="Batch capacity matrix [M×N]")
    initial: List[int] = Field(..., description="Initial deployment [M], -1 = no model")
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
            raise ValueError(f"B must have {M} rows (M instances), got {len(v)}")

        if N is not None:
            for i, row in enumerate(v):
                if len(row) != N:
                    raise ValueError(f"B row {i} must have {N} columns (N models), got {len(row)}")
                if any(val < 0 for val in row):
                    raise ValueError(f"B row {i} contains negative values")

        return v

    @field_validator("initial")
    @classmethod
    def validate_initial(cls, v, info):
        """Validate initial deployment."""
        M = info.data.get("M")
        N = info.data.get("N")

        if M is not None and len(v) != M:
            raise ValueError(f"initial must have length {M}, got {len(v)}")

        if N is not None:
            for i, val in enumerate(v):
                if val != -1 and (val < 0 or val >= N):
                    raise ValueError(f"initial[{i}] = {val} is invalid (must be -1 or 0..{N-1})")

        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v, info):
        """Validate target distribution."""
        N = info.data.get("N")

        if N is not None and len(v) != N:
            raise ValueError(f"target must have length {N}, got {len(v)}")

        if any(val < 0 for val in v):
            raise ValueError("target contains negative values")

        return v

    @model_validator(mode="after")
    def validate_temperature_range(self):
        """Validate temperature parameters."""
        if self.final_temp >= self.initial_temp:
            raise ValueError("final_temp must be less than initial_temp")
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

    @model_validator(mode="after")
    def validate_instances_match(self):
        """Validate that number of instances matches M."""
        if len(self.instances) != self.planner_input.M:
            raise ValueError(
                f"Number of instances ({len(self.instances)}) must match M ({self.planner_input.M})"
            )
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


class DeploymentOutput(PlannerOutput):
    """Output from deployment with execution status."""

    deployment_status: List[DeploymentStatus] = Field(..., description="Per-instance results")
    success: bool = Field(..., description="Overall success flag")
    failed_instances: List[int] = Field(..., description="Failed instance indices")
