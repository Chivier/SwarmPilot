"""Planner input/output models for the optimization algorithm."""

from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


class PlannerInput(BaseModel):
    """Input parameters for the optimization algorithm."""

    # Core parameters
    M: int = Field(..., description="Number of instances", gt=0)
    N: int = Field(..., description="Number of model types", gt=0)
    B: list[list[float]] = Field(..., description="Batch capacity matrix [M×N]")
    # When deploying the model, inital status will be computed from the instance informations
    initial: list[int] | None = Field(
        None, description="Initial deployment [M], -1 = no model"
    )
    a: float = Field(
        ..., description="Change constraint (0 < a ≤ 1)", gt=0, le=1
    )
    target: list[float] = Field(
        ..., description="Target request distribution [N]"
    )

    # Algorithm configuration
    algorithm: Literal["simulated_annealing", "integer_programming"] = Field(
        default="simulated_annealing",
        description="Optimization algorithm to use",
    )
    objective_method: Literal[
        "relative_error", "ratio_difference", "weighted_squared"
    ] = Field(default="relative_error", description="Objective function method")
    verbose: bool = Field(default=True, description="Enable logging")

    # Simulated Annealing parameters
    initial_temp: float = Field(
        default=100.0, description="Starting temperature", gt=0
    )
    final_temp: float = Field(
        default=0.01, description="Ending temperature", gt=0
    )
    cooling_rate: float = Field(
        default=0.95, description="Temperature decay", gt=0, lt=1
    )
    max_iterations: int = Field(
        default=5000, description="Max iterations", gt=0
    )
    iterations_per_temp: int = Field(
        default=100, description="Iterations per temperature", gt=0
    )

    # Integer Programming parameters
    solver_name: str = Field(
        default="PULP_CBC_CMD", description="Solver backend"
    )
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
                    error_msg = f"initial[{i}] has invalid model ID {model_id}, must be -1 or in range [0, {N - 1}]"
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

    deployment: list[int] = Field(
        ..., description="Optimized assignment [M] (model IDs)"
    )
    score: float = Field(..., description="Objective value (lower = better)")
    stats: dict[str, Any] = Field(..., description="Algorithm statistics")
    service_capacity: list[float] = Field(
        ..., description="Capacity per model [N]"
    )
    changes_count: int = Field(..., description="Changes from initial state")
