"""Demand reports the Scheduler sends to the Planner (plan item 3.2)."""

from pydantic import BaseModel, Field


class SubmitTargetRequest(BaseModel):
    """Outstanding work for one model, as the Scheduler sees it.

    Attributes:
        model_id: Model the Scheduler serves.
        value: Tasks pending or running for the model.
    """

    model_id: str = Field(..., min_length=1)
    value: float = Field(..., ge=0, allow_inf_nan=False)


class SubmitThroughputRequest(BaseModel):
    """Recent average execution time of one instance.

    Attributes:
        instance_url: Endpoint of the instance.
        avg_execution_time: Mean task execution time in seconds.
    """

    instance_url: str = Field(..., min_length=1)
    avg_execution_time: float = Field(..., gt=0, allow_inf_nan=False)


class DemandReportResponse(BaseModel):
    """Acknowledgement of a demand report.

    Attributes:
        success: Whether the report was stored.
        smoothed: The smoothed value now held for the model or instance.
    """

    success: bool = True
    smoothed: float


class DemandSnapshot(BaseModel):
    """What the Planner currently knows about demand.

    Attributes:
        targets: Smoothed outstanding work per model.
        execution_times: Smoothed execution time in seconds per instance.
        target_age_s: Seconds since each model last reported.
    """

    targets: dict[str, float]
    execution_times: dict[str, float]
    target_age_s: dict[str, float]
