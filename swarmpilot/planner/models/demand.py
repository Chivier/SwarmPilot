"""Demand reports the Scheduler sends to the Planner (plan item 3.2)."""

import math

from pydantic import BaseModel, Field, field_validator


class SubmitTargetRequest(BaseModel):
    """Outstanding work for one model, as the Scheduler sees it.

    The fields after ``value`` feed the tail- and call-aware scaler
    (item 3.4); a Scheduler that leaves them out gets the count scaler.

    Attributes:
        model_id: Model the Scheduler serves.
        value: Tasks pending or running for the model.
        arrival_rate: Tasks per second submitted since the last report.
        work_s: Upper (q0.9) estimate of the outstanding work, in
            seconds of one instance's time.
        calls_from: Tasks per second that other models' tasks made to
            this model since the last report, by calling model.
    """

    model_id: str = Field(..., min_length=1)
    value: float = Field(..., ge=0, allow_inf_nan=False)
    arrival_rate: float | None = Field(None, ge=0, allow_inf_nan=False)
    work_s: float | None = Field(None, ge=0, allow_inf_nan=False)
    calls_from: dict[str, float] = Field(default_factory=dict)

    @field_validator("calls_from")
    @classmethod
    def _rates_finite(cls, v: dict[str, float]) -> dict[str, float]:
        for parent, rate in v.items():
            if not parent or not math.isfinite(rate) or rate < 0:
                raise ValueError(f"bad call rate {parent!r}: {rate}")
        return v


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
        arrival_rates: Smoothed tasks per second per model.
        work_s: Smoothed q0.9 outstanding work in seconds per model.
        call_ratios: Calls each task of a model makes to another, as
            ``{caller: {callee: ratio}}``.
    """

    targets: dict[str, float]
    execution_times: dict[str, float]
    target_age_s: dict[str, float]
    arrival_rates: dict[str, float] = Field(default_factory=dict)
    work_s: dict[str, float] = Field(default_factory=dict)
    call_ratios: dict[str, dict[str, float]] = Field(default_factory=dict)
