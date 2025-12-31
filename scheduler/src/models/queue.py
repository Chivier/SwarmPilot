"""Queue information models for scheduling strategies.

This module defines the queue data structures used by different
scheduling strategies to track instance queue state.
"""

from pydantic import BaseModel


class InstanceQueueBase(BaseModel):
    """Base class for instance queue information."""

    instance_id: str


class InstanceQueueProbabilistic(InstanceQueueBase):
    """Queue information for probabilistic scheduling strategy."""

    quantiles: list[float]
    values: list[float]


class InstanceQueueExpectError(InstanceQueueBase):
    """Queue information for minimum expected time strategy."""

    expected_time_ms: float
    error_margin_ms: float
