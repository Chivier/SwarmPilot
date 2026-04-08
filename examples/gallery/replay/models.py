"""Pydantic data models for the replay latency benchmark system."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelEndpoint(BaseModel):
    """Configuration for one LLM endpoint."""

    base_url: str
    api_key: str
    model: str


class PlannerConfig(BaseModel):
    """Planner-based deployment configuration.

    When present in ExperimentConfig, the runner automatically deploys
    models via ``SwarmPilotClient.serve()`` before the experiment and
    terminates them after.

    Args:
        url: Planner HTTP endpoint.
        large_model_id: HuggingFace model ID for the large model.
        small_model_id: HuggingFace model ID for the small model.
        gpu_per_instance_large: GPUs per large-model replica.
        gpu_per_instance_small: GPUs per small-model replica.
        replicas_large: Number of large-model replicas.
        replicas_small: Number of small-model replicas.
    """

    url: str = "http://localhost:8002"
    large_model_id: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    small_model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    gpu_per_instance_large: int = 4
    gpu_per_instance_small: int = 1
    replicas_large: int = 2
    replicas_small: int = 2


class ExperimentConfig(BaseModel):
    """Experiment configuration loaded from YAML.

    Contains endpoint configuration and all timing/execution parameters.
    """

    large_model: ModelEndpoint
    small_model: ModelEndpoint
    planner: PlannerConfig | None = None
    poisson_qps: float = 0.1
    global_qps: float = 5.0
    agent_delay_ms: int = 100
    user_delay_ms: int = 5000
    timeout_s: float = 120.0
    max_tokens: int = 1


class ReplayStep(BaseModel):
    """One request within a replay group.

    Args:
        step_index: Position of this step within the group.
        model_size: Which endpoint to target ("large" or "small").
        sender_role: Determines inter-step delay ("agent"=100ms, "user"=5s).
        history_message: Message dict appended to cumulative history before sending.
    """

    step_index: int
    model_size: Literal["large", "small"]
    sender_role: Literal["agent", "user"]
    history_message: dict[str, Any]


class ReplayGroup(BaseModel):
    """A full conversation to replay (one session/task from the dataset).

    Args:
        group_id: Unique identifier for this conversation.
        dataset_name: Source dataset name.
        initial_messages: Base context messages (e.g. system prompt, first user prompt).
        steps: Ordered sequence of replay steps.
    """

    group_id: str
    dataset_name: str
    initial_messages: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[ReplayStep] = Field(default_factory=list)


class RequestMetrics(BaseModel):
    """Metrics for a single LLM request.

    Args:
        group_id: Which replay group this request belongs to.
        step_index: Position within the group.
        model_size: Which endpoint was used.
        start_time: Monotonic timestamp when request was sent.
        end_time: Monotonic timestamp when response was received.
        latency_ms: End-to-end latency in milliseconds.
        status: Outcome of the request.
        error_message: Error details if status is not "success".
        input_tokens: Prompt token count from API response (if available).
        output_tokens: Completion token count from API response (if available).
    """

    group_id: str
    step_index: int
    model_size: Literal["large", "small"]
    start_time: float
    end_time: float
    latency_ms: float
    status: Literal["success", "error", "timeout"]
    error_message: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class GroupMetrics(BaseModel):
    """Aggregated metrics for one replay group.

    Args:
        group_id: Unique identifier for this conversation.
        dataset_name: Source dataset name.
        total_steps: Total number of steps in the group.
        completed_steps: Number of successful steps.
        failed_steps: Number of failed/timed-out steps.
        total_latency_ms: Wall-clock time from first request send to last response.
        request_metrics: Per-step metrics.
    """

    group_id: str
    dataset_name: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    total_latency_ms: float
    request_metrics: list[RequestMetrics] = Field(default_factory=list)
