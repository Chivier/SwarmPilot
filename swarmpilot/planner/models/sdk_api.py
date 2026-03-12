"""SDK API request/response models for deployment management.

Pydantic models for the SDK-facing REST endpoints that provide
a simplified interface for model deployment, scaling, and management.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ServeRequest(BaseModel):
    """Request to deploy a model service.

    Attributes:
        model_or_command: Model name (e.g. 'Qwen/Qwen3-0.6B')
            or a full shell command.
        name: Deployment name. Defaults to sanitized model name.
        replicas: Number of replicas to deploy.
        gpu_count: GPUs per replica.
        backend: Model backend ('vllm' or 'sglang').
        scheduler: Scheduler resolution: 'auto' looks up
            SchedulerRegistry, None skips, or a URL string.
    """

    model_or_command: str = Field(
        ..., description="Model name or shell command"
    )
    name: str | None = Field(
        None, description="Deployment name (defaults to sanitized model)"
    )
    replicas: int = Field(1, ge=1, description="Number of replicas")
    gpu_count: int = Field(1, ge=0, description="GPUs per replica")
    backend: str = Field("vllm", description="Model backend")
    scheduler: str | None = Field(
        "auto", description="Scheduler: 'auto', None, or URL"
    )


class ServeResponse(BaseModel):
    """Response from a serve deployment.

    Attributes:
        success: Whether deployment succeeded.
        name: Deployment name.
        model: Model identifier.
        replicas: Number of replicas requested.
        instances: List of deployed instance IDs.
        scheduler_url: Resolved scheduler URL, if any.
        error: Error message if failed.
    """

    success: bool
    name: str
    model: str
    replicas: int
    instances: list[str] = Field(default_factory=list)
    scheduler_url: str | None = None
    error: str | None = None


class RunRequest(BaseModel):
    """Request to run a custom workload.

    Attributes:
        command: Shell command to execute.
        name: Deployment name.
        replicas: Number of replicas.
        gpu_count: GPUs per replica.
    """

    command: str = Field(..., description="Shell command to execute")
    name: str | None = Field(None, description="Deployment name")
    replicas: int = Field(1, ge=1, description="Number of replicas")
    gpu_count: int = Field(1, ge=0, description="GPUs per replica")


class RunResponse(BaseModel):
    """Response from a run deployment.

    Attributes:
        success: Whether deployment succeeded.
        name: Deployment name.
        command: Command that was executed.
        replicas: Number of replicas requested.
        instances: List of deployed instance IDs.
        error: Error message if failed.
    """

    success: bool
    name: str
    command: str
    replicas: int
    instances: list[str] = Field(default_factory=list)
    error: str | None = None


class RegisterRequest(BaseModel):
    """Request to register model requirements.

    Attributes:
        model: Model identifier.
        replicas: Desired replica count.
        gpu_count: GPUs per replica.
        backend: Model backend.
        priority: Scheduling priority (higher = more important).
    """

    model: str = Field(..., description="Model identifier")
    replicas: int = Field(1, ge=1, description="Desired replicas")
    gpu_count: int = Field(1, ge=0, description="GPUs per replica")
    backend: str = Field("vllm", description="Model backend")
    priority: float = Field(
        1.0, ge=0, description="Scheduling priority"
    )


class DeployResponse(BaseModel):
    """Response from optimized deployment trigger.

    Attributes:
        success: Whether deployment succeeded.
        deployed_models: List of models that were deployed.
        total_instances: Total number of instances deployed.
        error: Error message if failed.
    """

    success: bool
    deployed_models: list[str] = Field(default_factory=list)
    total_instances: int = 0
    error: str | None = None


class RegisteredModelsResponse(BaseModel):
    """Response listing registered models.

    Attributes:
        models: Dict mapping model name to RegisterRequest.
        total: Total number of registered models.
    """

    models: dict[str, RegisterRequest]
    total: int


class InstanceDetailResponse(BaseModel):
    """Detailed information about a managed instance.

    Attributes:
        pylet_id: PyLet instance UUID.
        instance_id: Scheduler instance ID.
        model_id: Model identifier.
        endpoint: HTTP endpoint when running.
        status: Current lifecycle status.
        gpu_count: Number of GPUs allocated.
        error: Error message if failed.
    """

    pylet_id: str
    instance_id: str
    model_id: str
    endpoint: str | None = None
    status: str
    gpu_count: int = 1
    error: str | None = None


class ScaleRequest(BaseModel):
    """Request to scale model replicas.

    Attributes:
        model: Model identifier.
        replicas: Target replica count.
    """

    model: str = Field(..., description="Model identifier")
    replicas: int = Field(..., ge=0, description="Target replicas")


class ScaleResponse(BaseModel):
    """Response from a scale operation.

    Attributes:
        success: Whether scaling succeeded.
        model: Model identifier.
        previous_count: Previous replica count.
        current_count: Current replica count.
        error: Error message if failed.
    """

    success: bool
    model: str
    previous_count: int
    current_count: int
    error: str | None = None


class TerminateRequest(BaseModel):
    """Request to terminate instances.

    Attributes:
        name: Terminate instances by deployment name.
        model: Terminate instances by model identifier.
        all: Terminate all instances.
    """

    name: str | None = Field(
        None, description="Terminate by deployment name"
    )
    model: str | None = Field(
        None, description="Terminate by model identifier"
    )
    all: bool = Field(False, description="Terminate all instances")


class TerminateResponse(BaseModel):
    """Response from a terminate operation.

    Attributes:
        success: Whether termination succeeded.
        terminated_count: Number of instances terminated.
        message: Status message.
        error: Error message if failed.
    """

    success: bool
    terminated_count: int = 0
    message: str = ""
    error: str | None = None


class SchedulerMapResponse(BaseModel):
    """Response with scheduler-to-model mapping.

    Attributes:
        schedulers: Dict mapping model_id to scheduler_url.
        total: Total number of registered schedulers.
    """

    schedulers: dict[str, str]
    total: int
