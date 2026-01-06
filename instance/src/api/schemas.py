"""Pydantic schemas for SwarmX Instance API.

This module defines all request/response schemas for the Instance API,
organized by API namespace:
- System API: Health and resource information
- File API: File management for inference workloads
- Models API: Model weight management and lifecycle
- Model Info API: Currently serving model information
- Chat Completions API: OpenAI-compatible chat interface
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Enums
# =============================================================================


class ModelType(str, Enum):
    """Supported model types.

    Currently only LLM is supported. Other types are planned for future.
    """

    LLM = "llm"
    # Future: DIFFUSION = "diffusion"
    # Future: EMBEDDING = "embedding"
    # Future: RERANKER = "reranker"
    # Future: TTS = "tts"
    # Future: ASR = "asr"


class ModelStatus(str, Enum):
    """Model lifecycle status."""

    PULLING = "pulling"
    UPLOADING = "uploading"
    EXTRACTING = "extracting"
    READY = "ready"
    LOADING = "loading"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class TaskOperation(str, Enum):
    """Type of background task operation."""

    PULL = "pull"
    UPLOAD = "upload"


class TaskStatus(str, Enum):
    """Status of a background task."""

    PENDING = "pending"
    PULLING = "pulling"
    UPLOADING = "uploading"
    EXTRACTING = "extracting"
    READY = "ready"
    ERROR = "error"


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response schema.

    Attributes:
        error: Error code identifier.
        message: Human-readable error message.
        details: Optional additional error details.
    """

    error: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )


# =============================================================================
# System API Schemas
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Health status string.
        timestamp: ISO-8601 timestamp of the check.
    """

    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="ISO-8601 timestamp")


class CPUInfo(BaseModel):
    """CPU resource information.

    Attributes:
        cores: Number of CPU cores.
        usage_percent: Current CPU usage percentage.
    """

    cores: int = Field(..., description="Number of CPU cores")
    usage_percent: float = Field(..., description="CPU usage percentage")


class MemoryInfo(BaseModel):
    """Memory resource information.

    Attributes:
        total_gb: Total memory in gigabytes.
        used_gb: Used memory in gigabytes.
        usage_percent: Memory usage percentage.
    """

    total_gb: float = Field(..., description="Total memory in GB")
    used_gb: float = Field(..., description="Used memory in GB")
    usage_percent: float = Field(..., description="Memory usage percentage")


class DiskInfo(BaseModel):
    """Disk resource information.

    Attributes:
        total_gb: Total disk space in gigabytes.
        used_gb: Used disk space in gigabytes.
        available_gb: Available disk space in gigabytes.
        min_free_gb: Minimum free space to maintain.
        usage_percent: Disk usage percentage.
    """

    total_gb: float = Field(..., description="Total disk space in GB")
    used_gb: float = Field(..., description="Used disk space in GB")
    available_gb: float = Field(..., description="Available disk space in GB")
    min_free_gb: float = Field(
        ..., description="Minimum free space to maintain"
    )
    usage_percent: float = Field(..., description="Disk usage percentage")


class GPUInfo(BaseModel):
    """GPU resource information.

    Attributes:
        index: GPU device index.
        name: GPU device name.
        memory_total_gb: Total GPU memory in gigabytes.
        memory_used_gb: Used GPU memory in gigabytes.
        utilization_percent: GPU utilization percentage.
        temperature_celsius: GPU temperature in Celsius.
    """

    index: int = Field(..., description="GPU device index")
    name: str = Field(..., description="GPU device name")
    memory_total_gb: float = Field(..., description="Total GPU memory in GB")
    memory_used_gb: float = Field(..., description="Used GPU memory in GB")
    utilization_percent: float = Field(
        ..., description="GPU utilization percentage"
    )
    temperature_celsius: int = Field(
        ..., description="GPU temperature in Celsius"
    )


class SystemResources(BaseModel):
    """System resource information container.

    Attributes:
        cpu: CPU resource information.
        memory: Memory resource information.
        disk: Disk resource information.
        gpu: List of GPU resource information.
    """

    cpu: CPUInfo = Field(..., description="CPU information")
    memory: MemoryInfo = Field(..., description="Memory information")
    disk: DiskInfo = Field(..., description="Disk information")
    gpu: list[GPUInfo] = Field(
        default_factory=list, description="GPU information"
    )


class InferenceServerInfo(BaseModel):
    """Inference server information.

    Attributes:
        type: Inference server type (e.g., 'vllm').
        version: Inference server version.
    """

    type: str = Field(..., description="Inference server type")
    version: str = Field(..., description="Inference server version")


class SystemInfo(BaseModel):
    """Full system information response.

    Attributes:
        instance_id: Unique instance identifier.
        uptime_seconds: Seconds since instance started.
        supported_model_types: List of supported model types.
        inference_server: Inference server information.
        resources: System resource information.
    """

    instance_id: str = Field(..., description="Unique instance identifier")
    uptime_seconds: int = Field(
        ..., description="Seconds since instance started"
    )
    supported_model_types: list[str] = Field(
        ..., description="Supported model types"
    )
    inference_server: InferenceServerInfo = Field(
        ..., description="Inference server info"
    )
    resources: SystemResources = Field(
        ..., description="System resource information"
    )


# =============================================================================
# File API Schemas
# =============================================================================


class FileUploadResponse(BaseModel):
    """Response after successful file upload.

    Attributes:
        file_id: Unique file identifier.
        filename: Original filename.
        purpose: Purpose of the file.
        size_bytes: File size in bytes.
        created_at: ISO-8601 creation timestamp.
    """

    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    purpose: str = Field(..., description="File purpose")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: str = Field(..., description="ISO-8601 creation timestamp")


class FileInfo(BaseModel):
    """File information.

    Attributes:
        file_id: Unique file identifier.
        filename: Original filename.
        purpose: Purpose of the file.
        size_bytes: File size in bytes.
        tags: Optional list of tags.
        created_at: ISO-8601 creation timestamp.
    """

    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    purpose: str = Field(..., description="File purpose")
    size_bytes: int = Field(..., description="File size in bytes")
    tags: list[str] = Field(default_factory=list, description="File tags")
    created_at: str = Field(..., description="ISO-8601 creation timestamp")


class FileListResponse(BaseModel):
    """Response for file listing.

    Attributes:
        files: List of file information.
        total: Total number of files.
    """

    files: list[FileInfo] = Field(..., description="List of files")
    total: int = Field(..., description="Total file count")


class FileDeleteResponse(BaseModel):
    """Response after file deletion.

    Attributes:
        deleted: Whether deletion was successful.
        file_id: ID of the deleted file.
    """

    deleted: bool = Field(..., description="Deletion success")
    file_id: str = Field(..., description="Deleted file ID")


# =============================================================================
# Models API Schemas - OpenAI Compatible
# =============================================================================


class OpenAIModel(BaseModel):
    """OpenAI-compatible model object.

    Attributes:
        id: Model identifier (name).
        object: Always 'model'.
        created: Unix timestamp of creation.
        owned_by: Model owner.
    """

    id: str = Field(..., description="Model identifier")
    object: Literal["model"] = Field(default="model", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    owned_by: str = Field(..., description="Model owner")


class OpenAIModelListResponse(BaseModel):
    """OpenAI-compatible model list response.

    Attributes:
        object: Always 'list'.
        data: List of model objects.
    """

    object: Literal["list"] = Field(default="list", description="Object type")
    data: list[OpenAIModel] = Field(..., description="List of models")


# =============================================================================
# Models API Schemas - SwarmX Extensions
# =============================================================================


class ModelSource(BaseModel):
    """Model source information.

    Attributes:
        type: Source type ('huggingface' or 'upload').
        repo: HuggingFace repository (for HF sources).
        revision: Git revision (for HF sources).
        endpoint: HuggingFace endpoint URL.
        filename: Upload filename (for upload sources).
    """

    type: str = Field(..., description="Source type")
    repo: str | None = Field(default=None, description="HuggingFace repo")
    revision: str | None = Field(default=None, description="Git revision")
    endpoint: str | None = Field(default=None, description="HF endpoint URL")
    filename: str | None = Field(default=None, description="Upload filename")


class ModelPullSource(BaseModel):
    """Source specification for model pull request.

    Attributes:
        repo: HuggingFace repository identifier.
        revision: Git revision (default: 'main').
        endpoint: HuggingFace endpoint URL.
        token: HuggingFace access token.
    """

    repo: str = Field(..., description="HuggingFace repository")
    revision: str = Field(default="main", description="Git revision")
    endpoint: str = Field(
        default="https://huggingface.co", description="HF endpoint"
    )
    token: str | None = Field(default=None, description="HF access token")


class ModelPullRequest(BaseModel):
    """Request to pull model from HuggingFace.

    Attributes:
        name: Local model name.
        type: Model type (must be 'llm' currently).
        source: Source specification.
    """

    name: str = Field(..., description="Local model name")
    type: str = Field(..., description="Model type")
    source: ModelPullSource = Field(..., description="Source specification")

    @field_validator("type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate that model type is supported."""
        supported = [t.value for t in ModelType]
        if v not in supported:
            raise ValueError(
                f"Model type '{v}' is not supported. "
                f"Supported types: {supported}"
            )
        return v


class ModelPullResponse(BaseModel):
    """Response after initiating model pull.

    Attributes:
        model_id: Assigned model identifier.
        task_id: Background task identifier.
        name: Model name.
        type: Model type.
        status: Current status.
        message: Status message.
    """

    model_id: str = Field(..., description="Model identifier")
    task_id: str = Field(..., description="Task identifier")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    status: str = Field(..., description="Current status")
    message: str = Field(..., description="Status message")


class ModelUploadResponse(BaseModel):
    """Response after model upload.

    Attributes:
        model_id: Assigned model identifier.
        name: Model name.
        type: Model type.
        status: Current status (ready for sync uploads).
        message: Status message.
    """

    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    status: str = Field(..., description="Current status")
    message: str = Field(..., description="Status message")


class TaskProgressResponse(BaseModel):
    """Response for task progress query.

    Attributes:
        task_id: Task identifier.
        model_id: Associated model identifier.
        type: Model type.
        operation: Task operation type.
        status: Current task status.
        progress_percent: Progress percentage (0-100).
        current_step: Human-readable current step.
        bytes_completed: Bytes processed so far.
        bytes_total: Total bytes to process.
        error: Error message if failed.
    """

    task_id: str = Field(..., description="Task identifier")
    model_id: str = Field(..., description="Model identifier")
    type: str = Field(..., description="Model type")
    operation: str = Field(..., description="Operation type")
    status: str = Field(..., description="Task status")
    progress_percent: int = Field(..., description="Progress percentage")
    current_step: str = Field(..., description="Current step description")
    bytes_completed: int = Field(..., description="Bytes completed")
    bytes_total: int = Field(..., description="Total bytes")
    error: str | None = Field(default=None, description="Error message")


class ModelRuntime(BaseModel):
    """Runtime information for a running model.

    Attributes:
        loaded_at: When model was loaded.
        config: Current runtime configuration.
        gpu_ids: GPUs being used.
        gpu_memory_used_gb: GPU memory consumption.
        requests_served: Total requests served.
        avg_latency_ms: Average latency in milliseconds.
    """

    loaded_at: str = Field(..., description="Load timestamp")
    config: dict[str, Any] = Field(..., description="Runtime config")
    gpu_ids: list[int] = Field(..., description="GPU IDs in use")
    gpu_memory_used_gb: float = Field(..., description="GPU memory used")
    requests_served: int = Field(..., description="Requests served")
    avg_latency_ms: float = Field(..., description="Average latency ms")


class ModelDetailResponse(BaseModel):
    """Detailed model information response.

    Attributes:
        model_id: Model identifier.
        name: Model name.
        type: Model type.
        status: Current status.
        source: Model source information.
        size_bytes: Model size in bytes.
        files: List of model files.
        created_at: Creation timestamp.
        default_config: Default runtime configuration.
        runtime: Runtime info (only when running).
    """

    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    status: str = Field(..., description="Current status")
    source: ModelSource = Field(..., description="Model source")
    size_bytes: int = Field(..., description="Model size in bytes")
    files: list[str] = Field(..., description="Model files")
    created_at: str = Field(..., description="Creation timestamp")
    default_config: dict[str, Any] | None = Field(
        default=None, description="Default config"
    )
    runtime: ModelRuntime | None = Field(
        default=None, description="Runtime info"
    )


class ModelListItem(BaseModel):
    """Model item in list response.

    Attributes:
        model_id: Model identifier.
        name: Model name.
        type: Model type.
        status: Current status.
        source: Model source.
        size_bytes: Model size.
        created_at: Creation timestamp.
        has_default_config: Whether default config exists.
    """

    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    status: str = Field(..., description="Current status")
    source: ModelSource = Field(..., description="Model source")
    size_bytes: int = Field(..., description="Model size")
    created_at: str = Field(..., description="Creation timestamp")
    has_default_config: bool = Field(..., description="Has default config")


class ModelListResponse(BaseModel):
    """SwarmX extended model list response.

    Attributes:
        models: List of model items.
        active_model: Currently active model ID.
        total_size_bytes: Total size of all models.
    """

    models: list[ModelListItem] = Field(..., description="Model list")
    active_model: str | None = Field(
        default=None, description="Active model ID"
    )
    total_size_bytes: int = Field(..., description="Total size")


class ModelConfigRequest(BaseModel):
    """Model runtime configuration request.

    Attributes:
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_model_len: Maximum sequence length.
        quantization: Quantization method.
        gpu_memory_utilization: GPU memory fraction (0.0-1.0).
        dtype: Data type.
        enforce_eager: Disable CUDA graphs.
    """

    tensor_parallel_size: int | None = Field(
        default=None, description="Tensor parallel size"
    )
    max_model_len: int | None = Field(
        default=None, description="Max sequence length"
    )
    quantization: str | None = Field(
        default=None, description="Quantization method"
    )
    gpu_memory_utilization: float | None = Field(
        default=None, description="GPU memory utilization"
    )
    dtype: str | None = Field(default=None, description="Data type")
    enforce_eager: bool | None = Field(
        default=None, description="Disable CUDA graphs"
    )


class ModelStartRequest(BaseModel):
    """Request to start a model.

    Attributes:
        gpu_ids: GPU device IDs to use.
        config: Runtime configuration (optional, uses default if not provided).
    """

    gpu_ids: list[int] = Field(..., description="GPU IDs to use")
    config: ModelConfigRequest | None = Field(
        default=None, description="Runtime config"
    )


class ModelStartResponse(BaseModel):
    """Response after starting a model.

    Attributes:
        model_id: Model identifier.
        type: Model type.
        status: Current status.
        message: Status message.
    """

    model_id: str = Field(..., description="Model identifier")
    type: str = Field(..., description="Model type")
    status: str = Field(..., description="Current status")
    message: str = Field(..., description="Status message")


class ModelStopRequest(BaseModel):
    """Request to stop a model.

    Attributes:
        force: Force immediate stop without waiting.
    """

    force: bool = Field(default=False, description="Force stop")


class ModelStopResponse(BaseModel):
    """Response after stopping a model.

    Attributes:
        model_id: Model identifier.
        status: Current status.
        message: Status message.
    """

    model_id: str = Field(..., description="Model identifier")
    status: str = Field(..., description="Current status")
    message: str = Field(..., description="Status message")


class ModelSwitchRequest(BaseModel):
    """Request to switch models.

    Attributes:
        target_model_id: Model to switch to.
        gpu_ids: GPU IDs to use.
        graceful_timeout_seconds: Wait time for in-flight requests.
        config: Runtime config (optional, uses target's default).
    """

    target_model_id: str = Field(..., description="Target model ID")
    gpu_ids: list[int] = Field(..., description="GPU IDs to use")
    graceful_timeout_seconds: int = Field(
        default=30, description="Graceful timeout"
    )
    config: ModelConfigRequest | None = Field(
        default=None, description="Runtime config"
    )


class ModelSwitchResponse(BaseModel):
    """Response after initiating model switch.

    Attributes:
        previous_model_id: Previous model ID.
        current_model_id: New model ID.
        status: Switch status.
        message: Status message.
    """

    previous_model_id: str | None = Field(
        default=None, description="Previous model"
    )
    current_model_id: str = Field(..., description="Current model")
    status: str = Field(..., description="Status")
    message: str = Field(..., description="Message")


class ModelConfigResponse(BaseModel):
    """Response after setting model config.

    Attributes:
        model_id: Model identifier.
        default_config: The saved configuration.
        message: Status message.
    """

    model_id: str = Field(..., description="Model identifier")
    default_config: dict[str, Any] = Field(
        ..., description="Saved configuration"
    )
    message: str = Field(..., description="Status message")


class ModelDeleteResponse(BaseModel):
    """Response after deleting a model.

    Attributes:
        deleted: Whether deletion succeeded.
        model_id: Deleted model ID.
        disk_freed_bytes: Disk space freed.
    """

    deleted: bool = Field(..., description="Deletion success")
    model_id: str = Field(..., description="Deleted model ID")
    disk_freed_bytes: int = Field(..., description="Disk space freed")


# =============================================================================
# Model Info API Schemas
# =============================================================================


class ModelBasicInfo(BaseModel):
    """Basic model identification info.

    Attributes:
        model_id: Model identifier.
        name: Model name.
        type: Model type.
        source: Optional source information.
    """

    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    source: ModelSource | None = Field(default=None, description="Source info")


class GPUDeviceInfo(BaseModel):
    """GPU device information for model info.

    Attributes:
        index: Device index.
        name: Device name.
        memory_total_gb: Total memory.
        memory_used_gb: Used memory.
        utilization_percent: Utilization.
        temperature_celsius: Temperature.
    """

    index: int = Field(..., description="Device index")
    name: str = Field(..., description="Device name")
    memory_total_gb: float = Field(..., description="Total memory GB")
    memory_used_gb: float = Field(..., description="Used memory GB")
    utilization_percent: float = Field(..., description="Utilization %")
    temperature_celsius: int = Field(..., description="Temperature C")


class ModelInfoResourcesGPU(BaseModel):
    """GPU resources for model info.

    Attributes:
        gpu_ids: GPU device IDs.
        gpu_count: Number of GPUs.
        total_memory_gb: Total GPU memory.
        used_memory_gb: Used GPU memory.
        memory_utilization_percent: Memory utilization.
        devices: Per-device information.
    """

    gpu_ids: list[int] = Field(..., description="GPU IDs")
    gpu_count: int = Field(..., description="GPU count")
    total_memory_gb: float = Field(..., description="Total memory GB")
    used_memory_gb: float = Field(..., description="Used memory GB")
    memory_utilization_percent: float = Field(
        ..., description="Memory utilization %"
    )
    devices: list[GPUDeviceInfo] = Field(..., description="Device info")


class ModelInfoResourcesMemory(BaseModel):
    """Memory resources for model info.

    Attributes:
        model_memory_gb: Memory used by model.
        kv_cache_memory_gb: Memory used by KV cache.
    """

    model_memory_gb: float = Field(..., description="Model memory GB")
    kv_cache_memory_gb: float = Field(..., description="KV cache memory GB")


class ModelInfoResources(BaseModel):
    """Resource information for model info.

    Attributes:
        gpu: GPU resource information.
        memory: Memory resource information.
    """

    gpu: ModelInfoResourcesGPU = Field(..., description="GPU resources")
    memory: ModelInfoResourcesMemory = Field(
        ..., description="Memory resources"
    )


class ModelRuntimeConfig(BaseModel):
    """Runtime configuration for model info.

    Attributes:
        tensor_parallel_size: Tensor parallel size.
        max_model_len: Max model length.
        quantization: Quantization method.
        gpu_memory_utilization: GPU memory utilization.
        dtype: Data type.
        enforce_eager: Eager mode flag.
    """

    tensor_parallel_size: int | None = Field(
        default=None, description="Tensor parallel"
    )
    max_model_len: int | None = Field(default=None, description="Max length")
    quantization: str | None = Field(default=None, description="Quantization")
    gpu_memory_utilization: float | None = Field(
        default=None, description="GPU memory util"
    )
    dtype: str | None = Field(default=None, description="Data type")
    enforce_eager: bool | None = Field(default=None, description="Eager mode")


class ModelParameters(BaseModel):
    """Model parameters for model info.

    Attributes:
        runtime_config: Runtime configuration.
        architecture_config: Model architecture config (serialized as model_config).
        tokenizer_config: Tokenizer configuration.
    """

    runtime_config: ModelRuntimeConfig = Field(
        ..., description="Runtime config"
    )
    architecture_config: dict[str, Any] | None = Field(
        default=None,
        description="Model config",
        serialization_alias="model_config",
        validation_alias="model_config",
    )
    tokenizer_config: dict[str, Any] | None = Field(
        default=None, description="Tokenizer config"
    )


class ModelInfoStats(BaseModel):
    """Runtime statistics for model info.

    Attributes:
        loaded_at: When model was loaded.
        uptime_seconds: Model uptime.
        requests_served: Total requests.
        tokens_generated: Total tokens generated.
        avg_latency_ms: Average latency.
        avg_tokens_per_second: Token throughput.
        current_batch_size: Current batch size.
        pending_requests: Pending request count.
    """

    loaded_at: str = Field(..., description="Load timestamp")
    uptime_seconds: int = Field(..., description="Uptime seconds")
    requests_served: int = Field(..., description="Requests served")
    tokens_generated: int = Field(..., description="Tokens generated")
    avg_latency_ms: float = Field(..., description="Avg latency ms")
    avg_tokens_per_second: float = Field(..., description="Tokens/sec")
    current_batch_size: int = Field(..., description="Current batch")
    pending_requests: int = Field(..., description="Pending requests")


class ModelInfoResponse(BaseModel):
    """Response for model info endpoint.

    Attributes:
        serving: Whether a model is serving.
        model: Basic model info (when serving).
        resources: Resource info (when serving).
        parameters: Model parameters (when serving).
        stats: Runtime stats (when serving).
        message: Message (when not serving).
    """

    serving: bool = Field(..., description="Is serving")
    model: ModelBasicInfo | None = Field(default=None, description="Model info")
    resources: ModelInfoResources | None = Field(
        default=None, description="Resources"
    )
    parameters: ModelParameters | None = Field(
        default=None, description="Parameters"
    )
    stats: ModelInfoStats | None = Field(default=None, description="Stats")
    message: str | None = Field(default=None, description="Message")


# =============================================================================
# Chat Completions API Schemas
# =============================================================================


class ChatMessage(BaseModel):
    """Chat message.

    Attributes:
        role: Message role (system, user, assistant).
        content: Message content.
    """

    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    Attributes:
        model: Model name to use.
        messages: List of chat messages.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Top-p sampling parameter.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        stop: Stop sequences.
        stream: Whether to stream response.
        file_refs: SwarmX extension - file references.
    """

    model: str = Field(..., description="Model name")
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    temperature: float | None = Field(default=None, description="Temperature")
    max_tokens: int | None = Field(default=None, description="Max tokens")
    top_p: float | None = Field(default=None, description="Top-p")
    frequency_penalty: float | None = Field(
        default=None, description="Frequency penalty"
    )
    presence_penalty: float | None = Field(
        default=None, description="Presence penalty"
    )
    stop: list[str] | None = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Stream response")
    # SwarmX extension
    file_refs: list[str] | None = Field(
        default=None, description="File references"
    )


class ChatCompletionChoice(BaseModel):
    """Chat completion choice.

    Attributes:
        index: Choice index.
        message: Response message.
        finish_reason: Reason for finishing.
    """

    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Response message")
    finish_reason: str = Field(..., description="Finish reason")


class ChatCompletionUsage(BaseModel):
    """Token usage information.

    Attributes:
        prompt_tokens: Input token count.
        completion_tokens: Output token count.
        total_tokens: Total token count.
    """

    prompt_tokens: int = Field(..., description="Prompt tokens")
    completion_tokens: int = Field(..., description="Completion tokens")
    total_tokens: int = Field(..., description="Total tokens")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response.

    Attributes:
        id: Response identifier.
        object: Always 'chat.completion'.
        created: Unix timestamp.
        model: Model used.
        choices: Response choices.
        usage: Token usage.
    """

    id: str = Field(..., description="Response ID")
    object: Literal["chat.completion"] = Field(
        default="chat.completion", description="Object type"
    )
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: list[ChatCompletionChoice] = Field(..., description="Choices")
    usage: ChatCompletionUsage = Field(..., description="Token usage")


class ChatCompletionDelta(BaseModel):
    """Delta content for streaming.

    Attributes:
        role: Message role (first chunk only).
        content: Content delta.
    """

    role: str | None = Field(default=None, description="Role")
    content: str | None = Field(default=None, description="Content delta")


class ChatCompletionChunkChoice(BaseModel):
    """Streaming chunk choice.

    Attributes:
        index: Choice index.
        delta: Content delta.
        finish_reason: Finish reason (last chunk only).
    """

    index: int = Field(..., description="Choice index")
    delta: ChatCompletionDelta = Field(..., description="Delta")
    finish_reason: str | None = Field(default=None, description="Finish reason")


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk.

    Attributes:
        id: Response identifier.
        object: Always 'chat.completion.chunk'.
        created: Unix timestamp.
        model: Model used.
        choices: Chunk choices.
    """

    id: str = Field(..., description="Response ID")
    object: Literal["chat.completion.chunk"] = Field(
        default="chat.completion.chunk", description="Object type"
    )
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: list[ChatCompletionChunkChoice] = Field(
        ..., description="Chunk choices"
    )
