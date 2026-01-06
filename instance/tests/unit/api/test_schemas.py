"""Unit tests for API Pydantic schemas.

Tests follow TDD principle - written before implementation.
Each schema is tested for:
1. Valid instantiation with required fields
2. Default values
3. Serialization to dict/JSON
4. Validation errors for invalid data
"""

import pytest
from datetime import datetime
from pydantic import ValidationError


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_valid_health_response(self):
        """Test creating a valid health response."""
        from src.api.schemas import HealthResponse

        response = HealthResponse(
            status="healthy",
            timestamp="2026-01-05T12:00:00Z",
        )
        assert response.status == "healthy"
        assert response.timestamp == "2026-01-05T12:00:00Z"

    def test_health_response_serialization(self):
        """Test serialization to dict."""
        from src.api.schemas import HealthResponse

        response = HealthResponse(
            status="healthy",
            timestamp="2026-01-05T12:00:00Z",
        )
        data = response.model_dump()
        assert data["status"] == "healthy"
        assert data["timestamp"] == "2026-01-05T12:00:00Z"


class TestSystemInfo:
    """Tests for SystemInfo and related schemas."""

    def test_cpu_info(self):
        """Test CPUInfo schema."""
        from src.api.schemas import CPUInfo

        cpu = CPUInfo(cores=8, usage_percent=45.2)
        assert cpu.cores == 8
        assert cpu.usage_percent == 45.2

    def test_memory_info(self):
        """Test MemoryInfo schema."""
        from src.api.schemas import MemoryInfo

        mem = MemoryInfo(
            total_gb=64.0,
            used_gb=32.5,
            usage_percent=50.8,
        )
        assert mem.total_gb == 64.0
        assert mem.used_gb == 32.5

    def test_disk_info(self):
        """Test DiskInfo schema."""
        from src.api.schemas import DiskInfo

        disk = DiskInfo(
            total_gb=500.0,
            used_gb=120.0,
            available_gb=380.0,
            min_free_gb=10.0,
            usage_percent=24.0,
        )
        assert disk.available_gb == 380.0
        assert disk.min_free_gb == 10.0

    def test_gpu_info(self):
        """Test GPUInfo schema."""
        from src.api.schemas import GPUInfo

        gpu = GPUInfo(
            index=0,
            name="NVIDIA A100",
            memory_total_gb=80.0,
            memory_used_gb=45.0,
            utilization_percent=78.5,
            temperature_celsius=65,
        )
        assert gpu.name == "NVIDIA A100"
        assert gpu.index == 0

    def test_system_resources(self):
        """Test SystemResources schema."""
        from src.api.schemas import (
            CPUInfo,
            DiskInfo,
            GPUInfo,
            MemoryInfo,
            SystemResources,
        )

        resources = SystemResources(
            cpu=CPUInfo(cores=8, usage_percent=45.2),
            memory=MemoryInfo(total_gb=64.0, used_gb=32.5, usage_percent=50.8),
            disk=DiskInfo(
                total_gb=500.0,
                used_gb=120.0,
                available_gb=380.0,
                min_free_gb=10.0,
                usage_percent=24.0,
            ),
            gpu=[
                GPUInfo(
                    index=0,
                    name="NVIDIA A100",
                    memory_total_gb=80.0,
                    memory_used_gb=45.0,
                    utilization_percent=78.5,
                    temperature_celsius=65,
                )
            ],
        )
        assert len(resources.gpu) == 1
        assert resources.cpu.cores == 8

    def test_system_info_full(self):
        """Test full SystemInfo schema."""
        from src.api.schemas import (
            CPUInfo,
            DiskInfo,
            GPUInfo,
            InferenceServerInfo,
            MemoryInfo,
            SystemInfo,
            SystemResources,
        )

        info = SystemInfo(
            instance_id="inst_xyz789",
            uptime_seconds=3600,
            supported_model_types=["llm"],
            inference_server=InferenceServerInfo(type="vllm", version="0.6.0"),
            resources=SystemResources(
                cpu=CPUInfo(cores=8, usage_percent=45.2),
                memory=MemoryInfo(
                    total_gb=64.0, used_gb=32.5, usage_percent=50.8
                ),
                disk=DiskInfo(
                    total_gb=500.0,
                    used_gb=120.0,
                    available_gb=380.0,
                    min_free_gb=10.0,
                    usage_percent=24.0,
                ),
                gpu=[],
            ),
        )
        assert info.instance_id == "inst_xyz789"
        assert info.uptime_seconds == 3600
        assert "llm" in info.supported_model_types


class TestFileAPISchemas:
    """Tests for File API schemas."""

    def test_file_upload_response(self):
        """Test FileUploadResponse schema."""
        from src.api.schemas import FileUploadResponse

        response = FileUploadResponse(
            file_id="file_abc123",
            filename="context.jsonl",
            purpose="inference",
            size_bytes=1048576,
            created_at="2026-01-05T12:00:00Z",
        )
        assert response.file_id == "file_abc123"
        assert response.size_bytes == 1048576

    def test_file_info(self):
        """Test FileInfo schema with optional tags."""
        from src.api.schemas import FileInfo

        info = FileInfo(
            file_id="file_abc123",
            filename="context.jsonl",
            purpose="inference",
            size_bytes=1048576,
            tags=["batch", "v1"],
            created_at="2026-01-05T12:00:00Z",
        )
        assert info.tags == ["batch", "v1"]

    def test_file_info_default_tags(self):
        """Test FileInfo with default empty tags."""
        from src.api.schemas import FileInfo

        info = FileInfo(
            file_id="file_abc123",
            filename="context.jsonl",
            purpose="inference",
            size_bytes=1048576,
            created_at="2026-01-05T12:00:00Z",
        )
        assert info.tags == []

    def test_file_list_response(self):
        """Test FileListResponse schema."""
        from src.api.schemas import FileInfo, FileListResponse

        response = FileListResponse(
            files=[
                FileInfo(
                    file_id="file_abc123",
                    filename="context.jsonl",
                    purpose="inference",
                    size_bytes=1048576,
                    created_at="2026-01-05T12:00:00Z",
                )
            ],
            total=1,
        )
        assert response.total == 1
        assert len(response.files) == 1

    def test_file_delete_response(self):
        """Test FileDeleteResponse schema."""
        from src.api.schemas import FileDeleteResponse

        response = FileDeleteResponse(
            deleted=True,
            file_id="file_abc123",
        )
        assert response.deleted is True


class TestModelsAPISchemas:
    """Tests for Models API schemas."""

    def test_openai_model_object(self):
        """Test OpenAI-compatible Model object."""
        from src.api.schemas import OpenAIModel

        model = OpenAIModel(
            id="llama-3.1-70b",
            created=1704456000,
            owned_by="swarmx",
        )
        assert model.id == "llama-3.1-70b"
        assert model.object == "model"

    def test_openai_model_list_response(self):
        """Test OpenAI-compatible model list response."""
        from src.api.schemas import OpenAIModel, OpenAIModelListResponse

        response = OpenAIModelListResponse(
            data=[
                OpenAIModel(
                    id="llama-3.1-70b",
                    created=1704456000,
                    owned_by="swarmx",
                )
            ]
        )
        assert response.object == "list"
        assert len(response.data) == 1

    def test_model_source_huggingface(self):
        """Test ModelSource for HuggingFace."""
        from src.api.schemas import ModelSource

        source = ModelSource(
            type="huggingface",
            repo="meta-llama/Llama-3.1-70B-Instruct",
            revision="main",
            endpoint="https://huggingface.co",
        )
        assert source.type == "huggingface"
        assert source.repo == "meta-llama/Llama-3.1-70B-Instruct"

    def test_model_source_upload(self):
        """Test ModelSource for upload."""
        from src.api.schemas import ModelSource

        source = ModelSource(
            type="upload",
            filename="model-weights.tar.gz",
        )
        assert source.type == "upload"
        assert source.filename == "model-weights.tar.gz"

    def test_model_pull_request(self):
        """Test ModelPullRequest schema."""
        from src.api.schemas import ModelPullRequest, ModelPullSource

        request = ModelPullRequest(
            name="llama-3.1-70b",
            type="llm",
            source=ModelPullSource(
                repo="meta-llama/Llama-3.1-70B-Instruct",
                revision="main",
            ),
        )
        assert request.name == "llama-3.1-70b"
        assert request.type == "llm"

    def test_model_pull_request_validation_unsupported_type(self):
        """Test ModelPullRequest rejects unsupported model types."""
        from src.api.schemas import ModelPullRequest, ModelPullSource

        # For now, only 'llm' is supported
        with pytest.raises(ValidationError):
            ModelPullRequest(
                name="diffusion-model",
                type="diffusion",
                source=ModelPullSource(repo="some/repo"),
            )

    def test_model_pull_response(self):
        """Test ModelPullResponse schema."""
        from src.api.schemas import ModelPullResponse

        response = ModelPullResponse(
            model_id="model_abc123",
            task_id="task_xyz",
            name="llama-3.1-70b",
            type="llm",
            status="pulling",
            message="Pulling weights from HuggingFace",
        )
        assert response.status == "pulling"

    def test_task_progress_response(self):
        """Test TaskProgressResponse schema."""
        from src.api.schemas import TaskProgressResponse

        response = TaskProgressResponse(
            task_id="task_xyz",
            model_id="model_abc123",
            type="llm",
            operation="pull",
            status="pulling",
            progress_percent=45,
            current_step="Downloading (23GB / 50GB)",
            bytes_completed=23000000000,
            bytes_total=50000000000,
        )
        assert response.progress_percent == 45
        assert response.error is None

    def test_model_detail_response(self):
        """Test ModelDetailResponse schema."""
        from src.api.schemas import ModelDetailResponse, ModelSource

        response = ModelDetailResponse(
            model_id="model_abc123",
            name="llama-3.1-70b",
            type="llm",
            status="ready",
            source=ModelSource(
                type="huggingface",
                repo="meta-llama/Llama-3.1-70B-Instruct",
            ),
            size_bytes=140000000000,
            files=["config.json", "tokenizer.json"],
            created_at="2026-01-05T12:00:00Z",
        )
        assert response.status == "ready"
        assert response.runtime is None  # Not running

    def test_model_list_response(self):
        """Test SwarmX extended model list response."""
        from src.api.schemas import (
            ModelListItem,
            ModelListResponse,
            ModelSource,
        )

        response = ModelListResponse(
            models=[
                ModelListItem(
                    model_id="model_abc123",
                    name="llama-3.1-70b",
                    type="llm",
                    status="running",
                    source=ModelSource(type="huggingface", repo="meta/llama"),
                    size_bytes=140000000000,
                    created_at="2026-01-05T12:00:00Z",
                    has_default_config=True,
                )
            ],
            active_model="model_abc123",
            total_size_bytes=140000000000,
        )
        assert response.active_model == "model_abc123"

    def test_model_config_request(self):
        """Test ModelConfigRequest schema."""
        from src.api.schemas import ModelConfigRequest

        config = ModelConfigRequest(
            tensor_parallel_size=2,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
        )
        assert config.tensor_parallel_size == 2
        assert config.quantization is None

    def test_model_start_request(self):
        """Test ModelStartRequest schema."""
        from src.api.schemas import ModelConfigRequest, ModelStartRequest

        request = ModelStartRequest(
            gpu_ids=[0, 1],
            config=ModelConfigRequest(
                tensor_parallel_size=2,
                max_model_len=8192,
            ),
        )
        assert request.gpu_ids == [0, 1]

    def test_model_start_request_no_config(self):
        """Test ModelStartRequest with no config (uses default)."""
        from src.api.schemas import ModelStartRequest

        request = ModelStartRequest(gpu_ids=[0, 1])
        assert request.config is None

    def test_model_stop_request(self):
        """Test ModelStopRequest schema."""
        from src.api.schemas import ModelStopRequest

        request = ModelStopRequest(force=False)
        assert request.force is False

    def test_model_switch_request(self):
        """Test ModelSwitchRequest schema."""
        from src.api.schemas import ModelSwitchRequest

        request = ModelSwitchRequest(
            target_model_id="model_def456",
            gpu_ids=[0, 1],
            graceful_timeout_seconds=30,
        )
        assert request.target_model_id == "model_def456"
        assert request.graceful_timeout_seconds == 30

    def test_model_delete_response(self):
        """Test ModelDeleteResponse schema."""
        from src.api.schemas import ModelDeleteResponse

        response = ModelDeleteResponse(
            deleted=True,
            model_id="model_abc123",
            disk_freed_bytes=140000000000,
        )
        assert response.disk_freed_bytes == 140000000000


class TestModelInfoSchemas:
    """Tests for Model Info API schemas."""

    def test_model_info_response_serving(self):
        """Test ModelInfoResponse when model is serving."""
        from src.api.schemas import (
            ModelBasicInfo,
            ModelInfoResponse,
            ModelInfoResources,
            ModelInfoResourcesGPU,
            ModelInfoResourcesMemory,
            ModelInfoStats,
            ModelParameters,
            ModelRuntimeConfig,
        )

        response = ModelInfoResponse(
            serving=True,
            model=ModelBasicInfo(
                model_id="model_abc123",
                name="llama-3.1-70b",
                type="llm",
            ),
            resources=ModelInfoResources(
                gpu=ModelInfoResourcesGPU(
                    gpu_ids=[0, 1],
                    gpu_count=2,
                    total_memory_gb=160.0,
                    used_memory_gb=130.0,
                    memory_utilization_percent=81.25,
                    devices=[],
                ),
                memory=ModelInfoResourcesMemory(
                    model_memory_gb=12.5,
                    kv_cache_memory_gb=8.2,
                ),
            ),
            parameters=ModelParameters(
                runtime_config=ModelRuntimeConfig(
                    tensor_parallel_size=2,
                    max_model_len=8192,
                ),
            ),
            stats=ModelInfoStats(
                loaded_at="2026-01-05T12:30:00Z",
                uptime_seconds=7200,
                requests_served=12345,
                tokens_generated=4567890,
                avg_latency_ms=145.0,
                avg_tokens_per_second=52.3,
                current_batch_size=8,
                pending_requests=3,
            ),
        )
        assert response.serving is True
        assert response.model.name == "llama-3.1-70b"

    def test_model_info_response_not_serving(self):
        """Test ModelInfoResponse when no model is serving."""
        from src.api.schemas import ModelInfoResponse

        response = ModelInfoResponse(
            serving=False,
            message="No model is currently serving on this instance",
        )
        assert response.serving is False
        assert response.model is None
        assert response.resources is None


class TestChatCompletionsSchemas:
    """Tests for Chat Completions API schemas."""

    def test_chat_message(self):
        """Test ChatMessage schema."""
        from src.api.schemas import ChatMessage

        msg = ChatMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_chat_completion_request(self):
        """Test ChatCompletionRequest schema."""
        from src.api.schemas import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="llama-3.1-70b",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello!"),
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        assert request.model == "llama-3.1-70b"
        assert len(request.messages) == 2
        assert request.stream is False  # default

    def test_chat_completion_request_with_file_refs(self):
        """Test ChatCompletionRequest with file_refs extension."""
        from src.api.schemas import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="llama-3.1-70b",
            messages=[ChatMessage(role="user", content="Analyze this.")],
            file_refs=["file_abc123", "file_def456"],
        )
        assert request.file_refs == ["file_abc123", "file_def456"]

    def test_chat_completion_response(self):
        """Test ChatCompletionResponse schema."""
        from src.api.schemas import (
            ChatCompletionChoice,
            ChatCompletionResponse,
            ChatCompletionUsage,
            ChatMessage,
        )

        response = ChatCompletionResponse(
            id="chatcmpl-abc123",
            created=1704456000,
            model="llama-3.1-70b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Hello! How can I help you today?",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
            ),
        )
        assert response.object == "chat.completion"
        assert response.choices[0].finish_reason == "stop"

    def test_chat_completion_chunk(self):
        """Test ChatCompletionChunk schema for streaming."""
        from src.api.schemas import (
            ChatCompletionChunk,
            ChatCompletionChunkChoice,
            ChatCompletionDelta,
        )

        chunk = ChatCompletionChunk(
            id="chatcmpl-abc123",
            created=1704456000,
            model="llama-3.1-70b",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionDelta(content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "Hello"

    def test_chat_completion_chunk_finish(self):
        """Test ChatCompletionChunk with finish_reason."""
        from src.api.schemas import (
            ChatCompletionChunk,
            ChatCompletionChunkChoice,
            ChatCompletionDelta,
        )

        chunk = ChatCompletionChunk(
            id="chatcmpl-abc123",
            created=1704456000,
            model="llama-3.1-70b",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionDelta(),
                    finish_reason="stop",
                )
            ],
        )
        assert chunk.choices[0].finish_reason == "stop"


class TestErrorSchemas:
    """Tests for error response schemas."""

    def test_error_response(self):
        """Test ErrorResponse schema."""
        from src.api.schemas import ErrorResponse

        error = ErrorResponse(
            error="unsupported_model_type",
            message="Model type 'diffusion' is not supported",
        )
        assert error.error == "unsupported_model_type"

    def test_error_response_with_details(self):
        """Test ErrorResponse with additional details."""
        from src.api.schemas import ErrorResponse

        error = ErrorResponse(
            error="insufficient_disk_space",
            message="Not enough disk space",
            details={
                "required_bytes": 150000000000,
                "available_bytes": 53687091200,
            },
        )
        assert error.details["required_bytes"] == 150000000000
