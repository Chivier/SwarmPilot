# API Schemas Reference

This document describes the Pydantic schemas used for the SwarmX Instance API.

## Quick Reference

| Namespace | Request Schemas | Response Schemas |
|-----------|-----------------|------------------|
| System | - | `HealthResponse`, `SystemInfo` |
| Files | - | `FileUploadResponse`, `FileInfo`, `FileListResponse`, `FileDeleteResponse` |
| Models (OpenAI) | - | `OpenAIModelListResponse`, `OpenAIModel` |
| Models (SwarmX) | `ModelPullRequest`, `ModelStartRequest`, `ModelStopRequest`, `ModelSwitchRequest`, `ModelConfigRequest` | `ModelPullResponse`, `ModelListResponse`, `ModelDetailResponse`, `TaskProgressResponse`, `ModelDeleteResponse` |
| Model Info | - | `ModelInfoResponse` |
| Chat | `ChatCompletionRequest` | `ChatCompletionResponse`, `ChatCompletionChunk` |
| Error | - | `ErrorResponse` |

## System API Schemas

### HealthResponse
Simple health check response.

```python
from src.api.schemas import HealthResponse

response = HealthResponse(
    status="healthy",
    timestamp="2026-01-05T12:00:00Z"
)
```

### SystemInfo
Full system resource information.

```python
from src.api.schemas import SystemInfo, SystemResources, CPUInfo, MemoryInfo, DiskInfo, GPUInfo, InferenceServerInfo

info = SystemInfo(
    instance_id="inst_xyz789",
    uptime_seconds=3600,
    supported_model_types=["llm"],
    inference_server=InferenceServerInfo(type="vllm", version="0.6.0"),
    resources=SystemResources(
        cpu=CPUInfo(cores=8, usage_percent=45.2),
        memory=MemoryInfo(total_gb=64.0, used_gb=32.5, usage_percent=50.8),
        disk=DiskInfo(total_gb=500.0, used_gb=120.0, available_gb=380.0, min_free_gb=10.0, usage_percent=24.0),
        gpu=[GPUInfo(index=0, name="NVIDIA A100", memory_total_gb=80.0, memory_used_gb=45.0, utilization_percent=78.5, temperature_celsius=65)]
    )
)
```

## File API Schemas

### FileUploadResponse
Response after successful file upload.

| Field | Type | Description |
|-------|------|-------------|
| file_id | str | Unique file identifier (e.g., "file_abc123") |
| filename | str | Original filename |
| purpose | str | File purpose ("inference", "batch") |
| size_bytes | int | File size in bytes |
| created_at | str | ISO-8601 creation timestamp |

### FileInfo
File information with optional tags.

| Field | Type | Description |
|-------|------|-------------|
| file_id | str | Unique file identifier |
| filename | str | Original filename |
| purpose | str | File purpose |
| size_bytes | int | File size in bytes |
| tags | list[str] | Optional tags (default: []) |
| created_at | str | ISO-8601 creation timestamp |

## Models API Schemas

### OpenAI-Compatible

#### OpenAIModel
OpenAI-compatible model object.

```python
from src.api.schemas import OpenAIModel

model = OpenAIModel(
    id="llama-3.1-70b",
    created=1704456000,
    owned_by="swarmx"
)
# model.object is automatically "model"
```

#### OpenAIModelListResponse
OpenAI-compatible model list.

```python
from src.api.schemas import OpenAIModelListResponse, OpenAIModel

response = OpenAIModelListResponse(
    data=[OpenAIModel(id="llama-3.1-70b", created=1704456000, owned_by="swarmx")]
)
# response.object is automatically "list"
```

### SwarmX Extensions

#### ModelPullRequest
Request to pull model from HuggingFace.

```python
from src.api.schemas import ModelPullRequest, ModelPullSource

request = ModelPullRequest(
    name="llama-3.1-70b",
    type="llm",  # Only "llm" supported currently
    source=ModelPullSource(
        repo="meta-llama/Llama-3.1-70B-Instruct",
        revision="main",
        endpoint="https://huggingface.co",
        token="hf_xxx..."  # Optional
    )
)
```

**Validation**: The `type` field is validated to only accept supported model types (currently only "llm").

#### ModelStartRequest
Request to start a model.

```python
from src.api.schemas import ModelStartRequest, ModelConfigRequest

# With explicit config
request = ModelStartRequest(
    gpu_ids=[0, 1],
    config=ModelConfigRequest(
        tensor_parallel_size=2,
        max_model_len=8192,
        quantization=None,
        gpu_memory_utilization=0.9
    )
)

# Using default config
request = ModelStartRequest(gpu_ids=[0, 1])
```

#### ModelConfigRequest
Runtime configuration options for LLM models.

| Field | Type | Description |
|-------|------|-------------|
| tensor_parallel_size | int | None | Number of GPUs for tensor parallelism |
| max_model_len | int | None | Maximum sequence length |
| quantization | str | None | Quantization method: null, "awq", "gptq", "squeezellm" |
| gpu_memory_utilization | float | None | GPU memory fraction (0.0-1.0) |
| dtype | str | None | Data type: "auto", "float16", "bfloat16" |
| enforce_eager | bool | None | Disable CUDA graphs for debugging |

## Chat Completions Schemas

### ChatCompletionRequest
OpenAI-compatible chat completion request with SwarmX extensions.

```python
from src.api.schemas import ChatCompletionRequest, ChatMessage

request = ChatCompletionRequest(
    model="llama-3.1-70b",
    messages=[
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello!")
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=False,
    # SwarmX extension:
    file_refs=["file_abc123"]  # Optional file references
)
```

### ChatCompletionResponse
OpenAI-compatible response.

```python
from src.api.schemas import ChatCompletionResponse, ChatCompletionChoice, ChatMessage, ChatCompletionUsage

response = ChatCompletionResponse(
    id="chatcmpl-abc123",
    created=1704456000,
    model="llama-3.1-70b",
    choices=[
        ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hello!"),
            finish_reason="stop"
        )
    ],
    usage=ChatCompletionUsage(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30
    )
)
```

### ChatCompletionChunk
Streaming chunk for Server-Sent Events.

```python
from src.api.schemas import ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta

chunk = ChatCompletionChunk(
    id="chatcmpl-abc123",
    created=1704456000,
    model="llama-3.1-70b",
    choices=[
        ChatCompletionChunkChoice(
            index=0,
            delta=ChatCompletionDelta(content="Hello"),
            finish_reason=None
        )
    ]
)
```

## Error Response

### ErrorResponse
Standard error response format.

```python
from src.api.schemas import ErrorResponse

error = ErrorResponse(
    error="insufficient_disk_space",
    message="Not enough disk space. Required: 140GB, Available: 50GB",
    details={
        "required_bytes": 150000000000,
        "available_bytes": 53687091200
    }
)
```

## Enums

### ModelType
Supported model types.

| Value | Description | Status |
|-------|-------------|--------|
| `llm` | Large Language Models | Supported |

### ModelStatus
Model lifecycle status values.

| Value | Description |
|-------|-------------|
| `pulling` | Downloading weights |
| `uploading` | Uploading weights |
| `extracting` | Extracting archive |
| `ready` | Ready to start |
| `loading` | Loading into GPU |
| `running` | Actively serving |
| `stopping` | Graceful shutdown |
| `stopped` | Not running |
| `error` | Error state |
