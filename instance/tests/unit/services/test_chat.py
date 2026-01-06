"""Unit tests for ChatService.

Tests follow TDD principle.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest


def create_test_file(files_dir: Path, content: bytes, filename: str) -> str:
    """Create a test file in the storage directory structure."""
    file_id = f"file_{uuid.uuid4().hex[:12]}"
    file_dir = files_dir / file_id
    file_dir.mkdir(parents=True, exist_ok=True)

    file_path = file_dir / filename
    file_path.write_bytes(content)

    metadata = {
        "file_id": file_id,
        "filename": filename,
        "size_bytes": len(content),
        "purpose": "assistants",
        "tags": [],
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "expires_at": None,
    }
    metadata_path = file_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    return file_id


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "models"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_files_dir(tmp_path):
    """Create a temporary files directory."""
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    return files_dir


@pytest.fixture
def chat_service(temp_data_dir, temp_files_dir):
    """Create a ChatService instance."""
    from src.services.chat import ChatService
    from src.services.file_storage import FileStorageService
    from src.services.inference_manager import InferenceManagerService
    from src.services.model_storage import ModelStorageService

    model_storage = ModelStorageService(data_dir=temp_data_dir)
    file_storage = FileStorageService(data_dir=temp_files_dir)
    inference_manager = InferenceManagerService(model_storage=model_storage)

    return ChatService(
        inference_manager=inference_manager,
        file_storage=file_storage,
    )


class TestChatServiceInit:
    """Tests for ChatService initialization."""

    def test_init_creates_service(self, chat_service):
        """Test that service can be instantiated."""
        assert chat_service is not None

    def test_init_with_dependencies(self, temp_data_dir, temp_files_dir):
        """Test initialization with injected dependencies."""
        from src.services.chat import ChatService
        from src.services.file_storage import FileStorageService
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        model_storage = ModelStorageService(data_dir=temp_data_dir)
        file_storage = FileStorageService(data_dir=temp_files_dir)
        inference_manager = InferenceManagerService(model_storage=model_storage)

        service = ChatService(
            inference_manager=inference_manager,
            file_storage=file_storage,
        )

        assert service.inference_manager is inference_manager
        assert service.file_storage is file_storage


class TestComplete:
    """Tests for complete method."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self, chat_service):
        """Test that complete returns a response."""
        from src.api.schemas import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello!")],
        )

        response = await chat_service.complete(request)

        assert response is not None
        assert response.id.startswith("chatcmpl-")

    @pytest.mark.asyncio
    async def test_complete_response_format(self, chat_service):
        """Test response follows OpenAI format."""
        from src.api.schemas import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello!")],
        )

        response = await chat_service.complete(request)

        assert response.object == "chat.completion"
        assert response.model == "test-model"
        assert len(response.choices) > 0
        assert response.choices[0].index == 0
        assert response.choices[0].message.role == "assistant"
        assert response.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_complete_with_system_message(self, chat_service):
        """Test complete with system message."""
        from src.api.schemas import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello!"),
            ],
        )

        response = await chat_service.complete(request)

        assert response is not None
        assert response.choices[0].finish_reason == "stop"


class TestCompleteStream:
    """Tests for complete_stream method."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, chat_service):
        """Test that streaming yields chunks."""
        from src.api.schemas import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello!")],
            stream=True,
        )

        chunks = []
        async for chunk in chat_service.complete_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_chunk_format(self, chat_service):
        """Test streaming chunks are SSE formatted."""
        from src.api.schemas import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello!")],
            stream=True,
        )

        chunks = []
        async for chunk in chat_service.complete_stream(request):
            chunks.append(chunk)

        # All chunks except [DONE] should be JSON
        for chunk in chunks[:-1]:
            assert chunk.startswith("data: ")
            data = json.loads(chunk[6:].strip())
            assert "id" in data
            assert data["object"] == "chat.completion.chunk"


class TestFileRefs:
    """Tests for file_refs handling."""

    @pytest.mark.asyncio
    async def test_resolve_file_refs_empty(self, chat_service):
        """Test resolving empty file refs."""
        result = await chat_service._resolve_file_refs(None)
        assert result == []

        result = await chat_service._resolve_file_refs([])
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_file_refs_success(self, chat_service, temp_files_dir):
        """Test resolving valid file refs."""
        file_id = create_test_file(
            temp_files_dir, b"Test content", "test.txt"
        )

        result = await chat_service._resolve_file_refs([file_id])

        assert len(result) == 1
        assert result[0] == "Test content"

    @pytest.mark.asyncio
    async def test_resolve_file_refs_not_found(self, chat_service):
        """Test resolving non-existent file raises error."""
        from src.services.chat import FileRefNotFoundError

        with pytest.raises(FileRefNotFoundError) as exc_info:
            await chat_service._resolve_file_refs(["file_nonexistent"])

        assert exc_info.value.file_id == "file_nonexistent"

    @pytest.mark.asyncio
    async def test_complete_with_file_refs(self, chat_service, temp_files_dir):
        """Test complete with file_refs."""
        from src.api.schemas import ChatCompletionRequest, ChatMessage

        file_id = create_test_file(
            temp_files_dir, b"Important context", "context.txt"
        )

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello!")],
            file_refs=[file_id],
        )

        response = await chat_service.complete(request)

        assert response is not None

    @pytest.mark.asyncio
    async def test_stream_with_file_refs(self, chat_service, temp_files_dir):
        """Test streaming with file_refs."""
        from src.api.schemas import ChatCompletionRequest, ChatMessage

        file_id = create_test_file(
            temp_files_dir, b"Streaming context", "context.txt"
        )

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello!")],
            file_refs=[file_id],
            stream=True,
        )

        chunks = []
        async for chunk in chat_service.complete_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0


class TestInjectFileContext:
    """Tests for _inject_file_context method."""

    def test_inject_empty_context(self, chat_service):
        """Test injecting empty file context."""
        from src.api.schemas import ChatMessage

        messages = [ChatMessage(role="user", content="Hello!")]
        result = chat_service._inject_file_context(messages, [])

        assert len(result) == 1
        assert result[0].content == "Hello!"

    def test_inject_context_no_system(self, chat_service):
        """Test injecting context when no system message exists."""
        from src.api.schemas import ChatMessage

        messages = [ChatMessage(role="user", content="Hello!")]
        result = chat_service._inject_file_context(messages, ["File content"])

        assert len(result) == 2
        assert result[0].role == "system"
        assert "File content" in result[0].content
        assert result[1].content == "Hello!"

    def test_inject_context_with_system(self, chat_service):
        """Test injecting context when system message exists."""
        from src.api.schemas import ChatMessage

        messages = [
            ChatMessage(role="system", content="Be helpful."),
            ChatMessage(role="user", content="Hello!"),
        ]
        result = chat_service._inject_file_context(messages, ["File content"])

        assert len(result) == 2
        assert result[0].role == "system"
        assert "File content" in result[0].content
        assert "Be helpful." in result[0].content
        assert result[1].content == "Hello!"
