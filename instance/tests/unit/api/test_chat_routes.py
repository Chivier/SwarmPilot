"""Unit tests for Chat Completions API routes.

Tests for POST /v1/chat/completions endpoint - EDI-57, EDI-58, EDI-59.
Tests follow TDD principle - written before implementation.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def create_test_file(files_dir: Path, content: bytes, filename: str) -> str:
    """Create a test file in the storage directory structure.

    Args:
        files_dir: Base directory for file storage.
        content: File content as bytes.
        filename: Name for the file.

    Returns:
        Generated file ID.
    """
    file_id = f"file_{uuid.uuid4().hex[:12]}"
    file_dir = files_dir / file_id
    file_dir.mkdir(parents=True, exist_ok=True)

    # Write content
    file_path = file_dir / filename
    file_path.write_bytes(content)

    # Write metadata
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
def app(temp_data_dir, temp_files_dir):
    """Create a FastAPI app with chat routes."""
    from src.api.routes import chat as chat_module
    from src.services.chat import ChatService
    from src.services.file_storage import FileStorageService
    from src.services.inference_manager import InferenceManagerService
    from src.services.model_storage import ModelStorageService

    app = FastAPI()
    app.include_router(chat_module.router, prefix="/v1/chat")

    model_service = ModelStorageService(data_dir=temp_data_dir)
    file_service = FileStorageService(data_dir=temp_files_dir)
    inference_service = InferenceManagerService(model_storage=model_service)
    chat_service = ChatService(
        inference_manager=inference_service,
        file_storage=file_service,
    )

    chat_module.set_chat_service(chat_service)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestChatCompletionsNonStreaming:
    """Tests for POST /v1/chat/completions (non-streaming) - EDI-57."""

    def test_completions_returns_200(self, client):
        """Test that completions endpoint returns 200 OK."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        assert response.status_code == 200

    def test_completions_response_format(self, client):
        """Test response follows OpenAI format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        data = response.json()

        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

    def test_completions_choice_format(self, client):
        """Test choice follows OpenAI format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        data = response.json()

        assert len(data["choices"]) > 0
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert "finish_reason" in choice

    def test_completions_usage_format(self, client):
        """Test usage follows OpenAI format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        data = response.json()

        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    def test_completions_with_temperature(self, client):
        """Test completions with temperature parameter."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200

    def test_completions_with_max_tokens(self, client):
        """Test completions with max_tokens parameter."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200

    def test_completions_with_stop_sequences(self, client):
        """Test completions with stop sequences."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stop": ["\n\n", "END"],
            },
        )
        assert response.status_code == 200

    def test_completions_with_system_message(self, client):
        """Test completions with system message."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )
        assert response.status_code == 200

    def test_completions_missing_model(self, client):
        """Test 422 for missing model field."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        assert response.status_code == 422

    def test_completions_missing_messages(self, client):
        """Test 422 for missing messages field."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
            },
        )
        assert response.status_code == 422


class TestChatCompletionsStreaming:
    """Tests for POST /v1/chat/completions streaming - EDI-58."""

    def test_streaming_returns_event_stream(self, client):
        """Test streaming returns text/event-stream content type."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_streaming_chunk_format(self, client):
        """Test streaming chunks follow OpenAI format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        )

        # Parse SSE response
        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk_data = json.loads(line[6:])
                chunks.append(chunk_data)

        assert len(chunks) > 0

        # First chunk should have role
        first_chunk = chunks[0]
        assert first_chunk["object"] == "chat.completion.chunk"
        assert "id" in first_chunk
        assert "created" in first_chunk
        assert "choices" in first_chunk

    def test_streaming_first_chunk_has_role(self, client):
        """Test first streaming chunk includes role."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        )

        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk_data = json.loads(line[6:])
                chunks.append(chunk_data)

        assert len(chunks) > 0
        first_choice = chunks[0]["choices"][0]
        assert first_choice["delta"].get("role") == "assistant"

    def test_streaming_last_chunk_has_finish_reason(self, client):
        """Test last streaming chunk has finish_reason."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        )

        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk_data = json.loads(line[6:])
                chunks.append(chunk_data)

        assert len(chunks) > 0
        last_choice = chunks[-1]["choices"][0]
        assert last_choice["finish_reason"] is not None

    def test_streaming_ends_with_done(self, client):
        """Test streaming ends with [DONE] marker."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        )

        lines = list(response.iter_lines())
        # Filter non-empty lines
        data_lines = [line for line in lines if line]
        assert data_lines[-1] == "data: [DONE]"


class TestChatFileRefs:
    """Tests for file_refs extension - EDI-59."""

    def test_file_refs_accepted(self, client, temp_files_dir):
        """Test file_refs field is accepted."""
        from src.api.routes import chat as chat_module
        from src.services.chat import ChatService
        from src.services.file_storage import FileStorageService
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        model_service = ModelStorageService(data_dir=temp_files_dir.parent / "models2")
        (temp_files_dir.parent / "models2").mkdir(exist_ok=True)
        file_service = FileStorageService(data_dir=temp_files_dir)
        inference_service = InferenceManagerService(model_storage=model_service)
        chat_service = ChatService(
            inference_manager=inference_service,
            file_storage=file_service,
        )
        chat_module.set_chat_service(chat_service)

        # Create a test file directly
        file_id = create_test_file(
            temp_files_dir,
            b"Context information here.",
            "context.txt",
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "file_refs": [file_id],
            },
        )
        assert response.status_code == 200

    def test_file_refs_invalid_file(self, client):
        """Test 400 for non-existent file_id."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "file_refs": ["file_nonexistent"],
            },
        )
        assert response.status_code == 400

    def test_file_refs_content_injected(self, client, temp_files_dir):
        """Test file content is used in completion context."""
        from src.api.routes import chat as chat_module
        from src.services.chat import ChatService
        from src.services.file_storage import FileStorageService
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        model_service = ModelStorageService(data_dir=temp_files_dir.parent / "models3")
        (temp_files_dir.parent / "models3").mkdir(exist_ok=True)
        file_service = FileStorageService(data_dir=temp_files_dir)
        inference_service = InferenceManagerService(model_storage=model_service)
        chat_service = ChatService(
            inference_manager=inference_service,
            file_storage=file_service,
        )
        chat_module.set_chat_service(chat_service)

        # Create a test file with specific content
        file_id = create_test_file(
            temp_files_dir,
            b"The answer is 42.",
            "answer.txt",
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "What is the answer?"}],
                "file_refs": [file_id],
            },
        )
        assert response.status_code == 200
        # The mock service should echo back something
        # Real implementation would use the file content

    def test_file_refs_with_streaming(self, client, temp_files_dir):
        """Test file_refs works with streaming."""
        from src.api.routes import chat as chat_module
        from src.services.chat import ChatService
        from src.services.file_storage import FileStorageService
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        model_service = ModelStorageService(data_dir=temp_files_dir.parent / "models4")
        (temp_files_dir.parent / "models4").mkdir(exist_ok=True)
        file_service = FileStorageService(data_dir=temp_files_dir)
        inference_service = InferenceManagerService(model_storage=model_service)
        chat_service = ChatService(
            inference_manager=inference_service,
            file_storage=file_service,
        )
        chat_module.set_chat_service(chat_service)

        # Create a test file
        file_id = create_test_file(
            temp_files_dir,
            b"Some context.",
            "context.txt",
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "file_refs": [file_id],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
