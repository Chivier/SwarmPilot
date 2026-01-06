"""Unit tests for Resumable Upload API routes (tus protocol).

Tests for POST/PATCH/HEAD /v1/models/upload/resumable - EDI-48.
Tests follow TDD principle - written before implementation.
"""

import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for model storage."""
    data_dir = tmp_path / "models"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_upload_dir(tmp_path):
    """Create a temporary directory for upload sessions."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture
def app(temp_data_dir, temp_upload_dir):
    """Create a FastAPI app with resumable upload routes."""
    from src.api.routes import resumable_upload as upload_module
    from src.services.model_storage import ModelStorageService
    from src.services.upload_session import UploadSessionService

    app = FastAPI()
    app.include_router(
        upload_module.router, prefix="/v1/models/upload/resumable"
    )

    model_service = ModelStorageService(data_dir=temp_data_dir)
    upload_service = UploadSessionService(
        upload_dir=temp_upload_dir,
        model_storage=model_service,
    )
    upload_module.set_upload_session_service(upload_service)
    upload_module.set_model_storage_service(model_service)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


def encode_metadata(name: str, model_type: str) -> str:
    """Encode upload metadata in tus format.

    Args:
        name: Model name.
        model_type: Model type.

    Returns:
        Tus-formatted metadata string.
    """
    name_b64 = base64.b64encode(name.encode()).decode()
    type_b64 = base64.b64encode(model_type.encode()).decode()
    return f"name {name_b64},type {type_b64}"


class TestCreateUploadSession:
    """Tests for POST /v1/models/upload/resumable (create session)."""

    def test_create_session_returns_201(self, client):
        """Test creating upload session returns 201 Created."""
        response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "1000",
                "Upload-Metadata": encode_metadata("my-model", "llm"),
            },
        )
        assert response.status_code == 201

    def test_create_session_returns_location(self, client):
        """Test created session has Location header."""
        response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "1000",
                "Upload-Metadata": encode_metadata("my-model", "llm"),
            },
        )
        assert "Location" in response.headers
        assert "/v1/models/upload/resumable/" in response.headers["Location"]

    def test_create_session_returns_tus_headers(self, client):
        """Test response includes required tus headers."""
        response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "1000",
                "Upload-Metadata": encode_metadata("my-model", "llm"),
            },
        )
        assert response.headers.get("Tus-Resumable") == "1.0.0"

    def test_create_session_missing_tus_version(self, client):
        """Test 412 when Tus-Resumable header is missing."""
        response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Upload-Length": "1000",
                "Upload-Metadata": encode_metadata("my-model", "llm"),
            },
        )
        assert response.status_code == 412

    def test_create_session_invalid_tus_version(self, client):
        """Test 412 when Tus-Resumable version is invalid."""
        response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "2.0.0",
                "Upload-Length": "1000",
                "Upload-Metadata": encode_metadata("my-model", "llm"),
            },
        )
        assert response.status_code == 412

    def test_create_session_missing_upload_length(self, client):
        """Test 400 when Upload-Length is missing."""
        response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Metadata": encode_metadata("my-model", "llm"),
            },
        )
        assert response.status_code == 400

    def test_create_session_missing_metadata(self, client):
        """Test 400 when Upload-Metadata is missing."""
        response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "1000",
            },
        )
        assert response.status_code == 400


class TestUploadChunk:
    """Tests for PATCH /v1/models/upload/resumable/{upload_id}."""

    def test_upload_chunk_returns_204(self, client):
        """Test uploading chunk returns 204 No Content."""
        # Create session first
        create_response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": encode_metadata("test-model", "llm"),
            },
        )
        location = create_response.headers["Location"]
        upload_id = location.split("/")[-1]

        # Upload chunk
        response = client.patch(
            f"/v1/models/upload/resumable/{upload_id}",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"x" * 50,
        )
        assert response.status_code == 204

    def test_upload_chunk_returns_offset(self, client):
        """Test chunk upload returns new Upload-Offset."""
        # Create session
        create_response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": encode_metadata("test-model", "llm"),
            },
        )
        location = create_response.headers["Location"]
        upload_id = location.split("/")[-1]

        # Upload chunk
        response = client.patch(
            f"/v1/models/upload/resumable/{upload_id}",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"x" * 50,
        )
        assert response.headers["Upload-Offset"] == "50"

    def test_upload_multiple_chunks(self, client):
        """Test uploading file in multiple chunks."""
        # Create session
        create_response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": encode_metadata("test-model", "llm"),
            },
        )
        location = create_response.headers["Location"]
        upload_id = location.split("/")[-1]

        # Upload first chunk
        response1 = client.patch(
            f"/v1/models/upload/resumable/{upload_id}",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"a" * 50,
        )
        assert response1.status_code == 204
        assert response1.headers["Upload-Offset"] == "50"

        # Upload second chunk
        response2 = client.patch(
            f"/v1/models/upload/resumable/{upload_id}",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "50",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"b" * 50,
        )
        assert response2.status_code == 204
        assert response2.headers["Upload-Offset"] == "100"

    def test_upload_chunk_invalid_offset(self, client):
        """Test 409 when Upload-Offset doesn't match server state."""
        # Create session
        create_response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": encode_metadata("test-model", "llm"),
            },
        )
        location = create_response.headers["Location"]
        upload_id = location.split("/")[-1]

        # Upload with wrong offset
        response = client.patch(
            f"/v1/models/upload/resumable/{upload_id}",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "50",  # Should be 0
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"x" * 50,
        )
        assert response.status_code == 409

    def test_upload_chunk_not_found(self, client):
        """Test 404 for non-existent upload session."""
        response = client.patch(
            "/v1/models/upload/resumable/nonexistent_id",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"x" * 50,
        )
        assert response.status_code == 404


class TestCheckProgress:
    """Tests for HEAD /v1/models/upload/resumable/{upload_id}."""

    def test_check_progress_returns_200(self, client):
        """Test checking progress returns 200 OK."""
        # Create session
        create_response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": encode_metadata("test-model", "llm"),
            },
        )
        location = create_response.headers["Location"]
        upload_id = location.split("/")[-1]

        # Check progress
        response = client.head(f"/v1/models/upload/resumable/{upload_id}")
        assert response.status_code == 200

    def test_check_progress_returns_headers(self, client):
        """Test progress check returns required headers."""
        # Create session
        create_response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": encode_metadata("test-model", "llm"),
            },
        )
        location = create_response.headers["Location"]
        upload_id = location.split("/")[-1]

        # Check progress
        response = client.head(f"/v1/models/upload/resumable/{upload_id}")
        assert response.headers["Upload-Length"] == "100"
        assert response.headers["Upload-Offset"] == "0"
        assert response.headers["Tus-Resumable"] == "1.0.0"

    def test_check_progress_after_partial_upload(self, client):
        """Test progress reflects uploaded data."""
        # Create session
        create_response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": encode_metadata("test-model", "llm"),
            },
        )
        location = create_response.headers["Location"]
        upload_id = location.split("/")[-1]

        # Upload partial data
        client.patch(
            f"/v1/models/upload/resumable/{upload_id}",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"x" * 50,
        )

        # Check progress
        response = client.head(f"/v1/models/upload/resumable/{upload_id}")
        assert response.headers["Upload-Offset"] == "50"

    def test_check_progress_not_found(self, client):
        """Test 404 for non-existent upload session."""
        response = client.head("/v1/models/upload/resumable/nonexistent_id")
        assert response.status_code == 404


class TestUploadCompletion:
    """Tests for upload completion and model finalization."""

    def test_complete_upload_creates_model(self, client, temp_data_dir):
        """Test completed upload creates ready model."""
        from src.services.model_storage import ModelStorageService

        # Create session
        create_response = client.post(
            "/v1/models/upload/resumable",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": encode_metadata("complete-model", "llm"),
            },
        )
        location = create_response.headers["Location"]
        upload_id = location.split("/")[-1]

        # Upload all data
        response = client.patch(
            f"/v1/models/upload/resumable/{upload_id}",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"x" * 100,
        )
        assert response.status_code == 204

        # Get model_id from response header or body
        model_id = response.headers.get("X-Model-Id")
        assert model_id is not None

        # Verify model exists
        storage = ModelStorageService(data_dir=temp_data_dir)
        import asyncio

        model = asyncio.get_event_loop().run_until_complete(
            storage.get_model(model_id)
        )
        assert model is not None
        assert model.status == "ready"
