"""Unit tests for Model Upload API routes.

Tests for POST /v1/models/upload endpoint - EDI-47.
Tests follow TDD principle - written before implementation.
"""

import io
import tarfile
import zipfile

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
def app(temp_data_dir):
    """Create a FastAPI app with model upload routes."""
    from src.api.routes import models as models_module
    from src.services.model_storage import ModelStorageService

    app = FastAPI()
    app.include_router(models_module.router, prefix="/v1/models")

    model_service = ModelStorageService(data_dir=temp_data_dir)
    models_module.set_model_storage_service(model_service)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


def create_tar_gz(files: dict[str, bytes]) -> bytes:
    """Create a tar.gz archive in memory.

    Args:
        files: Dictionary mapping filenames to content bytes.

    Returns:
        tar.gz archive as bytes.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    buffer.seek(0)
    return buffer.read()


def create_zip(files: dict[str, bytes]) -> bytes:
    """Create a zip archive in memory.

    Args:
        files: Dictionary mapping filenames to content bytes.

    Returns:
        zip archive as bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    buffer.seek(0)
    return buffer.read()


class TestModelUploadBasic:
    """Basic tests for POST /v1/models/upload."""

    def test_upload_returns_200(self, client):
        """Test that upload endpoint returns 200 OK."""
        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.safetensors", b"fake model data")},
            data={"name": "test-model", "type": "llm"},
        )
        assert response.status_code == 200

    def test_upload_response_format(self, client):
        """Test upload response has expected fields."""
        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.safetensors", b"fake model data")},
            data={"name": "test-model", "type": "llm"},
        )
        data = response.json()

        assert "model_id" in data
        assert data["model_id"].startswith("model_")
        assert "name" in data
        assert data["name"] == "test-model"
        assert "type" in data
        assert data["type"] == "llm"
        assert "status" in data
        assert "message" in data

    def test_upload_single_safetensors(self, client, temp_data_dir):
        """Test uploading a single .safetensors file."""
        response = client.post(
            "/v1/models/upload",
            files={"file": ("weights.safetensors", b"fake safetensors data")},
            data={"name": "my-model", "type": "llm"},
        )
        assert response.status_code == 200

        data = response.json()
        model_id = data["model_id"]

        # Verify file was saved
        model_dir = temp_data_dir / model_id
        assert model_dir.exists()
        assert (model_dir / "weights.safetensors").exists()


class TestModelUploadArchives:
    """Tests for archive extraction during upload."""

    def test_upload_tar_gz_archive(self, client, temp_data_dir):
        """Test uploading a .tar.gz archive."""
        archive = create_tar_gz({
            "model.safetensors": b"fake model weights",
            "config.json": b'{"model_type": "llm"}',
        })

        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.tar.gz", archive)},
            data={"name": "tar-model", "type": "llm"},
        )
        assert response.status_code == 200

        data = response.json()
        model_id = data["model_id"]

        # Verify files were extracted
        model_dir = temp_data_dir / model_id
        assert model_dir.exists()
        assert (model_dir / "model.safetensors").exists()
        assert (model_dir / "config.json").exists()

    def test_upload_zip_archive(self, client, temp_data_dir):
        """Test uploading a .zip archive."""
        archive = create_zip({
            "weights.bin": b"fake model weights",
            "tokenizer.json": b'{"type": "bpe"}',
        })

        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.zip", archive)},
            data={"name": "zip-model", "type": "llm"},
        )
        assert response.status_code == 200

        data = response.json()
        model_id = data["model_id"]

        # Verify files were extracted
        model_dir = temp_data_dir / model_id
        assert model_dir.exists()
        assert (model_dir / "weights.bin").exists()
        assert (model_dir / "tokenizer.json").exists()


class TestModelUploadValidation:
    """Tests for upload validation."""

    def test_upload_missing_file(self, client):
        """Test 422 when file is missing."""
        response = client.post(
            "/v1/models/upload",
            data={"name": "test-model", "type": "llm"},
        )
        assert response.status_code == 422

    def test_upload_missing_name(self, client):
        """Test 422 when name is missing."""
        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.safetensors", b"data")},
            data={"type": "llm"},
        )
        assert response.status_code == 422

    def test_upload_missing_type(self, client):
        """Test 422 when type is missing."""
        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.safetensors", b"data")},
            data={"name": "test-model"},
        )
        assert response.status_code == 422

    def test_upload_invalid_archive_type(self, client):
        """Test 400 for unsupported archive type."""
        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.rar", b"not a valid archive")},
            data={"name": "test-model", "type": "llm"},
        )
        assert response.status_code == 400

    def test_upload_corrupted_tar_gz(self, client):
        """Test 400 for corrupted tar.gz archive."""
        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.tar.gz", b"not a valid tar.gz")},
            data={"name": "test-model", "type": "llm"},
        )
        assert response.status_code == 400

    def test_upload_corrupted_zip(self, client):
        """Test 400 for corrupted zip archive."""
        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.zip", b"not a valid zip")},
            data={"name": "test-model", "type": "llm"},
        )
        assert response.status_code == 400


class TestModelUploadStatus:
    """Tests for model status after upload."""

    def test_upload_sets_ready_status(self, client, temp_data_dir):
        """Test that upload sets model status to ready."""
        from src.services.model_storage import ModelStorageService

        response = client.post(
            "/v1/models/upload",
            files={"file": ("model.safetensors", b"fake model data")},
            data={"name": "test-model", "type": "llm"},
        )
        assert response.status_code == 200

        data = response.json()
        model_id = data["model_id"]

        # Check status via storage service
        storage = ModelStorageService(data_dir=temp_data_dir)
        import asyncio

        model = asyncio.get_event_loop().run_until_complete(
            storage.get_model(model_id)
        )
        assert model is not None
        assert model.status == "ready"


class TestModelUploadSource:
    """Tests for model source metadata after upload."""

    def test_upload_sets_upload_source(self, client, temp_data_dir):
        """Test that uploaded model has 'upload' source type."""
        from src.services.model_storage import ModelStorageService

        response = client.post(
            "/v1/models/upload",
            files={"file": ("weights.safetensors", b"fake data")},
            data={"name": "uploaded-model", "type": "llm"},
        )
        assert response.status_code == 200

        data = response.json()
        model_id = data["model_id"]

        storage = ModelStorageService(data_dir=temp_data_dir)
        import asyncio

        model = asyncio.get_event_loop().run_until_complete(
            storage.get_model(model_id)
        )
        assert model is not None
        assert model.source.type == "upload"
        assert model.source.filename == "weights.safetensors"
