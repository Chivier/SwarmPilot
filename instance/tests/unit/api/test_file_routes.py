"""Unit tests for File API routes.

Tests follow TDD principle - written before implementation.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "files"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def app(temp_data_dir):
    """Create a FastAPI app with file routes."""
    from src.api.routes.files import router, set_storage_service
    from src.services.file_storage import FileStorageService

    app = FastAPI()
    app.include_router(router, prefix="/v1/files")

    # Initialize storage service with temp directory
    service = FileStorageService(data_dir=temp_data_dir)
    set_storage_service(service)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestUploadEndpoint:
    """Tests for POST /v1/files/upload endpoint."""

    def test_upload_returns_201(self, client):
        """Test that upload endpoint returns 201 Created."""
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}

        response = client.post("/v1/files/upload", files=files, data=data)

        assert response.status_code == 201

    def test_upload_returns_file_info(self, client):
        """Test that upload returns FileUploadResponse fields."""
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}

        response = client.post("/v1/files/upload", files=files, data=data)
        result = response.json()

        assert "file_id" in result
        assert result["file_id"].startswith("file_")
        assert result["filename"] == "test.jsonl"
        assert result["purpose"] == "inference"
        assert "size_bytes" in result
        assert "created_at" in result

    def test_upload_with_tags(self, client):
        """Test that tags are parsed correctly."""
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference", "tags": "training,v2"}

        response = client.post("/v1/files/upload", files=files, data=data)

        assert response.status_code == 201

    def test_upload_with_ttl(self, client):
        """Test that ttl_hours is accepted."""
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference", "ttl_hours": "24"}

        response = client.post("/v1/files/upload", files=files, data=data)

        assert response.status_code == 201

    def test_upload_batch_purpose(self, client):
        """Test upload with batch purpose."""
        files = {"file": ("batch.jsonl", b'{"data": "batch"}\n', "application/jsonl")}
        data = {"purpose": "batch"}

        response = client.post("/v1/files/upload", files=files, data=data)
        result = response.json()

        assert response.status_code == 201
        assert result["purpose"] == "batch"

    def test_upload_missing_file(self, client):
        """Test error when file is missing."""
        data = {"purpose": "inference"}

        response = client.post("/v1/files/upload", data=data)

        assert response.status_code == 422

    def test_upload_missing_purpose(self, client):
        """Test error when purpose is missing."""
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}

        response = client.post("/v1/files/upload", files=files)

        assert response.status_code == 422

    def test_upload_insufficient_disk_space(self, client, temp_data_dir):
        """Test error when disk space is insufficient."""
        from src.api.routes.files import set_storage_service
        from src.services.file_storage import FileStorageService

        # Create service with impossible min free requirement
        service = FileStorageService(data_dir=temp_data_dir, min_free_disk_gb=10000000)
        set_storage_service(service)

        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}

        response = client.post("/v1/files/upload", files=files, data=data)

        assert response.status_code == 507


class TestDownloadEndpoint:
    """Tests for GET /v1/files/{file_id} endpoint."""

    def test_download_returns_200(self, client):
        """Test that download returns 200 for existing file."""
        # First upload a file
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}
        upload_response = client.post("/v1/files/upload", files=files, data=data)
        file_id = upload_response.json()["file_id"]

        # Then download it
        response = client.get(f"/v1/files/{file_id}")

        assert response.status_code == 200

    def test_download_returns_file_content(self, client):
        """Test that download returns correct file content."""
        content = b'{"data": "test content"}\n'
        files = {"file": ("test.jsonl", content, "application/jsonl")}
        data = {"purpose": "inference"}
        upload_response = client.post("/v1/files/upload", files=files, data=data)
        file_id = upload_response.json()["file_id"]

        response = client.get(f"/v1/files/{file_id}")

        assert response.content == content

    def test_download_sets_content_disposition(self, client):
        """Test that download sets Content-Disposition header."""
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}
        upload_response = client.post("/v1/files/upload", files=files, data=data)
        file_id = upload_response.json()["file_id"]

        response = client.get(f"/v1/files/{file_id}")

        assert "content-disposition" in response.headers
        assert "test.jsonl" in response.headers["content-disposition"]

    def test_download_not_found(self, client):
        """Test 404 for non-existent file."""
        response = client.get("/v1/files/file_nonexistent")

        assert response.status_code == 404

    def test_download_error_response(self, client):
        """Test error response format for 404."""
        response = client.get("/v1/files/file_nonexistent")
        result = response.json()

        assert "detail" in result
        assert result["detail"]["error"] == "file_not_found"


class TestListEndpoint:
    """Tests for GET /v1/files endpoint."""

    def test_list_returns_200(self, client):
        """Test that list returns 200 OK."""
        response = client.get("/v1/files")

        assert response.status_code == 200

    def test_list_empty_returns_structure(self, client):
        """Test list returns proper structure when empty."""
        response = client.get("/v1/files")
        result = response.json()

        assert "files" in result
        assert result["files"] == []
        assert "total" in result
        assert result["total"] == 0

    def test_list_returns_uploaded_files(self, client):
        """Test that list returns uploaded files."""
        # Upload a file
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}
        client.post("/v1/files/upload", files=files, data=data)

        response = client.get("/v1/files")
        result = response.json()

        assert len(result["files"]) == 1
        assert result["total"] == 1

    def test_list_filter_by_purpose(self, client):
        """Test filtering by purpose."""
        # Upload two files with different purposes
        client.post(
            "/v1/files/upload",
            files={"file": ("test1.jsonl", b'{"data": "1"}\n', "application/jsonl")},
            data={"purpose": "inference"},
        )
        client.post(
            "/v1/files/upload",
            files={"file": ("test2.jsonl", b'{"data": "2"}\n', "application/jsonl")},
            data={"purpose": "batch"},
        )

        response = client.get("/v1/files?purpose=inference")
        result = response.json()

        assert len(result["files"]) == 1
        assert result["files"][0]["purpose"] == "inference"

    def test_list_filter_by_tag(self, client):
        """Test filtering by tag."""
        # Upload two files with different tags
        client.post(
            "/v1/files/upload",
            files={"file": ("test1.jsonl", b'{"data": "1"}\n', "application/jsonl")},
            data={"purpose": "inference", "tags": "training"},
        )
        client.post(
            "/v1/files/upload",
            files={"file": ("test2.jsonl", b'{"data": "2"}\n', "application/jsonl")},
            data={"purpose": "inference", "tags": "evaluation"},
        )

        response = client.get("/v1/files?tag=training")
        result = response.json()

        assert len(result["files"]) == 1
        assert "training" in result["files"][0]["tags"]

    def test_list_pagination_limit(self, client):
        """Test pagination with limit."""
        # Upload multiple files
        for i in range(5):
            client.post(
                "/v1/files/upload",
                files={
                    "file": (f"test{i}.jsonl", b'{"data": "test"}\n', "application/jsonl")
                },
                data={"purpose": "inference"},
            )

        response = client.get("/v1/files?limit=3")
        result = response.json()

        assert len(result["files"]) == 3
        assert result["total"] == 5

    def test_list_pagination_offset(self, client):
        """Test pagination with offset."""
        # Upload multiple files
        for i in range(5):
            client.post(
                "/v1/files/upload",
                files={
                    "file": (f"test{i}.jsonl", b'{"data": "test"}\n', "application/jsonl")
                },
                data={"purpose": "inference"},
            )

        response = client.get("/v1/files?limit=10&offset=3")
        result = response.json()

        assert len(result["files"]) == 2


class TestDeleteEndpoint:
    """Tests for DELETE /v1/files/{file_id} endpoint."""

    def test_delete_returns_200(self, client):
        """Test that delete returns 200 OK."""
        # Upload a file first
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}
        upload_response = client.post("/v1/files/upload", files=files, data=data)
        file_id = upload_response.json()["file_id"]

        response = client.delete(f"/v1/files/{file_id}")

        assert response.status_code == 200

    def test_delete_returns_response(self, client):
        """Test delete response format."""
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}
        upload_response = client.post("/v1/files/upload", files=files, data=data)
        file_id = upload_response.json()["file_id"]

        response = client.delete(f"/v1/files/{file_id}")
        result = response.json()

        assert result["deleted"] is True
        assert result["file_id"] == file_id

    def test_delete_removes_file(self, client):
        """Test that delete actually removes the file."""
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}
        upload_response = client.post("/v1/files/upload", files=files, data=data)
        file_id = upload_response.json()["file_id"]

        # Delete the file
        client.delete(f"/v1/files/{file_id}")

        # Try to download - should fail
        response = client.get(f"/v1/files/{file_id}")
        assert response.status_code == 404

    def test_delete_not_found(self, client):
        """Test 404 for non-existent file."""
        response = client.delete("/v1/files/file_nonexistent")

        assert response.status_code == 404

    def test_delete_error_response(self, client):
        """Test error response format for 404."""
        response = client.delete("/v1/files/file_nonexistent")
        result = response.json()

        assert "detail" in result
        assert result["detail"]["error"] == "file_not_found"

    def test_delete_not_in_list_after(self, client):
        """Test file doesn't appear in list after deletion."""
        files = {"file": ("test.jsonl", b'{"data": "test"}\n', "application/jsonl")}
        data = {"purpose": "inference"}
        upload_response = client.post("/v1/files/upload", files=files, data=data)
        file_id = upload_response.json()["file_id"]

        # Delete the file
        client.delete(f"/v1/files/{file_id}")

        # List files - should be empty
        response = client.get("/v1/files")
        result = response.json()

        assert result["total"] == 0
        assert len(result["files"]) == 0
