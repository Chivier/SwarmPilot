"""Unit tests for FileStorageService.

Tests follow TDD principle - written before implementation.
"""

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "files"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def storage_service(temp_data_dir):
    """Create a FileStorageService instance."""
    from src.services.file_storage import FileStorageService

    return FileStorageService(data_dir=temp_data_dir, min_free_disk_gb=10)


@pytest.fixture
def mock_upload_file():
    """Create a mock UploadFile for testing."""
    content = b'{"prompt": "test", "completion": "response"}\n'
    file = MagicMock(spec=UploadFile)
    file.filename = "test_context.jsonl"
    file.content_type = "application/jsonl"
    file.size = len(content)
    file.read = AsyncMock(return_value=content)
    file.seek = AsyncMock()
    return file


class TestFileStorageServiceInit:
    """Tests for FileStorageService initialization."""

    def test_init_creates_data_dir(self, tmp_path):
        """Test that init creates data directory if not exists."""
        from src.services.file_storage import FileStorageService

        data_dir = tmp_path / "nonexistent"
        service = FileStorageService(data_dir=data_dir)
        assert data_dir.exists()

    def test_init_with_existing_dir(self, temp_data_dir):
        """Test that init works with existing directory."""
        from src.services.file_storage import FileStorageService

        service = FileStorageService(data_dir=temp_data_dir)
        assert service.data_dir == temp_data_dir

    def test_init_sets_min_free_disk(self, temp_data_dir):
        """Test that min_free_disk_gb is configurable."""
        from src.services.file_storage import FileStorageService

        service = FileStorageService(data_dir=temp_data_dir, min_free_disk_gb=20)
        assert service.min_free_disk_gb == 20


class TestSaveFile:
    """Tests for save_file method."""

    @pytest.mark.asyncio
    async def test_save_file_returns_file_info(
        self, storage_service, mock_upload_file
    ):
        """Test that save_file returns a FileInfo object."""
        from src.api.schemas import FileInfo

        result = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        assert isinstance(result, FileInfo)

    @pytest.mark.asyncio
    async def test_save_file_generates_file_id_with_prefix(
        self, storage_service, mock_upload_file
    ):
        """Test that generated file_id has file_ prefix."""
        result = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        assert result.file_id.startswith("file_")

    @pytest.mark.asyncio
    async def test_save_file_stores_correct_filename(
        self, storage_service, mock_upload_file
    ):
        """Test that original filename is preserved."""
        result = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        assert result.filename == "test_context.jsonl"

    @pytest.mark.asyncio
    async def test_save_file_stores_purpose(
        self, storage_service, mock_upload_file
    ):
        """Test that purpose is stored correctly."""
        result = await storage_service.save_file(
            file=mock_upload_file, purpose="batch", tags=[], ttl_hours=None
        )

        assert result.purpose == "batch"

    @pytest.mark.asyncio
    async def test_save_file_stores_tags(
        self, storage_service, mock_upload_file
    ):
        """Test that tags are stored correctly."""
        result = await storage_service.save_file(
            file=mock_upload_file,
            purpose="inference",
            tags=["training", "v2"],
            ttl_hours=None,
        )

        assert result.tags == ["training", "v2"]

    @pytest.mark.asyncio
    async def test_save_file_creates_directory_structure(
        self, storage_service, mock_upload_file, temp_data_dir
    ):
        """Test that save_file creates proper directory structure."""
        result = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        file_dir = temp_data_dir / result.file_id
        assert file_dir.exists()
        assert (file_dir / "metadata.json").exists()
        assert (file_dir / "test_context.jsonl").exists()

    @pytest.mark.asyncio
    async def test_save_file_writes_content(
        self, storage_service, mock_upload_file, temp_data_dir
    ):
        """Test that file content is written correctly."""
        result = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        file_path = temp_data_dir / result.file_id / "test_context.jsonl"
        content = file_path.read_bytes()
        assert content == b'{"prompt": "test", "completion": "response"}\n'

    @pytest.mark.asyncio
    async def test_save_file_writes_metadata(
        self, storage_service, mock_upload_file, temp_data_dir
    ):
        """Test that metadata.json is written correctly."""
        result = await storage_service.save_file(
            file=mock_upload_file,
            purpose="inference",
            tags=["test"],
            ttl_hours=24,
        )

        metadata_path = temp_data_dir / result.file_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["file_id"] == result.file_id
        assert metadata["filename"] == "test_context.jsonl"
        assert metadata["purpose"] == "inference"
        assert metadata["tags"] == ["test"]
        assert "created_at" in metadata

    @pytest.mark.asyncio
    async def test_save_file_stores_size_bytes(
        self, storage_service, mock_upload_file
    ):
        """Test that size_bytes is captured correctly."""
        result = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        assert result.size_bytes > 0

    @pytest.mark.asyncio
    async def test_save_file_stores_created_at(
        self, storage_service, mock_upload_file
    ):
        """Test that created_at timestamp is stored."""
        result = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        assert result.created_at is not None
        # Should be ISO format
        datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))

    @pytest.mark.asyncio
    async def test_save_file_checks_disk_space(
        self, storage_service, mock_upload_file
    ):
        """Test that disk space is checked before saving."""
        with patch.object(
            storage_service, "check_disk_space", return_value=False
        ) as mock_check:
            from src.services.file_storage import InsufficientDiskSpaceError

            with pytest.raises(InsufficientDiskSpaceError):
                await storage_service.save_file(
                    file=mock_upload_file,
                    purpose="inference",
                    tags=[],
                    ttl_hours=None,
                )


class TestGetFile:
    """Tests for get_file method."""

    @pytest.mark.asyncio
    async def test_get_file_returns_path_and_info(
        self, storage_service, mock_upload_file
    ):
        """Test that get_file returns file path and FileInfo."""
        saved = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        path, info = await storage_service.get_file(saved.file_id)

        assert isinstance(path, Path)
        assert path.exists()
        assert info.file_id == saved.file_id

    @pytest.mark.asyncio
    async def test_get_file_not_found(self, storage_service):
        """Test that get_file raises for nonexistent file."""
        from src.services.file_storage import FileNotFoundError

        with pytest.raises(FileNotFoundError):
            await storage_service.get_file("file_nonexistent")

    @pytest.mark.asyncio
    async def test_get_file_returns_correct_path(
        self, storage_service, mock_upload_file, temp_data_dir
    ):
        """Test that get_file returns path to actual file content."""
        saved = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        path, _ = await storage_service.get_file(saved.file_id)

        expected_path = temp_data_dir / saved.file_id / "test_context.jsonl"
        assert path == expected_path


class TestListFiles:
    """Tests for list_files method."""

    @pytest.mark.asyncio
    async def test_list_files_empty(self, storage_service):
        """Test list_files returns empty list when no files."""
        result = await storage_service.list_files(
            purpose=None, tag=None, limit=10, offset=0
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_list_files_returns_all(
        self, storage_service, mock_upload_file
    ):
        """Test list_files returns all stored files."""
        await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        # Reset mock for second file
        mock_upload_file.read.reset_mock()
        await storage_service.save_file(
            file=mock_upload_file, purpose="batch", tags=[], ttl_hours=None
        )

        result = await storage_service.list_files(
            purpose=None, tag=None, limit=10, offset=0
        )

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_files_filter_by_purpose(
        self, storage_service, mock_upload_file
    ):
        """Test list_files filters by purpose."""
        await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )
        mock_upload_file.read.reset_mock()
        await storage_service.save_file(
            file=mock_upload_file, purpose="batch", tags=[], ttl_hours=None
        )

        result = await storage_service.list_files(
            purpose="inference", tag=None, limit=10, offset=0
        )

        assert len(result) == 1
        assert result[0].purpose == "inference"

    @pytest.mark.asyncio
    async def test_list_files_filter_by_tag(
        self, storage_service, mock_upload_file
    ):
        """Test list_files filters by tag."""
        await storage_service.save_file(
            file=mock_upload_file,
            purpose="inference",
            tags=["training"],
            ttl_hours=None,
        )
        mock_upload_file.read.reset_mock()
        await storage_service.save_file(
            file=mock_upload_file,
            purpose="inference",
            tags=["evaluation"],
            ttl_hours=None,
        )

        result = await storage_service.list_files(
            purpose=None, tag="training", limit=10, offset=0
        )

        assert len(result) == 1
        assert "training" in result[0].tags

    @pytest.mark.asyncio
    async def test_list_files_pagination_limit(
        self, storage_service, mock_upload_file
    ):
        """Test list_files respects limit parameter."""
        for i in range(5):
            mock_upload_file.read.reset_mock()
            await storage_service.save_file(
                file=mock_upload_file,
                purpose="inference",
                tags=[],
                ttl_hours=None,
            )

        result = await storage_service.list_files(
            purpose=None, tag=None, limit=3, offset=0
        )

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_files_pagination_offset(
        self, storage_service, mock_upload_file
    ):
        """Test list_files respects offset parameter."""
        for i in range(5):
            mock_upload_file.read.reset_mock()
            await storage_service.save_file(
                file=mock_upload_file,
                purpose="inference",
                tags=[],
                ttl_hours=None,
            )

        result = await storage_service.list_files(
            purpose=None, tag=None, limit=10, offset=3
        )

        assert len(result) == 2


class TestDeleteFile:
    """Tests for delete_file method."""

    @pytest.mark.asyncio
    async def test_delete_file_returns_true(
        self, storage_service, mock_upload_file
    ):
        """Test delete_file returns True on success."""
        saved = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        result = await storage_service.delete_file(saved.file_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_file_removes_directory(
        self, storage_service, mock_upload_file, temp_data_dir
    ):
        """Test delete_file removes the file directory."""
        saved = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )

        file_dir = temp_data_dir / saved.file_id
        assert file_dir.exists()

        await storage_service.delete_file(saved.file_id)

        assert not file_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, storage_service):
        """Test delete_file raises for nonexistent file."""
        from src.services.file_storage import FileNotFoundError

        with pytest.raises(FileNotFoundError):
            await storage_service.delete_file("file_nonexistent")

    @pytest.mark.asyncio
    async def test_delete_file_cannot_get_after_delete(
        self, storage_service, mock_upload_file
    ):
        """Test that deleted file cannot be retrieved."""
        from src.services.file_storage import FileNotFoundError

        saved = await storage_service.save_file(
            file=mock_upload_file, purpose="inference", tags=[], ttl_hours=None
        )
        await storage_service.delete_file(saved.file_id)

        with pytest.raises(FileNotFoundError):
            await storage_service.get_file(saved.file_id)


class TestCheckDiskSpace:
    """Tests for check_disk_space method."""

    @pytest.mark.asyncio
    async def test_check_disk_space_returns_bool(self, storage_service):
        """Test check_disk_space returns a boolean."""
        result = await storage_service.check_disk_space(required_bytes=1024)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_check_disk_space_small_file(self, storage_service):
        """Test check_disk_space returns True for small files."""
        # Small file should pass
        result = await storage_service.check_disk_space(required_bytes=1024)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_disk_space_respects_min_free(self, temp_data_dir):
        """Test check_disk_space respects min_free_disk_gb setting."""
        from src.services.file_storage import FileStorageService

        # Create service with huge min free requirement
        service = FileStorageService(
            data_dir=temp_data_dir, min_free_disk_gb=10000000
        )

        # Should fail since we need more than available
        result = await service.check_disk_space(required_bytes=1024)
        assert result is False
