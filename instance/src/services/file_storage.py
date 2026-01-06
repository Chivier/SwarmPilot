"""File storage service for managing inference files.

Handles file persistence, metadata management, and disk space monitoring.
"""

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from fastapi import UploadFile

from src.api.schemas import FileInfo


class FileNotFoundError(Exception):
    """Raised when a requested file does not exist."""

    def __init__(self, file_id: str):
        """Initialize with file ID.

        Args:
            file_id: The ID of the file that was not found.
        """
        self.file_id = file_id
        super().__init__(f"File not found: {file_id}")


class InsufficientDiskSpaceError(Exception):
    """Raised when there is not enough disk space for an operation."""

    def __init__(self, required_bytes: int, available_bytes: int):
        """Initialize with disk space details.

        Args:
            required_bytes: Number of bytes needed.
            available_bytes: Number of bytes available.
        """
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        super().__init__(
            f"Insufficient disk space. Required: {required_bytes}, "
            f"Available: {available_bytes}"
        )


class FileStorageService:
    """Service for managing file storage.

    Handles saving, retrieving, listing, and deleting files with associated
    metadata. Files are stored in a directory structure:

        data_dir/
        ├── file_abc123/
        │   ├── metadata.json
        │   └── original_filename.jsonl
        └── file_def456/
            ├── metadata.json
            └── another_file.jsonl

    Attributes:
        data_dir: Root directory for file storage.
        min_free_disk_gb: Minimum free disk space to maintain (in GB).
    """

    def __init__(self, data_dir: Path, min_free_disk_gb: int = 10):
        """Initialize the file storage service.

        Args:
            data_dir: Root directory for file storage. Created if not exists.
            min_free_disk_gb: Minimum free disk space to maintain (in GB).
        """
        self.data_dir = Path(data_dir)
        self.min_free_disk_gb = min_free_disk_gb

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _generate_file_id(self) -> str:
        """Generate a unique file ID with prefix.

        Returns:
            A unique file ID in the format 'file_<uuid>'.
        """
        return f"file_{uuid.uuid4().hex[:12]}"

    def _get_file_dir(self, file_id: str) -> Path:
        """Get the directory path for a file ID.

        Args:
            file_id: The file identifier.

        Returns:
            Path to the file's directory.
        """
        return self.data_dir / file_id

    def _load_metadata(self, file_id: str) -> FileInfo:
        """Load metadata for a file.

        Args:
            file_id: The file identifier.

        Returns:
            FileInfo object with file metadata.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        file_dir = self._get_file_dir(file_id)
        metadata_path = file_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(file_id)

        with open(metadata_path) as f:
            data = json.load(f)

        return FileInfo(**data)

    def _save_metadata(self, file_dir: Path, info: FileInfo) -> None:
        """Save metadata for a file.

        Args:
            file_dir: Directory where metadata should be saved.
            info: FileInfo object to persist.
        """
        metadata_path = file_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(info.model_dump(), f, indent=2)

    async def check_disk_space(self, required_bytes: int) -> bool:
        """Check if there is sufficient disk space.

        Args:
            required_bytes: Number of bytes needed for the operation.

        Returns:
            True if sufficient space available, False otherwise.
        """
        disk_usage = psutil.disk_usage(str(self.data_dir))
        min_free_bytes = self.min_free_disk_gb * 1024**3

        # Need space for the file plus minimum free space
        available = disk_usage.free - min_free_bytes
        return available >= required_bytes

    async def save_file(
        self,
        file: "UploadFile",
        purpose: str,
        tags: list[str],
        ttl_hours: int | None,
    ) -> FileInfo:
        """Save an uploaded file with metadata.

        Args:
            file: The uploaded file object.
            purpose: Purpose of the file (e.g., 'inference', 'batch').
            tags: List of tags for categorization.
            ttl_hours: Optional time-to-live in hours.

        Returns:
            FileInfo with file details and generated ID.

        Raises:
            InsufficientDiskSpaceError: If not enough disk space.
        """
        # Read file content
        content = await file.read()
        size_bytes = len(content)

        # Check disk space
        if not await self.check_disk_space(size_bytes):
            disk_usage = psutil.disk_usage(str(self.data_dir))
            raise InsufficientDiskSpaceError(
                required_bytes=size_bytes, available_bytes=disk_usage.free
            )

        # Generate file ID and create directory
        file_id = self._generate_file_id()
        file_dir = self._get_file_dir(file_id)
        file_dir.mkdir(parents=True, exist_ok=True)

        # Save file content
        file_path = file_dir / file.filename
        file_path.write_bytes(content)

        # Create FileInfo
        info = FileInfo(
            file_id=file_id,
            filename=file.filename,
            purpose=purpose,
            size_bytes=size_bytes,
            tags=tags,
            created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

        # Save metadata
        self._save_metadata(file_dir, info)

        return info

    async def get_file(self, file_id: str) -> tuple[Path, FileInfo]:
        """Retrieve a file by ID.

        Args:
            file_id: The file identifier.

        Returns:
            Tuple of (file_path, file_info).

        Raises:
            FileNotFoundError: If file does not exist.
        """
        info = self._load_metadata(file_id)
        file_dir = self._get_file_dir(file_id)
        file_path = file_dir / info.filename

        if not file_path.exists():
            raise FileNotFoundError(file_id)

        return file_path, info

    async def list_files(
        self,
        purpose: str | None,
        tag: str | None,
        limit: int,
        offset: int,
    ) -> list[FileInfo]:
        """List files with optional filtering and pagination.

        Args:
            purpose: Filter by purpose (optional).
            tag: Filter by tag (optional).
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of FileInfo objects matching the criteria.
        """
        files = []

        # Iterate through file directories
        for file_dir in self.data_dir.iterdir():
            if not file_dir.is_dir():
                continue

            metadata_path = file_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                info = FileInfo(**data)

                # Apply filters
                if purpose and info.purpose != purpose:
                    continue
                if tag and tag not in info.tags:
                    continue

                files.append(info)
            except (json.JSONDecodeError, ValueError):
                # Skip invalid metadata files
                continue

        # Sort by created_at descending
        files.sort(key=lambda f: f.created_at, reverse=True)

        # Apply pagination
        return files[offset : offset + limit]

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file by ID.

        Args:
            file_id: The file identifier.

        Returns:
            True if file was deleted.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        file_dir = self._get_file_dir(file_id)

        if not file_dir.exists():
            raise FileNotFoundError(file_id)

        # Remove entire directory
        shutil.rmtree(file_dir)
        return True
