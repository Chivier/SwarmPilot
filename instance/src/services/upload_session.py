"""Upload Session service for managing resumable uploads.

Handles tus protocol upload sessions for large file uploads.
"""

import contextlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.services.model_storage import ModelStorageService


class UploadSessionNotFoundError(Exception):
    """Raised when an upload session does not exist."""

    def __init__(self, upload_id: str):
        """Initialize with upload ID.

        Args:
            upload_id: The ID of the upload session that was not found.
        """
        self.upload_id = upload_id
        super().__init__(f"Upload session not found: {upload_id}")


class UploadOffsetMismatchError(Exception):
    """Raised when upload offset doesn't match server state."""

    def __init__(self, expected: int, actual: int):
        """Initialize with offset details.

        Args:
            expected: The expected offset.
            actual: The actual offset provided.
        """
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Upload offset mismatch: expected {expected}, got {actual}"
        )


@dataclass
class UploadSession:
    """State for an upload session.

    Attributes:
        upload_id: Unique session identifier.
        model_id: Associated model identifier.
        name: Model name.
        model_type: Model type.
        upload_length: Total file size in bytes.
        upload_offset: Current upload position.
        file_path: Path to partial upload file.
    """

    upload_id: str
    model_id: str
    name: str
    model_type: str
    upload_length: int
    upload_offset: int
    file_path: Path


class UploadSessionService:
    """Service for managing resumable upload sessions.

    Implements storage and tracking for tus protocol uploads.

    Attributes:
        upload_dir: Directory for storing partial uploads.
        model_storage: Model storage service for finalizing uploads.
    """

    def __init__(
        self,
        upload_dir: Path,
        model_storage: ModelStorageService,
    ):
        """Initialize the upload session service.

        Args:
            upload_dir: Directory for storing partial uploads.
            model_storage: Model storage service instance.
        """
        self.upload_dir = Path(upload_dir)
        self.model_storage = model_storage
        self._sessions: dict[str, UploadSession] = {}

        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _generate_upload_id(self) -> str:
        """Generate a unique upload session ID.

        Returns:
            Unique upload ID.
        """
        return f"upload_{uuid.uuid4().hex[:12]}"

    async def create_session(
        self,
        name: str,
        model_type: str,
        upload_length: int,
    ) -> UploadSession:
        """Create a new upload session.

        Args:
            name: Model name.
            model_type: Model type.
            upload_length: Total file size in bytes.

        Returns:
            Created upload session.
        """
        # Create model entry
        source_dict = {
            "type": "upload",
            "filename": f"{name}.bin",
        }
        model_id = await self.model_storage.create_model_entry(
            name=name,
            model_type=model_type,
            source=source_dict,
        )

        # Create upload session
        upload_id = self._generate_upload_id()
        file_path = self.upload_dir / f"{upload_id}.partial"

        # Create empty file
        file_path.touch()

        session = UploadSession(
            upload_id=upload_id,
            model_id=model_id,
            name=name,
            model_type=model_type,
            upload_length=upload_length,
            upload_offset=0,
            file_path=file_path,
        )

        self._sessions[upload_id] = session
        return session

    async def get_session(self, upload_id: str) -> UploadSession:
        """Get an upload session by ID.

        Args:
            upload_id: Upload session identifier.

        Returns:
            Upload session.

        Raises:
            UploadSessionNotFoundError: If session doesn't exist.
        """
        session = self._sessions.get(upload_id)
        if session is None:
            raise UploadSessionNotFoundError(upload_id)
        return session

    async def append_chunk(
        self,
        upload_id: str,
        offset: int,
        data: bytes,
    ) -> tuple[int, str | None]:
        """Append a chunk to an upload session.

        Args:
            upload_id: Upload session identifier.
            offset: Expected current offset.
            data: Chunk data to append.

        Returns:
            Tuple of (new_offset, model_id or None if not complete).

        Raises:
            UploadSessionNotFoundError: If session doesn't exist.
            UploadOffsetMismatchError: If offset doesn't match server state.
        """
        session = await self.get_session(upload_id)

        # Verify offset matches
        if offset != session.upload_offset:
            raise UploadOffsetMismatchError(session.upload_offset, offset)

        # Append data to file
        with open(session.file_path, "ab") as f:
            f.write(data)

        # Update offset
        session.upload_offset += len(data)

        # Check if upload is complete
        model_id = None
        if session.upload_offset >= session.upload_length:
            model_id = await self._finalize_upload(session)

        return session.upload_offset, model_id

    async def _finalize_upload(self, session: UploadSession) -> str:
        """Finalize a completed upload.

        Moves the uploaded file to the model directory and updates status.

        Args:
            session: Completed upload session.

        Returns:
            Model ID.
        """
        # Get model directory
        model_dir = await self.model_storage.get_model_path(session.model_id)

        # Move uploaded file to model directory
        dest_path = model_dir / f"{session.name}.bin"
        shutil.move(str(session.file_path), str(dest_path))

        # Update model size and status
        await self.model_storage.update_model_size(session.model_id)
        await self.model_storage.update_model_status(session.model_id, "ready")

        # Clean up session
        del self._sessions[session.upload_id]

        return session.model_id

    async def delete_session(self, upload_id: str) -> None:
        """Delete an upload session.

        Args:
            upload_id: Upload session identifier.

        Raises:
            UploadSessionNotFoundError: If session doesn't exist.
        """
        session = await self.get_session(upload_id)

        # Remove partial file
        if session.file_path.exists():
            session.file_path.unlink()

        # Delete model entry (may not exist yet)
        with contextlib.suppress(Exception):
            await self.model_storage.delete_model(session.model_id)

        # Remove session
        del self._sessions[upload_id]
