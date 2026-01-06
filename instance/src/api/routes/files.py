"""File API routes.

Provides endpoints for file upload, download, listing, and deletion.
"""

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from src.api.schemas import (
    FileDeleteResponse,
    FileListResponse,
    FileUploadResponse,
)
from src.services.file_storage import (
    FileNotFoundError,
    FileStorageService,
    InsufficientDiskSpaceError,
)

router = APIRouter(tags=["files"])

# Storage service instance (set during app startup)
_storage_service: FileStorageService | None = None


def set_storage_service(service: FileStorageService) -> None:
    """Set the storage service instance.

    Args:
        service: FileStorageService instance to use.
    """
    global _storage_service
    _storage_service = service


def get_storage_service() -> FileStorageService:
    """Get the storage service instance.

    Returns:
        The FileStorageService instance.

    Raises:
        RuntimeError: If storage service not initialized.
    """
    if _storage_service is None:
        raise RuntimeError("Storage service not initialized")
    return _storage_service


@router.post("/upload", response_model=FileUploadResponse, status_code=201)
async def upload_file(
    file: Annotated[UploadFile, File(...)],
    purpose: Annotated[str, Form(...)],
    tags: Annotated[str | None, Form()] = None,
    ttl_hours: Annotated[int | None, Form()] = None,
) -> FileUploadResponse:
    """Upload a file for inference workloads.

    Args:
        file: The file to upload.
        purpose: Purpose of the file ('inference' or 'batch').
        tags: Optional comma-separated tags.
        ttl_hours: Optional auto-delete timer in hours.

    Returns:
        FileUploadResponse with file details.

    Raises:
        HTTPException: 507 if insufficient disk space.
    """
    storage = get_storage_service()

    # Parse tags from comma-separated string
    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        info = await storage.save_file(
            file=file,
            purpose=purpose,
            tags=tag_list,
            ttl_hours=ttl_hours,
        )

        return FileUploadResponse(
            file_id=info.file_id,
            filename=info.filename,
            purpose=info.purpose,
            size_bytes=info.size_bytes,
            created_at=info.created_at,
        )
    except InsufficientDiskSpaceError as e:
        raise HTTPException(
            status_code=507,
            detail={
                "error": "insufficient_disk_space",
                "message": str(e),
                "details": {
                    "required_bytes": e.required_bytes,
                    "available_bytes": e.available_bytes,
                },
            },
        )


@router.get("/{file_id}")
async def download_file(file_id: str) -> FileResponse:
    """Download a file by ID.

    Args:
        file_id: The file identifier.

    Returns:
        FileResponse streaming the file content.

    Raises:
        HTTPException: 404 if file not found.
    """
    storage = get_storage_service()

    try:
        path, info = await storage.get_file(file_id)

        return FileResponse(
            path=path,
            filename=info.filename,
            media_type="application/octet-stream",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "file_not_found",
                "message": f"File not found: {file_id}",
            },
        )


@router.get("", response_model=FileListResponse)
async def list_files(
    purpose: Annotated[str | None, Query()] = None,
    tag: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FileListResponse:
    """List files with optional filtering.

    Args:
        purpose: Filter by purpose.
        tag: Filter by tag.
        limit: Maximum number of results (default 100, max 1000).
        offset: Pagination offset.

    Returns:
        FileListResponse with files and total count.
    """
    storage = get_storage_service()

    # Get all files matching filters (for total count)
    all_files = await storage.list_files(
        purpose=purpose,
        tag=tag,
        limit=10000,  # Large limit to get all
        offset=0,
    )
    total = len(all_files)

    # Get paginated results
    files = await storage.list_files(
        purpose=purpose,
        tag=tag,
        limit=limit,
        offset=offset,
    )

    return FileListResponse(
        files=files,
        total=total,
    )


@router.delete("/{file_id}", response_model=FileDeleteResponse)
async def delete_file(file_id: str) -> FileDeleteResponse:
    """Delete a file by ID.

    Args:
        file_id: The file identifier.

    Returns:
        FileDeleteResponse confirming deletion.

    Raises:
        HTTPException: 404 if file not found.
    """
    storage = get_storage_service()

    try:
        await storage.delete_file(file_id)
        return FileDeleteResponse(deleted=True, file_id=file_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "file_not_found",
                "message": f"File not found: {file_id}",
            },
        )
