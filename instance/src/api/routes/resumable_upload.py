"""Resumable Upload API routes (tus protocol).

Provides endpoints for resumable file uploads following tus 1.0.0 protocol.
"""

import base64
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response

from src.services.model_storage import ModelStorageService
from src.services.upload_session import (
    UploadOffsetMismatchError,
    UploadSessionNotFoundError,
    UploadSessionService,
)

router = APIRouter(tags=["resumable-upload"])

# Supported tus version
TUS_VERSION = "1.0.0"

# Service instances (set during app startup)
_upload_session_service: UploadSessionService | None = None
_model_storage_service: ModelStorageService | None = None


def set_upload_session_service(service: UploadSessionService) -> None:
    """Set the upload session service instance.

    Args:
        service: UploadSessionService instance to use.
    """
    global _upload_session_service
    _upload_session_service = service


def get_upload_session_service() -> UploadSessionService:
    """Get the upload session service instance.

    Returns:
        The UploadSessionService instance.

    Raises:
        RuntimeError: If service not initialized.
    """
    if _upload_session_service is None:
        raise RuntimeError("Upload session service not initialized")
    return _upload_session_service


def set_model_storage_service(service: ModelStorageService) -> None:
    """Set the model storage service instance.

    Args:
        service: ModelStorageService instance to use.
    """
    global _model_storage_service
    _model_storage_service = service


def _parse_metadata(metadata_header: str) -> dict[str, str]:
    """Parse tus Upload-Metadata header.

    Args:
        metadata_header: Raw metadata header value.

    Returns:
        Dictionary of decoded metadata key-value pairs.
    """
    metadata = {}
    for item in metadata_header.split(","):
        item = item.strip()
        if " " in item:
            key, value_b64 = item.split(" ", 1)
            try:
                value = base64.b64decode(value_b64).decode("utf-8")
            except Exception:
                value = value_b64
            metadata[key] = value
    return metadata


def _tus_response(response: Response) -> Response:
    """Add standard tus headers to response.

    Args:
        response: Response object to modify.

    Returns:
        Modified response.
    """
    response.headers["Tus-Resumable"] = TUS_VERSION
    return response


@router.post("", status_code=201)
async def create_upload_session(
    request: Request,
    tus_resumable: Annotated[str | None, Header(alias="Tus-Resumable")] = None,
    upload_length: Annotated[str | None, Header(alias="Upload-Length")] = None,
    upload_metadata: Annotated[
        str | None, Header(alias="Upload-Metadata")
    ] = None,
) -> Response:
    """Create a new resumable upload session.

    Implements tus POST creation endpoint.

    Args:
        request: FastAPI request object.
        tus_resumable: Required tus protocol version header.
        upload_length: Total file size in bytes.
        upload_metadata: Base64-encoded metadata (name, type).

    Returns:
        201 Created with Location header.

    Raises:
        HTTPException: 400/412 for validation errors.
    """
    # Validate tus version
    if tus_resumable is None or tus_resumable != TUS_VERSION:
        raise HTTPException(
            status_code=412,
            detail={
                "error": "tus_version_mismatch",
                "message": f"Unsupported tus version. Required: {TUS_VERSION}",
            },
        )

    # Validate upload length
    if upload_length is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_upload_length",
                "message": "Upload-Length header is required",
            },
        )

    try:
        length = int(upload_length)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_upload_length",
                "message": "Upload-Length must be a valid integer",
            },
        )

    # Validate and parse metadata
    if upload_metadata is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_metadata",
                "message": "Upload-Metadata header is required (name, type)",
            },
        )

    metadata = _parse_metadata(upload_metadata)
    if "name" not in metadata or "type" not in metadata:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_metadata",
                "message": "Upload-Metadata must include 'name' and 'type'",
            },
        )

    # Create upload session
    service = get_upload_session_service()
    session = await service.create_session(
        name=metadata["name"],
        model_type=metadata["type"],
        upload_length=length,
    )

    # Build location URL
    base_url = str(request.base_url).rstrip("/")
    location = f"{base_url}/v1/models/upload/resumable/{session.upload_id}"

    response = Response(status_code=201)
    response.headers["Location"] = location
    response.headers["Tus-Resumable"] = TUS_VERSION
    return response


@router.patch("/{upload_id}", status_code=204)
async def upload_chunk(
    upload_id: str,
    request: Request,
    tus_resumable: Annotated[str | None, Header(alias="Tus-Resumable")] = None,
    upload_offset: Annotated[str | None, Header(alias="Upload-Offset")] = None,
) -> Response:
    """Upload a chunk of data.

    Implements tus PATCH endpoint for chunk uploads.

    Args:
        upload_id: Upload session identifier.
        request: FastAPI request object.
        tus_resumable: Required tus protocol version header.
        upload_offset: Current byte offset.

    Returns:
        204 No Content with Upload-Offset header.

    Raises:
        HTTPException: 404/409/412 for errors.
    """
    # Validate tus version
    if tus_resumable is None or tus_resumable != TUS_VERSION:
        raise HTTPException(
            status_code=412,
            detail={
                "error": "tus_version_mismatch",
                "message": f"Unsupported tus version. Required: {TUS_VERSION}",
            },
        )

    # Validate offset
    if upload_offset is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_upload_offset",
                "message": "Upload-Offset header is required",
            },
        )

    try:
        offset = int(upload_offset)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_upload_offset",
                "message": "Upload-Offset must be a valid integer",
            },
        )

    # Read chunk data
    data = await request.body()

    # Append chunk
    service = get_upload_session_service()
    try:
        new_offset, model_id = await service.append_chunk(
            upload_id=upload_id,
            offset=offset,
            data=data,
        )
    except UploadSessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "upload_not_found",
                "message": f"Upload session not found: {upload_id}",
            },
        )
    except UploadOffsetMismatchError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "offset_mismatch",
                "message": str(e),
                "expected_offset": e.expected,
            },
        )

    response = Response(status_code=204)
    response.headers["Upload-Offset"] = str(new_offset)
    response.headers["Tus-Resumable"] = TUS_VERSION

    # If upload complete, include model ID
    if model_id is not None:
        response.headers["X-Model-Id"] = model_id

    return response


@router.head("/{upload_id}")
async def check_upload_progress(upload_id: str) -> Response:
    """Check upload progress.

    Implements tus HEAD endpoint for resumption.

    Args:
        upload_id: Upload session identifier.

    Returns:
        200 OK with Upload-Offset and Upload-Length headers.

    Raises:
        HTTPException: 404 if upload not found.
    """
    service = get_upload_session_service()
    try:
        session = await service.get_session(upload_id)
    except UploadSessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "upload_not_found",
                "message": f"Upload session not found: {upload_id}",
            },
        )

    response = Response(status_code=200)
    response.headers["Upload-Offset"] = str(session.upload_offset)
    response.headers["Upload-Length"] = str(session.upload_length)
    response.headers["Tus-Resumable"] = TUS_VERSION
    return response
