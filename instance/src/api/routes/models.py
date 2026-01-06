"""Models API routes.

Provides endpoints for model management including listing, details, and lifecycle.
"""

import shutil
import tarfile
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from src.api.schemas import (
    ModelConfigRequest,
    ModelConfigResponse,
    ModelDeleteResponse,
    ModelDetailResponse,
    ModelListItem,
    ModelListResponse,
    ModelPullRequest,
    ModelPullResponse,
    ModelStartRequest,
    ModelStartResponse,
    ModelStopRequest,
    ModelStopResponse,
    ModelSwitchRequest,
    ModelSwitchResponse,
    ModelUploadResponse,
    OpenAIModel,
    OpenAIModelListResponse,
)
from src.services.inference_manager import (
    InferenceManagerService,
    InvalidModelStateError,
    ModelNotFoundError as InferenceModelNotFoundError,
)
from src.services.model_storage import ModelNotFoundError, ModelStorageService
from src.services.task_tracking import TaskTrackingService

router = APIRouter(tags=["models"])

# Service instances (set during app startup)
_model_storage_service: ModelStorageService | None = None
_task_tracking_service: TaskTrackingService | None = None
_inference_manager_service: InferenceManagerService | None = None


def set_model_storage_service(service: ModelStorageService) -> None:
    """Set the model storage service instance.

    Args:
        service: ModelStorageService instance to use.
    """
    global _model_storage_service
    _model_storage_service = service


def get_model_storage_service() -> ModelStorageService:
    """Get the model storage service instance.

    Returns:
        The ModelStorageService instance.

    Raises:
        RuntimeError: If storage service not initialized.
    """
    if _model_storage_service is None:
        raise RuntimeError("Model storage service not initialized")
    return _model_storage_service


def set_task_tracking_service(service: TaskTrackingService) -> None:
    """Set the task tracking service instance.

    Args:
        service: TaskTrackingService instance to use.
    """
    global _task_tracking_service
    _task_tracking_service = service


def get_task_tracking_service() -> TaskTrackingService:
    """Get the task tracking service instance.

    Returns:
        The TaskTrackingService instance.

    Raises:
        RuntimeError: If task tracking service not initialized.
    """
    if _task_tracking_service is None:
        raise RuntimeError("Task tracking service not initialized")
    return _task_tracking_service


def set_inference_manager_service(service: InferenceManagerService) -> None:
    """Set the inference manager service instance.

    Args:
        service: InferenceManagerService instance to use.
    """
    global _inference_manager_service
    _inference_manager_service = service


def get_inference_manager_service() -> InferenceManagerService:
    """Get the inference manager service instance.

    Returns:
        The InferenceManagerService instance.

    Raises:
        RuntimeError: If inference manager service not initialized.
    """
    if _inference_manager_service is None:
        raise RuntimeError("Inference manager service not initialized")
    return _inference_manager_service


@router.post("/pull", response_model=ModelPullResponse, status_code=202)
async def pull_model(request: ModelPullRequest) -> ModelPullResponse:
    """Pull a model from HuggingFace.

    Creates a model entry and starts a background task to download the model.

    Args:
        request: Model pull request with name, type, and source.

    Returns:
        Model pull response with model_id, task_id, and status.
    """
    storage = get_model_storage_service()
    task_service = get_task_tracking_service()

    # Create model entry
    source_dict = {
        "type": "huggingface",
        "repo": request.source.repo,
        "revision": request.source.revision,
        "endpoint": request.source.endpoint,
    }

    model_id = await storage.create_model_entry(
        name=request.name,
        model_type=request.type,
        source=source_dict,
    )

    # Create task for background download
    task_id = await task_service.create_task(
        model_id=model_id,
        model_type=request.type,
        operation="pull",
    )

    # Update task status to pulling
    await task_service.update_task_status(task_id, "pulling")

    return ModelPullResponse(
        model_id=model_id,
        task_id=task_id,
        name=request.name,
        type=request.type,
        status="pulling",
        message=f"Started pulling model from {request.source.repo}",
    )


# Valid archive extensions for model upload
ARCHIVE_EXTENSIONS = {".tar.gz", ".tgz", ".zip"}
SINGLE_FILE_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}


def _is_archive(filename: str) -> bool:
    """Check if filename is a supported archive."""
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def _is_valid_single_file(filename: str) -> bool:
    """Check if filename is a valid single model file."""
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in SINGLE_FILE_EXTENSIONS)


def _extract_archive(
    file_content: bytes,
    filename: str,
    dest_dir: Path,
) -> None:
    """Extract an archive to destination directory.

    Args:
        file_content: Archive content as bytes.
        filename: Original filename (to determine archive type).
        dest_dir: Destination directory.

    Raises:
        ValueError: If archive is corrupted or unsupported.
    """
    lower = filename.lower()

    if lower.endswith(".zip"):
        try:
            with zipfile.ZipFile(BytesIO(file_content), "r") as zf:
                zf.extractall(dest_dir)
        except zipfile.BadZipFile:
            raise ValueError("Corrupted or invalid ZIP archive")

    elif lower.endswith((".tar.gz", ".tgz")):
        try:
            with tarfile.open(fileobj=BytesIO(file_content), mode="r:gz") as tf:
                tf.extractall(dest_dir, filter="data")
        except tarfile.TarError:
            raise ValueError("Corrupted or invalid tar.gz archive")

    else:
        raise ValueError(f"Unsupported archive format: {filename}")


@router.post("/upload", response_model=ModelUploadResponse)
async def upload_model(
    file: Annotated[UploadFile, File(...)],
    name: Annotated[str, Form(...)],
    type: Annotated[str, Form(...)],
) -> ModelUploadResponse:
    """Upload a model from a local file.

    Accepts single model files (.safetensors, .bin, etc.) or archives
    (.tar.gz, .zip) containing model weights and configuration.

    Args:
        file: Model file or archive.
        name: Human-readable model name.
        type: Model type (e.g., 'llm').

    Returns:
        Model upload response with model_id and status.

    Raises:
        HTTPException: 400 if file format is invalid or archive is corrupted.
    """
    storage = get_model_storage_service()

    filename = file.filename or "unknown"

    # Validate file type
    if not _is_archive(filename) and not _is_valid_single_file(filename):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file_type",
                "message": (
                    f"Unsupported file type: {filename}. "
                    f"Supported: {', '.join(ARCHIVE_EXTENSIONS | SINGLE_FILE_EXTENSIONS)}"
                ),
            },
        )

    # Create model entry with upload source
    source_dict = {
        "type": "upload",
        "filename": filename,
    }

    model_id = await storage.create_model_entry(
        name=name,
        model_type=type,
        source=source_dict,
    )

    # Get model directory
    model_dir = await storage.get_model_path(model_id)

    try:
        # Read file content
        file_content = await file.read()

        if _is_archive(filename):
            # Extract archive
            try:
                _extract_archive(file_content, filename, model_dir)
            except ValueError as e:
                # Cleanup on error
                shutil.rmtree(model_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_archive",
                        "message": str(e),
                    },
                )
        else:
            # Save single file
            dest_path = model_dir / filename
            dest_path.write_bytes(file_content)

        # Update model size and status
        await storage.update_model_size(model_id)
        await storage.update_model_status(model_id, "ready")

        return ModelUploadResponse(
            model_id=model_id,
            name=name,
            type=type,
            status="ready",
            message=f"Model uploaded successfully from {filename}",
        )

    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on unexpected error
        shutil.rmtree(model_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "upload_failed",
                "message": str(e),
            },
        )


@router.post("/switch", response_model=ModelSwitchResponse)
async def switch_model(request: ModelSwitchRequest) -> ModelSwitchResponse:
    """Switch to a different model.

    Stops the current model (if any) and starts the target model.

    Args:
        request: Model switch request with target model and configuration.

    Returns:
        Model switch response.

    Raises:
        HTTPException: 404 if target not found, 409 if target not ready.
    """
    inference = get_inference_manager_service()

    try:
        config_dict = None
        if request.config:
            config_dict = {
                k: v for k, v in request.config.model_dump().items() if v is not None
            }

        result = await inference.switch_model(
            target_model_id=request.target_model_id,
            gpu_ids=request.gpu_ids,
            graceful_timeout_seconds=request.graceful_timeout_seconds,
            config=config_dict,
        )

        return ModelSwitchResponse(
            previous_model_id=result["previous_model_id"],
            current_model_id=result["current_model_id"],
            status=result["status"],
            message=result["message"],
        )

    except InferenceModelNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": str(e),
            },
        )
    except InvalidModelStateError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "invalid_model_state",
                "message": str(e),
            },
        )


@router.get("", response_model=OpenAIModelListResponse)
async def list_models_openai() -> OpenAIModelListResponse:
    """List models in OpenAI-compatible format.

    Only returns models with status 'ready' or 'running'.

    Returns:
        OpenAI-compatible model list response.
    """
    storage = get_model_storage_service()

    # Get all models
    all_models = await storage.list_models(model_type=None, status=None)

    # Filter to only ready/running models
    available_models = [
        m for m in all_models if m.status in ("ready", "running")
    ]

    # Convert to OpenAI format
    openai_models = []
    for model in available_models:
        # Parse created_at to Unix timestamp
        created_dt = datetime.fromisoformat(
            model.created_at.replace("Z", "+00:00")
        )
        created_ts = int(created_dt.timestamp())

        openai_models.append(
            OpenAIModel(
                id=model.name,
                created=created_ts,
                owned_by="swarmx",
            )
        )

    return OpenAIModelListResponse(data=openai_models)


@router.get("/list", response_model=ModelListResponse)
async def list_models_detailed(
    type: Annotated[str | None, Query(alias="type")] = None,
    status: Annotated[str | None, Query()] = None,
) -> ModelListResponse:
    """List models with detailed information.

    Args:
        type: Filter by model type.
        status: Filter by status.

    Returns:
        Detailed model list response.
    """
    storage = get_model_storage_service()

    models = await storage.list_models(model_type=type, status=status)

    # Convert to ModelListItem format
    model_items = []
    total_size = 0
    for model in models:
        model_items.append(
            ModelListItem(
                model_id=model.model_id,
                name=model.name,
                type=model.type,
                status=model.status,
                source=model.source,
                size_bytes=model.size_bytes,
                created_at=model.created_at,
                has_default_config=model.default_config is not None,
            )
        )
        total_size += model.size_bytes

    return ModelListResponse(
        models=model_items,
        total_size_bytes=total_size,
    )


@router.put("/{model_id}/config", response_model=ModelConfigResponse)
async def set_model_config(
    model_id: str,
    config: ModelConfigRequest,
) -> ModelConfigResponse:
    """Set default configuration for a model.

    Args:
        model_id: The model identifier.
        config: Configuration to save.

    Returns:
        Model config response with saved configuration.

    Raises:
        HTTPException: 404 if model not found.
    """
    storage = get_model_storage_service()

    # Verify model exists
    model = await storage.get_model(model_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model not found: {model_id}",
            },
        )

    # Convert config to dict, filtering None values
    config_dict = {k: v for k, v in config.model_dump().items() if v is not None}

    # Save config
    await storage.save_default_config(model_id, config_dict)

    return ModelConfigResponse(
        model_id=model_id,
        default_config=config_dict,
        message="Default configuration saved",
    )


@router.delete("/{model_id}/config")
async def delete_model_config(model_id: str) -> dict:
    """Delete default configuration for a model.

    Args:
        model_id: The model identifier.

    Returns:
        Success message.

    Raises:
        HTTPException: 404 if model not found.
    """
    storage = get_model_storage_service()

    # Verify model exists
    model = await storage.get_model(model_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model not found: {model_id}",
            },
        )

    await storage.delete_default_config(model_id)

    return {"message": "Default configuration deleted", "model_id": model_id}


@router.post("/{model_id}/start", response_model=ModelStartResponse)
async def start_model(
    model_id: str,
    request: ModelStartRequest,
) -> ModelStartResponse:
    """Start a model on the inference server.

    Args:
        model_id: The model identifier.
        request: Start request with GPU IDs and optional configuration.

    Returns:
        Model start response.

    Raises:
        HTTPException: 404 if model not found, 409 if model not ready.
    """
    inference = get_inference_manager_service()

    try:
        config_dict = None
        if request.config:
            config_dict = {
                k: v for k, v in request.config.model_dump().items() if v is not None
            }

        result = await inference.start_model(
            model_id=model_id,
            gpu_ids=request.gpu_ids,
            config=config_dict,
        )

        return ModelStartResponse(
            model_id=result["model_id"],
            type=result["type"],
            status=result["status"],
            message=result["message"],
        )

    except InferenceModelNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model not found: {model_id}",
            },
        )
    except InvalidModelStateError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "invalid_model_state",
                "message": str(e),
            },
        )


@router.post("/{model_id}/stop", response_model=ModelStopResponse)
async def stop_model(
    model_id: str,
    request: ModelStopRequest,
) -> ModelStopResponse:
    """Stop a running model.

    Args:
        model_id: The model identifier.
        request: Stop request with optional force flag.

    Returns:
        Model stop response.

    Raises:
        HTTPException: 404 if model not found, 409 if model not running.
    """
    inference = get_inference_manager_service()

    try:
        result = await inference.stop_model(
            model_id=model_id,
            force=request.force,
        )

        return ModelStopResponse(
            model_id=result["model_id"],
            status=result["status"],
            message=result["message"],
        )

    except InferenceModelNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model not found: {model_id}",
            },
        )
    except InvalidModelStateError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "invalid_model_state",
                "message": str(e),
            },
        )


@router.delete("/{model_id}", response_model=ModelDeleteResponse)
async def delete_model(model_id: str) -> ModelDeleteResponse:
    """Delete a model.

    Args:
        model_id: The model identifier.

    Returns:
        Model delete response with freed disk space.

    Raises:
        HTTPException: 404 if model not found, 409 if model is in use.
    """
    storage = get_model_storage_service()

    # Verify model exists and check status
    model = await storage.get_model(model_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model not found: {model_id}",
            },
        )

    # Prevent deletion of running models
    if model.status in ("running", "loading"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "model_in_use",
                "message": f"Cannot delete model in '{model.status}' state. "
                "Stop the model first.",
            },
        )

    # Delete the model
    try:
        freed_bytes = await storage.delete_model(model_id)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model not found: {model_id}",
            },
        )

    return ModelDeleteResponse(
        deleted=True,
        model_id=model_id,
        disk_freed_bytes=freed_bytes,
    )


@router.get("/{model_id}", response_model=ModelDetailResponse)
async def get_model(model_id: str) -> ModelDetailResponse:
    """Get detailed information about a model.

    Args:
        model_id: The model identifier.

    Returns:
        Model detail response.

    Raises:
        HTTPException: 404 if model not found.
    """
    storage = get_model_storage_service()

    model = await storage.get_model(model_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model not found: {model_id}",
            },
        )

    return model
