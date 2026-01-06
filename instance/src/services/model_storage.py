"""Model storage service for managing model weights.

Handles model persistence, metadata management, and configuration storage.
"""

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.api.schemas import ModelDetailResponse, ModelSource, ModelStatus


class ModelNotFoundError(Exception):
    """Raised when a requested model does not exist."""

    def __init__(self, model_id: str):
        """Initialize with model ID.

        Args:
            model_id: The ID of the model that was not found.
        """
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class ModelStorageService:
    """Service for managing model weight storage.

    Handles creating, retrieving, listing, and deleting models with associated
    metadata and configuration. Models are stored in a directory structure:

        data_dir/
        ├── model_abc123/
        │   ├── metadata.json       # name, type, source, status, size
        │   ├── default_config.json # optional saved config
        │   └── *.safetensors       # weight files
        └── model_def456/
            ├── metadata.json
            └── ...

    Attributes:
        data_dir: Root directory for model storage.
        min_free_disk_gb: Minimum free disk space to maintain (in GB).
    """

    def __init__(self, data_dir: Path, min_free_disk_gb: int = 10):
        """Initialize the model storage service.

        Args:
            data_dir: Root directory for model storage. Created if not exists.
            min_free_disk_gb: Minimum free disk space to maintain (in GB).
        """
        self.data_dir = Path(data_dir)
        self.min_free_disk_gb = min_free_disk_gb

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _generate_model_id(self) -> str:
        """Generate a unique model ID with prefix.

        Returns:
            A unique model ID in the format 'model_<uuid>'.
        """
        return f"model_{uuid.uuid4().hex[:12]}"

    def _get_model_dir(self, model_id: str) -> Path:
        """Get the directory path for a model ID.

        Args:
            model_id: The model identifier.

        Returns:
            Path to the model's directory.
        """
        return self.data_dir / model_id

    def _load_metadata(self, model_id: str) -> dict | None:
        """Load metadata for a model.

        Args:
            model_id: The model identifier.

        Returns:
            Metadata dict or None if not found.
        """
        model_dir = self._get_model_dir(model_id)
        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            return json.load(f)

    def _save_metadata(self, model_id: str, metadata: dict) -> None:
        """Save metadata for a model.

        Args:
            model_id: The model identifier.
            metadata: Metadata dict to persist.
        """
        model_dir = self._get_model_dir(model_id)
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _calculate_model_size(self, model_id: str) -> int:
        """Calculate total size of model files.

        Args:
            model_id: The model identifier.

        Returns:
            Total size in bytes.
        """
        model_dir = self._get_model_dir(model_id)
        total_size = 0
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    async def create_model_entry(
        self,
        name: str,
        model_type: str,
        source: dict,
    ) -> str:
        """Create a new model entry.

        Args:
            name: Human-readable model name.
            model_type: Type of model (e.g., 'llm').
            source: Source information dict.

        Returns:
            The generated model_id.
        """
        model_id = self._generate_model_id()
        model_dir = self._get_model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_id": model_id,
            "name": name,
            "type": model_type,
            "source": source,
            "status": ModelStatus.PULLING.value,
            "size_bytes": 0,
            "files": [],
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

        self._save_metadata(model_id, metadata)
        return model_id

    async def get_model(self, model_id: str) -> ModelDetailResponse | None:
        """Retrieve a model by ID.

        Args:
            model_id: The model identifier.

        Returns:
            ModelDetailResponse or None if not found.
        """
        metadata = self._load_metadata(model_id)
        if metadata is None:
            return None

        # Get default config if exists
        default_config = await self.get_default_config(model_id)

        # Convert source dict to ModelSource
        source_dict = metadata["source"]
        source = ModelSource(
            type=source_dict.get("type", "unknown"),
            repo=source_dict.get("repo"),
            revision=source_dict.get("revision"),
            endpoint=source_dict.get("endpoint"),
            filename=source_dict.get("filename"),
        )

        return ModelDetailResponse(
            model_id=metadata["model_id"],
            name=metadata["name"],
            type=metadata["type"],
            status=metadata["status"],
            source=source,
            size_bytes=metadata.get("size_bytes", 0),
            files=metadata.get("files", []),
            created_at=metadata["created_at"],
            default_config=default_config,
        )

    async def list_models(
        self,
        model_type: str | None,
        status: str | None,
    ) -> list[ModelDetailResponse]:
        """List models with optional filtering.

        Args:
            model_type: Filter by model type.
            status: Filter by status.

        Returns:
            List of ModelDetailResponse objects.
        """
        models = []

        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_id = model_dir.name
            if not model_id.startswith("model_"):
                continue

            model = await self.get_model(model_id)
            if model is None:
                continue

            # Apply filters
            if model_type and model.type != model_type:
                continue
            if status and model.status != status:
                continue

            models.append(model)

        # Sort by created_at descending
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models

    async def update_model_status(self, model_id: str, status: str) -> None:
        """Update a model's status.

        Args:
            model_id: The model identifier.
            status: New status value.

        Raises:
            ModelNotFoundError: If model doesn't exist.
        """
        metadata = self._load_metadata(model_id)
        if metadata is None:
            raise ModelNotFoundError(model_id)

        metadata["status"] = status
        self._save_metadata(model_id, metadata)

    async def delete_model(self, model_id: str) -> int:
        """Delete a model by ID.

        Args:
            model_id: The model identifier.

        Returns:
            Number of bytes freed.

        Raises:
            ModelNotFoundError: If model doesn't exist.
        """
        model_dir = self._get_model_dir(model_id)
        if not model_dir.exists():
            raise ModelNotFoundError(model_id)

        # Calculate size before deletion
        size = self._calculate_model_size(model_id)

        # Remove entire directory
        shutil.rmtree(model_dir)
        return size

    async def save_default_config(self, model_id: str, config: dict) -> None:
        """Save default configuration for a model.

        Args:
            model_id: The model identifier.
            config: Configuration dict to save.

        Raises:
            ModelNotFoundError: If model doesn't exist.
        """
        model_dir = self._get_model_dir(model_id)
        if not model_dir.exists():
            raise ModelNotFoundError(model_id)

        config_path = model_dir / "default_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    async def get_default_config(self, model_id: str) -> dict | None:
        """Get default configuration for a model.

        Args:
            model_id: The model identifier.

        Returns:
            Configuration dict or None if not set.
        """
        model_dir = self._get_model_dir(model_id)
        config_path = model_dir / "default_config.json"

        if not config_path.exists():
            return None

        with open(config_path) as f:
            return json.load(f)

    async def delete_default_config(self, model_id: str) -> None:
        """Delete default configuration for a model.

        Args:
            model_id: The model identifier.
        """
        model_dir = self._get_model_dir(model_id)
        config_path = model_dir / "default_config.json"

        if config_path.exists():
            config_path.unlink()

    async def update_model_size(self, model_id: str) -> None:
        """Recalculate and update model size.

        Args:
            model_id: The model identifier.

        Raises:
            ModelNotFoundError: If model doesn't exist.
        """
        metadata = self._load_metadata(model_id)
        if metadata is None:
            raise ModelNotFoundError(model_id)

        size = self._calculate_model_size(model_id)
        metadata["size_bytes"] = size

        # Update files list
        model_dir = self._get_model_dir(model_id)
        files = []
        for file_path in model_dir.iterdir():
            if file_path.is_file() and file_path.name not in [
                "metadata.json",
                "default_config.json",
            ]:
                files.append(file_path.name)
        metadata["files"] = files

        self._save_metadata(model_id, metadata)

    async def get_model_path(self, model_id: str) -> Path:
        """Get the filesystem path for a model.

        Args:
            model_id: The model identifier.

        Returns:
            Path to the model directory.

        Raises:
            ModelNotFoundError: If model doesn't exist.
        """
        model_dir = self._get_model_dir(model_id)
        if not model_dir.exists():
            raise ModelNotFoundError(model_id)

        return model_dir
