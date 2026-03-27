"""Model storage layer for persisting and retrieving trained models.

Uses joblib for serialization and local filesystem for storage.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()


class ModelStorage:
    """Manages model persistence using local filesystem.

    Models are stored with metadata including training date, sample count, etc.

    Attributes:
        storage_dir: Path to the directory where models are stored.
    """

    def __init__(self, storage_dir: str = "models") -> None:
        """Initialize model storage.

        Args:
            storage_dir: Directory for storing models (created if not exists).
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: dict[str, Any] = {}

    def generate_model_key(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str = "expect_error",
    ) -> str:
        """Generate unique model key from model_id and platform_info.

        Format: {model_id}__{software}-{version}__{hardware}__{prediction_type}

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict with software_name,
                software_version, hardware_name.
            prediction_type: Type of prediction model (expect_error or quantile).

        Returns:
            Unique model key string.
        """
        software = f"{platform_info['software_name']}-{platform_info['software_version']}"
        hardware = platform_info["hardware_name"]
        return f"{model_id}__{software}__{hardware}__{prediction_type}"

    def save_model(
        self,
        model_key: str,
        predictor_state: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Save model and metadata to disk.

        Args:
            model_key: Unique key for the model.
            predictor_state: Complete predictor state from get_model_state().
            metadata: Additional metadata (model_id, platform_info, etc).

        Raises:
            Exception: If saving fails (joblib error, disk error, etc).
        """
        # Create complete state with metadata
        complete_state = {
            "predictor_state": predictor_state,
            "metadata": metadata,
            "saved_at": datetime.now(UTC).isoformat(),
        }

        # Save to disk
        model_path = self.storage_dir / f"{model_key}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(complete_state, model_path)
            logger.debug(f"Model saved successfully: {model_path}")
        except (OSError, RuntimeError) as e:
            logger.opt(exception=True).error(
                f"Failed to save model\n"
                f"Model key: {model_key}\n"
                f"Model path: {model_path}\n"
                f"Metadata: {metadata}\n"
                f"Exception: {type(e).__name__}: {e!s}"
            )
            raise

    def load_model(self, model_key: str) -> dict[str, Any] | None:
        """Load model and metadata from disk.

        Args:
            model_key: Unique key for the model.

        Returns:
            Dict with predictor_state and metadata keys, or None if not found.

        Raises:
            Exception: If loading fails (corrupted file, deserialization error).
        """
        model_path = self.storage_dir / f"{model_key}.joblib"

        if not model_path.exists():
            logger.debug(f"Model not found at path: {model_path}")
            return None

        try:
            result = joblib.load(model_path)
            logger.debug(f"Model loaded successfully: {model_path}")
            return result
        except (OSError, EOFError, RuntimeError) as e:
            logger.opt(exception=True).error(
                f"Failed to load model\n"
                f"Model key: {model_key}\n"
                f"Model path: {model_path}\n"
                f"Exception: {type(e).__name__}: {e!s}"
            )
            raise

    def model_exists(self, model_key: str) -> bool:
        """Check if a model exists in storage.

        Args:
            model_key: Unique key for the model.

        Returns:
            True if model exists, False otherwise.
        """
        model_path = self.storage_dir / f"{model_key}.joblib"
        return model_path.exists()

    def list_models(self) -> list[dict[str, Any]]:
        """List all stored models with their metadata.

        Returns:
            List of model metadata dicts.
        """
        models = []

        for model_file in self.storage_dir.glob("*.joblib"):
            try:
                complete_state = joblib.load(model_file)
                metadata = complete_state.get("metadata", {})

                # Extract key information
                model_info = {
                    "model_id": metadata.get("model_id"),
                    "platform_info": metadata.get("platform_info"),
                    "prediction_type": metadata.get("prediction_type"),
                    "samples_count": metadata.get("samples_count"),
                    "last_trained": complete_state.get("saved_at"),
                }

                models.append(model_info)

            except (OSError, EOFError, RuntimeError, KeyError) as e:
                # Log corrupted files with full details
                logger.opt(exception=True).warning(
                    f"Failed to load model file (skipping)\n"
                    f"File: {model_file}\n"
                    f"Exception: {type(e).__name__}: {e!s}"
                )
                continue

        return models

    def delete_model(self, model_key: str) -> bool:
        """Delete a model from storage.

        Args:
            model_key: Unique key for the model.

        Returns:
            True if model was deleted, False if not found.
        """
        model_path = self.storage_dir / f"{model_key}.joblib"

        if not model_path.exists():
            return False

        model_path.unlink()
        return True

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about the storage system.

        Returns:
            Dict with storage statistics (dir, count, size, accessibility).
        """
        model_files = list(self.storage_dir.glob("*.joblib"))
        total_size = sum(f.stat().st_size for f in model_files)

        return {
            "storage_dir": str(self.storage_dir.absolute()),
            "model_count": len(model_files),
            "total_size_bytes": total_size,
            "is_accessible": self.storage_dir.exists()
            and os.access(self.storage_dir, os.W_OK),
        }
