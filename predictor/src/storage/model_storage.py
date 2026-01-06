"""Model storage layer for persisting and retrieving trained models.

Uses joblib for serialization and local filesystem for storage.
Supports versioned model storage with Unix timestamps for distributed consistency.
"""

from __future__ import annotations

import os
import re
import time
import traceback
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import joblib

from src.utils.logging import get_logger

logger = get_logger()

# Version suffix pattern: __v{unix_timestamp}
VERSION_PATTERN = re.compile(r"__v(\d+)$")


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
        hardware = platform_info['hardware_name']
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
            'predictor_state': predictor_state,
            'metadata': metadata,
            'saved_at': datetime.now(timezone.utc).isoformat()
        }

        # Save to disk
        model_path = self.storage_dir / f"{model_key}.joblib"
        try:
            joblib.dump(complete_state, model_path)
            logger.debug(f"Model saved successfully: {model_path}")
        except Exception as e:
            logger.error(
                f"Failed to save model\n"
                f"Model key: {model_key}\n"
                f"Model path: {model_path}\n"
                f"Metadata: {metadata}\n"
                f"Exception: {type(e).__name__}: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
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
        except Exception as e:
            logger.error(
                f"Failed to load model\n"
                f"Model key: {model_key}\n"
                f"Model path: {model_path}\n"
                f"Exception: {type(e).__name__}: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
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
                metadata = complete_state.get('metadata', {})

                # Extract key information
                model_info = {
                    'model_id': metadata.get('model_id'),
                    'platform_info': metadata.get('platform_info'),
                    'prediction_type': metadata.get('prediction_type'),
                    'samples_count': metadata.get('samples_count'),
                    'last_trained': complete_state.get('saved_at')
                }

                models.append(model_info)

            except Exception as e:
                # Log corrupted files with full details
                logger.warning(
                    f"Failed to load model file (skipping)\n"
                    f"File: {model_file}\n"
                    f"Exception: {type(e).__name__}: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
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
            'storage_dir': str(self.storage_dir.absolute()),
            'model_count': len(model_files),
            'total_size_bytes': total_size,
            'is_accessible': self.storage_dir.exists() and os.access(self.storage_dir, os.W_OK)
        }

    # =========================================================================
    # Versioned Model Storage Methods
    # =========================================================================

    def get_base_model_key(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str = "expect_error",
    ) -> str:
        """Generate base model key without version suffix.

        This is the same as generate_model_key() but explicitly named
        to distinguish from versioned keys.

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict.
            prediction_type: Type of prediction model.

        Returns:
            Base model key string (without version suffix).
        """
        return self.generate_model_key(model_id, platform_info, prediction_type)

    def generate_versioned_model_key(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str = "expect_error",
        version: int | None = None,
    ) -> str:
        """Generate model key with version suffix.

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict.
            prediction_type: Type of prediction model.
            version: Unix timestamp version. If None, uses current time.

        Returns:
            Versioned model key: {base_key}__v{timestamp}
        """
        base_key = self.get_base_model_key(model_id, platform_info, prediction_type)
        if version is None:
            version = int(time.time())
        return f"{base_key}__v{version}"

    def _extract_version_from_filename(self, filename: str) -> int | None:
        """Extract version number from a filename.

        Args:
            filename: Filename (with or without .joblib extension).

        Returns:
            Unix timestamp version, or None if no version suffix.
        """
        # Remove .joblib extension if present
        name = filename.replace(".joblib", "")
        match = VERSION_PATTERN.search(name)
        if match:
            return int(match.group(1))
        return None

    def _strip_version_from_key(self, model_key: str) -> str:
        """Strip version suffix from model key.

        Args:
            model_key: Model key (may or may not have version suffix).

        Returns:
            Base model key without version suffix.
        """
        return VERSION_PATTERN.sub("", model_key)

    def list_versions(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str = "expect_error",
    ) -> list[int]:
        """List all available versions for a model configuration.

        Scans the storage directory for all versioned files matching
        the base key pattern.

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict.
            prediction_type: Type of prediction model.

        Returns:
            List of unix timestamps (versions), sorted descending (newest first).
            Includes version 0 if a legacy (unversioned) file exists.
        """
        base_key = self.get_base_model_key(model_id, platform_info, prediction_type)
        versions: list[int] = []

        # Check for versioned files: {base_key}__v*.joblib
        pattern = f"{base_key}__v*.joblib"
        for model_file in self.storage_dir.glob(pattern):
            version = self._extract_version_from_filename(model_file.name)
            if version is not None:
                versions.append(version)

        # Check for legacy file (no version suffix) - treated as version 0
        legacy_path = self.storage_dir / f"{base_key}.joblib"
        if legacy_path.exists():
            versions.append(0)

        # Sort descending (newest first)
        versions.sort(reverse=True)
        return versions

    def get_latest_version(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str = "expect_error",
    ) -> int | None:
        """Get the latest version timestamp for a model configuration.

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict.
            prediction_type: Type of prediction model.

        Returns:
            Unix timestamp of latest version, or None if no versions exist.
        """
        versions = self.list_versions(model_id, platform_info, prediction_type)
        return versions[0] if versions else None

    def save_model_versioned(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str,
        predictor_state: dict[str, Any],
        metadata: dict[str, Any],
        version: int | None = None,
    ) -> int:
        """Save model with version suffix.

        Creates a new versioned file. Does not overwrite existing versions.

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict.
            prediction_type: Type of prediction model.
            predictor_state: Complete predictor state from get_model_state().
            metadata: Additional metadata (will be augmented with version info).
            version: Unix timestamp version. If None, uses current time.

        Returns:
            The version timestamp used for saving.

        Raises:
            Exception: If saving fails.
        """
        if version is None:
            version = int(time.time())

        # Add version info to metadata
        metadata = metadata.copy()
        metadata["version"] = version
        metadata["version_iso"] = datetime.fromtimestamp(
            version, tz=timezone.utc
        ).isoformat()

        # Generate versioned key
        versioned_key = self.generate_versioned_model_key(
            model_id, platform_info, prediction_type, version
        )

        # Create complete state with metadata
        complete_state = {
            'predictor_state': predictor_state,
            'metadata': metadata,
            'saved_at': datetime.now(timezone.utc).isoformat()
        }

        # Save to disk
        model_path = self.storage_dir / f"{versioned_key}.joblib"
        try:
            joblib.dump(complete_state, model_path)
            logger.info(f"Model saved with version {version}: {model_path}")
            return version
        except Exception as e:
            logger.error(
                f"Failed to save versioned model\n"
                f"Versioned key: {versioned_key}\n"
                f"Model path: {model_path}\n"
                f"Version: {version}\n"
                f"Exception: {type(e).__name__}: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise

    def load_model_versioned(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str = "expect_error",
        version: int | None = None,
    ) -> tuple[dict[str, Any] | None, int | None]:
        """Load specific version or latest version of a model.

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict.
            prediction_type: Type of prediction model.
            version: Specific version to load. If None, loads latest version.
                     Version 0 means load legacy (unversioned) file.

        Returns:
            Tuple of (model_data, loaded_version):
            - model_data: Dict with predictor_state and metadata, or None if not found.
            - loaded_version: The version that was loaded, or None if not found.

        Raises:
            Exception: If loading fails (corrupted file, deserialization error).
        """
        # Determine which version to load
        if version is None:
            version = self.get_latest_version(model_id, platform_info, prediction_type)
            if version is None:
                logger.debug(f"No versions found for model: {model_id}")
                return None, None

        # Construct path based on version
        if version == 0:
            # Legacy file (no version suffix)
            base_key = self.get_base_model_key(
                model_id, platform_info, prediction_type
            )
            model_path = self.storage_dir / f"{base_key}.joblib"
        else:
            # Versioned file
            versioned_key = self.generate_versioned_model_key(
                model_id, platform_info, prediction_type, version
            )
            model_path = self.storage_dir / f"{versioned_key}.joblib"

        if not model_path.exists():
            logger.debug(f"Model version {version} not found at: {model_path}")
            return None, None

        try:
            result = joblib.load(model_path)
            logger.debug(f"Loaded model version {version}: {model_path}")
            return result, version
        except Exception as e:
            logger.error(
                f"Failed to load versioned model\n"
                f"Model path: {model_path}\n"
                f"Version: {version}\n"
                f"Exception: {type(e).__name__}: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise

    def delete_version(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str,
        version: int,
    ) -> bool:
        """Delete a specific version of a model.

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict.
            prediction_type: Type of prediction model.
            version: Version to delete. 0 means legacy file.

        Returns:
            True if deleted, False if not found.
        """
        if version == 0:
            # Legacy file
            base_key = self.get_base_model_key(
                model_id, platform_info, prediction_type
            )
            model_path = self.storage_dir / f"{base_key}.joblib"
        else:
            # Versioned file
            versioned_key = self.generate_versioned_model_key(
                model_id, platform_info, prediction_type, version
            )
            model_path = self.storage_dir / f"{versioned_key}.joblib"

        if not model_path.exists():
            return False

        model_path.unlink()
        logger.info(f"Deleted model version {version}: {model_path}")
        return True

    def get_version_info(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str = "expect_error",
    ) -> dict[str, Any]:
        """Get comprehensive version information for a model configuration.

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict.
            prediction_type: Type of prediction model.

        Returns:
            Dict with version information:
            - model_id: str
            - platform_info: dict
            - prediction_type: str
            - latest_version: int | None
            - latest_version_iso: str | None
            - available_versions: list[int]
            - version_count: int
        """
        versions = self.list_versions(model_id, platform_info, prediction_type)
        latest = versions[0] if versions else None
        latest_iso = None
        if latest is not None and latest > 0:
            latest_iso = datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()
        elif latest == 0:
            # Legacy file - no timestamp available
            latest_iso = "legacy"

        return {
            "model_id": model_id,
            "platform_info": platform_info,
            "prediction_type": prediction_type,
            "latest_version": latest,
            "latest_version_iso": latest_iso,
            "available_versions": versions,
            "version_count": len(versions),
        }
