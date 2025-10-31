"""
Model storage layer for persisting and retrieving trained models.

Uses joblib for serialization and local filesystem for storage.
"""

import os
import joblib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path


class ModelStorage:
    """
    Manages model persistence using local filesystem.

    Models are stored with metadata including training date, sample count, etc.
    """

    def __init__(self, storage_dir: str = "models"):
        """
        Initialize model storage.

        Args:
            storage_dir: Directory for storing models (created if doesn't exist)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def generate_model_key(self, model_id: str, platform_info: Dict[str, str]) -> str:
        """
        Generate unique model key from model_id and platform_info.

        Format: {model_id}__{software_name}-{software_version}__{hardware_name}

        Args:
            model_id: Model identifier
            platform_info: Platform information dict with software_name,
                          software_version, hardware_name

        Returns:
            Unique model key string

        Example:
            >>> generate_model_key("image-classifier-v1",
            ...                   {"software_name": "pytorch",
            ...                    "software_version": "2.0.1",
            ...                    "hardware_name": "nvidia-a100"})
            'image-classifier-v1__pytorch-2.0.1__nvidia-a100'
        """
        software = f"{platform_info['software_name']}-{platform_info['software_version']}"
        hardware = platform_info['hardware_name']
        return f"{model_id}__{software}__{hardware}"

    def save_model(self,
                   model_key: str,
                   predictor_state: Dict[str, Any],
                   metadata: Dict[str, Any]) -> None:
        """
        Save model and metadata to disk.

        Args:
            model_key: Unique key for the model
            predictor_state: Complete predictor state from get_model_state()
            metadata: Additional metadata (model_id, platform_info, samples_count, etc.)
        """
        # Create complete state with metadata
        complete_state = {
            'predictor_state': predictor_state,
            'metadata': metadata,
            'saved_at': datetime.now(timezone.utc).isoformat()
        }

        # Save to disk
        model_path = self.storage_dir / f"{model_key}.joblib"
        joblib.dump(complete_state, model_path)

    def load_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """
        Load model and metadata from disk.

        Args:
            model_key: Unique key for the model

        Returns:
            Dict with 'predictor_state' and 'metadata' keys, or None if not found
        """
        model_path = self.storage_dir / f"{model_key}.joblib"

        if not model_path.exists():
            return None

        return joblib.load(model_path)

    def model_exists(self, model_key: str) -> bool:
        """
        Check if a model exists in storage.

        Args:
            model_key: Unique key for the model

        Returns:
            True if model exists, False otherwise
        """
        model_path = self.storage_dir / f"{model_key}.joblib"
        return model_path.exists()

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all stored models with their metadata.

        Returns:
            List of model metadata dicts
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
                # Skip corrupted files
                print(f"Warning: Failed to load {model_file}: {e}")
                continue

        return models

    def delete_model(self, model_key: str) -> bool:
        """
        Delete a model from storage.

        Args:
            model_key: Unique key for the model

        Returns:
            True if model was deleted, False if not found
        """
        model_path = self.storage_dir / f"{model_key}.joblib"

        if not model_path.exists():
            return False

        model_path.unlink()
        return True

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage system.

        Returns:
            Dict with storage statistics
        """
        model_files = list(self.storage_dir.glob("*.joblib"))
        total_size = sum(f.stat().st_size for f in model_files)

        return {
            'storage_dir': str(self.storage_dir.absolute()),
            'model_count': len(model_files),
            'total_size_bytes': total_size,
            'is_accessible': self.storage_dir.exists() and os.access(self.storage_dir, os.W_OK)
        }
