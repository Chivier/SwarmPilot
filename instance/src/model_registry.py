"""
Model Registry management
"""

import traceback
import yaml
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from .config import config
from .models import ModelRegistryEntry


def log_error_with_traceback(
    error: Exception,
    context: str,
    additional_info: str = "",
) -> None:
    """
    Log error with detailed information and traceback.

    Args:
        error: The exception that occurred
        context: Context description of where the error occurred
        additional_info: Additional context information
    """
    tb_str = traceback.format_exc()
    if additional_info:
        logger.error(
            f"[ModelRegistry] [{context}] Error occurred:\n"
            f"  Internal error: {type(error).__name__}: {error}\n"
            f"  {additional_info}\n"
            f"  Traceback:\n{tb_str}"
        )
    else:
        logger.error(
            f"[ModelRegistry] [{context}] Error occurred:\n"
            f"  Internal error: {type(error).__name__}: {error}\n"
            f"  Traceback:\n{tb_str}"
        )


class ModelRegistry:
    """
    Manages the model registry configuration.

    Loads and provides access to model metadata from the registry file.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or config.registry_path
        self.models: Dict[str, ModelRegistryEntry] = {}
        self._load_registry()

    def _load_registry(self):
        """Load model registry from YAML file"""
        if not self.registry_path.exists():
            error_msg = f"Model registry not found at: {self.registry_path}"
            logger.error(
                f"[ModelRegistry] [_load_registry] Error occurred:\n"
                f"  Internal error: FileNotFoundError: {error_msg}\n"
                f"  Registry path: {self.registry_path}"
            )
            raise FileNotFoundError(error_msg)

        try:
            with open(self.registry_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            log_error_with_traceback(
                error=e,
                context="_load_registry/yaml_parse",
                additional_info=f"Registry path: {self.registry_path}",
            )
            raise

        if not data or "models" not in data:
            error_msg = "Invalid registry format: 'models' key not found"
            logger.error(
                f"[ModelRegistry] [_load_registry] Error occurred:\n"
                f"  Internal error: ValueError: {error_msg}\n"
                f"  Registry path: {self.registry_path}\n"
                f"  Data keys: {list(data.keys()) if data else 'None'}"
            )
            raise ValueError(error_msg)

        for model_data in data["models"]:
            try:
                entry = ModelRegistryEntry(**model_data)
                self.models[entry.model_id] = entry
            except Exception as e:
                log_error_with_traceback(
                    error=e,
                    context="_load_registry/create_entry",
                    additional_info=f"Model data: {model_data}",
                )
                raise

    def get_model(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """Get model entry by ID"""
        return self.models.get(model_id)

    def list_models(self) -> Dict[str, ModelRegistryEntry]:
        """List all registered models"""
        return self.models.copy()

    def model_exists(self, model_id: str) -> bool:
        """Check if a model is registered"""
        return model_id in self.models

    def get_model_directory(self, model_id: str) -> Optional[Path]:
        """Get full path to model directory"""
        entry = self.get_model(model_id)
        if entry:
            return config.get_model_directory(entry.directory)
        return None

    def reload(self):
        """Reload registry from file"""
        self.models.clear()
        self._load_registry()


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the global registry instance"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
