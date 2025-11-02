"""
Model Registry management
"""

import yaml
from pathlib import Path
from typing import Dict, Optional

from .config import config
from .models import ModelRegistryEntry


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
            raise FileNotFoundError(
                f"Model registry not found at: {self.registry_path}"
            )

        with open(self.registry_path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "models" not in data:
            raise ValueError("Invalid registry format: 'models' key not found")

        for model_data in data["models"]:
            entry = ModelRegistryEntry(**model_data)
            self.models[entry.model_id] = entry

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
