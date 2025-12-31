"""Instance registry for managing model instances.

This module provides backward compatibility by re-exporting
from src.registry.instance_registry.
"""

from src.registry.instance_registry import InstanceRegistry

__all__ = ["InstanceRegistry"]
