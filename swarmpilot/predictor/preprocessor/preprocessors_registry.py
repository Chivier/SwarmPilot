"""V1 preprocessor registry (uses :class:`BasePreprocessor`).

.. deprecated::
    See :mod:`~swarmpilot.predictor.preprocessor.registry_v2` for the
    V2 registry that supports flexible feature operations.
"""

from __future__ import annotations

from pathlib import Path

from swarmpilot.predictor.preprocessor.base_preprocessor import BasePreprocessor
from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()


class PreprocessorsRegistry:
    """Registry for managing preprocessor instances.

    Preprocessors are loaded lazily on first access to avoid import-time
    failures when model files are not available (e.g., in CI environments).

    Attributes:
        preprocessors: Dictionary mapping names to preprocessor instances.
    """

    def __init__(self) -> None:
        """Initialize the registry with lazy loading."""
        self._preprocessors: dict[str, BasePreprocessor] = {}
        self._initialized = False

    def _init_preprocessors(self) -> None:
        """Initialize preprocessors lazily on first access."""
        if self._initialized:
            return

        self._initialized = True

        # Get the predictor directory path (parent of src)
        predictor_dir = Path(__file__).parent.parent.parent
        model_dir = predictor_dir / "preprocessors" / "sematic_35M"
        model_path = model_dir / "model_35M.pt"
        config_path = model_dir / "model_35M.yaml"

        # Only load semantic predictor if model files exist
        if model_path.exists() and config_path.exists():
            try:
                from swarmpilot.predictor.preprocessor.semantic_predictor import (
                    SemanticPredictor,
                )

                self._preprocessors["semantic"] = SemanticPredictor(
                    str(model_path),
                    str(config_path),
                )
                logger.info("Loaded semantic preprocessor successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic preprocessor: {e}")
        else:
            logger.info(
                f"Semantic preprocessor model not found at {model_path}, skipping"
            )

    @property
    def preprocessors(self) -> dict[str, BasePreprocessor]:
        """Get preprocessors dict, initializing lazily if needed."""
        self._init_preprocessors()
        return self._preprocessors

    def get_preprocessor(self, preprocessor_name: str) -> BasePreprocessor:
        """Get a preprocessor by name.

        Args:
            preprocessor_name: Name of the preprocessor to retrieve.

        Returns:
            The preprocessor instance.

        Raises:
            KeyError: If preprocessor name not found.
        """
        if preprocessor_name not in self.preprocessors:
            raise KeyError(f"Preprocessor {preprocessor_name} not found")
        return self.preprocessors[preprocessor_name]
