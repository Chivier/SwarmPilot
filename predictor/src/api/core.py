"""Library-style API for the predictor module.

This module provides a two-level API for using the predictor without HTTP:

- **PredictorLowLevel**: Full control over model management, storage, and cache.
- **PredictorCore**: Simple high-level API with accumulator pattern for training.

Example:
    >>> from src.api.core import PredictorCore
    >>> from src.models import PlatformInfo
    >>>
    >>> core = PredictorCore()
    >>> platform = PlatformInfo(
    ...     software_name="PyTorch",
    ...     software_version="2.0",
    ...     hardware_name="NVIDIA A100"
    ... )
    >>>
    >>> # Collect samples
    >>> for i in range(10):
    ...     core.collect("model", platform, "quantile", {"x": i}, runtime_ms=i*10)
    >>>
    >>> # Train on accumulated data
    >>> result = core.train("model", platform, "quantile")
    >>>
    >>> # Predict
    >>> pred = core.predict("model", platform, "quantile", {"x": 5})
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from src.api.cache import ModelCache
from src.config import PredictorConfig
from src.config import get_config
from src.models import CollectedSample
from src.preprocessor.chain_v2 import PreprocessorChainV2
from src.preprocessor.preprocessors_registry import PreprocessorsRegistry
from src.models import ModelInfo
from src.models import PlatformInfo
from src.models import PredictionResult
from src.models import TrainingResult
from src.predictor.base import BasePredictor
from src.predictor.decision_tree import DecisionTreePredictor
from src.predictor.expect_error import ExpectErrorPredictor
from src.predictor.linear_regression import LinearRegressionPredictor
from src.predictor.quantile import QuantilePredictor
from src.storage.model_storage import ModelStorage
from src.utils.logging import get_logger


logger = get_logger()


# =============================================================================
# Exceptions
# =============================================================================


class PredictorError(Exception):
    """Base exception for all predictor errors."""

    pass


class ModelNotFoundError(PredictorError):
    """Raised when a requested model does not exist."""

    pass


class ValidationError(PredictorError):
    """Raised when input validation fails."""

    pass


class TrainingError(PredictorError):
    """Raised when model training fails."""

    pass


class PredictionError(PredictorError):
    """Raised when prediction fails."""

    pass


# =============================================================================
# Constants
# =============================================================================


PREDICTOR_CLASSES: dict[str, type[BasePredictor]] = {
    "expect_error": ExpectErrorPredictor,
    "quantile": QuantilePredictor,
    "linear_regression": LinearRegressionPredictor,
    "decision_tree": DecisionTreePredictor,
}

MIN_TRAINING_SAMPLES = 10


# =============================================================================
# PredictorLowLevel - Low-Level API
# =============================================================================


class PredictorLowLevel:
    """Low-level API for direct predictor control.

    This class provides full control over model management, storage,
    and cache operations. Use this when you need fine-grained control
    over the predictor lifecycle.

    Attributes:
        storage_dir: Directory where models are stored.
        cache_size: Maximum number of models to cache.

    Example:
        >>> low = PredictorLowLevel()
        >>> predictor = low.train_predictor(training_data, "expect_error")
        >>> low.save_model("my_model", platform, "expect_error", predictor)
        >>> loaded = low.load_model("my_model", platform, "expect_error")
        >>> result = low.predict_with_predictor(loaded, features)
    """

    def __init__(
        self,
        config: PredictorConfig | None = None,
        storage_dir: str | None = None,
        cache_size: int = 100,
    ) -> None:
        """Initialize low-level predictor API.

        Args:
            config: Configuration object. If None, uses default config.
            storage_dir: Override storage directory from config.
            cache_size: Maximum number of models to cache in memory.
        """
        self._config = config or get_config()
        storage_path = storage_dir or self._config.storage_dir
        self._storage = ModelStorage(storage_dir=storage_path)
        self._cache = ModelCache(max_size=cache_size)
        self._preprocessors_registry = PreprocessorsRegistry()
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # Model Management
    # -------------------------------------------------------------------------

    def train_predictor(
        self,
        features_list: list[dict[str, Any]],
        prediction_type: str,
        config: dict[str, Any] | None = None,
    ) -> BasePredictor:
        """Train a new predictor instance.

        Args:
            features_list: Training data with runtime_ms field.
            prediction_type: Type of predictor to train.
            config: Optional training configuration.

        Returns:
            Trained predictor instance.

        Raises:
            ValidationError: If training data is invalid or insufficient.
            TrainingError: If training fails.
        """
        # Validate prediction type
        if prediction_type not in PREDICTOR_CLASSES:
            raise ValidationError(
                f"Invalid prediction_type: {prediction_type}. "
                f"Must be one of: {list(PREDICTOR_CLASSES.keys())}"
            )

        # Validate minimum samples
        if len(features_list) < MIN_TRAINING_SAMPLES:
            raise ValidationError(
                f"Insufficient training data. Got {len(features_list)} samples, "
                f"but minimum {MIN_TRAINING_SAMPLES} samples required."
            )

        # Create predictor instance
        predictor_class = PREDICTOR_CLASSES[prediction_type]
        predictor = predictor_class()

        # Train
        try:
            predictor.train(features_list, config)
            logger.info(
                f"Trained {prediction_type} predictor with "
                f"{len(features_list)} samples"
            )
            return predictor
        except Exception as e:
            raise TrainingError(f"Training failed: {e}") from e

    def save_model(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        predictor: BasePredictor,
        samples_count: int | None = None,
        extra_metadata: dict[str, Any] | None = None,
        version: int | None = None,
    ) -> int:
        """Save a predictor to storage with versioning.

        Creates a new versioned model file. Does not overwrite existing versions.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            predictor: Trained predictor instance.
            samples_count: Number of samples used for training (optional).
                If not provided, attempts to get from predictor state.
            extra_metadata: Additional metadata to store with the model.
            version: Unix timestamp version. If None, uses current time.

        Returns:
            The version timestamp used for saving.
        """
        # Get predictor state
        predictor_state = predictor.get_model_state()

        # Determine samples count - prefer explicit parameter, fall back to predictor state
        actual_samples_count = samples_count
        if actual_samples_count is None:
            actual_samples_count = predictor_state.get("samples_count", 0)

        # Prepare metadata (version will be added by storage layer)
        metadata = {
            "model_id": model_id,
            "platform_info": platform_info.model_dump(),
            "prediction_type": prediction_type,
            "samples_count": actual_samples_count,
            "feature_names": predictor_state.get("feature_names", []),
            "saved_at": datetime.now().isoformat(),
        }

        # Merge extra metadata if provided
        if extra_metadata:
            metadata.update(extra_metadata)

        # Save to versioned storage
        with self._lock:
            saved_version = self._storage.save_model_versioned(
                model_id=model_id,
                platform_info=platform_info.model_dump(),
                prediction_type=prediction_type,
                predictor_state=predictor_state,
                metadata=metadata,
                version=version,
            )

            # Invalidate all cached versions for this model
            base_key = self.generate_model_key(model_id, platform_info, prediction_type)
            self._cache.invalidate_prefix(base_key)

        logger.info(f"Saved model {model_id} version {saved_version}")
        return saved_version

    def load_model(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        version: int | None = None,
    ) -> BasePredictor:
        """Load a predictor from storage.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            version: Specific version to load. If None, loads latest version.

        Returns:
            Loaded predictor instance.

        Raises:
            ModelNotFoundError: If model does not exist.
        """
        base_key = self.generate_model_key(model_id, platform_info, prediction_type)

        with self._lock:
            # Determine which version to load
            if version is None:
                version = self._storage.get_latest_version(
                    model_id, platform_info.model_dump(), prediction_type
                )
                if version is None:
                    raise ModelNotFoundError(f"Model not found: {base_key}")

            # Create cache key with version
            cache_key = f"{base_key}__v{version}"

            # Check cache first
            cached = self._cache.get(cache_key)
            if cached is not None:
                predictor, _ = cached
                logger.debug(f"Cache hit for model: {cache_key}")
                return predictor

            # Load from versioned storage
            model_data, loaded_version = self._storage.load_model_versioned(
                model_id=model_id,
                platform_info=platform_info.model_dump(),
                prediction_type=prediction_type,
                version=version,
            )
            if model_data is None:
                raise ModelNotFoundError(f"Model version {version} not found: {base_key}")

            # Create predictor instance
            if prediction_type not in PREDICTOR_CLASSES:
                raise ValidationError(f"Invalid prediction_type: {prediction_type}")

            predictor_class = PREDICTOR_CLASSES[prediction_type]
            predictor = predictor_class()
            predictor.load_model_state(model_data["predictor_state"])

            # Cache it with version-specific key
            self._cache.put(cache_key, predictor, prediction_type)

            logger.debug(f"Loaded model version {loaded_version}: {cache_key}")
            return predictor

    def delete_model(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> bool:
        """Delete all versions of a model from storage.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            True if any version was deleted, False if none found.
        """
        base_key = self.generate_model_key(model_id, platform_info, prediction_type)

        with self._lock:
            # Get all versions
            versions = self._storage.list_versions(
                model_id=model_id,
                platform_info=platform_info.model_dump(),
                prediction_type=prediction_type,
            )

            if not versions:
                return False

            # Delete each version and invalidate cache
            for version in versions:
                cache_key = f"{base_key}__v{version}"
                self._cache.invalidate(cache_key)

                self._storage.delete_version(
                    model_id=model_id,
                    platform_info=platform_info.model_dump(),
                    prediction_type=prediction_type,
                    version=version,
                )

            # Also invalidate base key cache (legacy)
            self._cache.invalidate(base_key)

        logger.info(f"Deleted model {model_id} ({len(versions)} versions)")
        return True

    def list_models(
        self,
        model_id: str | None = None,
        prediction_type: str | None = None,
    ) -> list[ModelInfo]:
        """List all models with optional filtering.

        Args:
            model_id: Filter by model_id prefix.
            prediction_type: Filter by prediction type.

        Returns:
            List of ModelInfo objects.
        """
        all_models = self._storage.list_models()

        result = []
        for model_data in all_models:
            # ModelStorage.list_models() returns flat structure, not nested metadata
            current_model_id = model_data.get("model_id", "")
            current_prediction_type = model_data.get("prediction_type", "")

            # Apply filters
            if model_id and not current_model_id.startswith(model_id):
                continue
            if prediction_type and current_prediction_type != prediction_type:
                continue

            # Convert to ModelInfo
            platform_dict = model_data.get("platform_info", {})
            if not platform_dict:
                logger.warning(f"Skipping model with missing platform_info: {current_model_id}")
                continue

            try:
                info = ModelInfo(
                    model_id=current_model_id,
                    platform_info=PlatformInfo(**platform_dict),
                    prediction_type=current_prediction_type,
                    samples_count=model_data.get("samples_count", 0),
                    last_trained=model_data.get("last_trained", ""),
                    feature_names=model_data.get("feature_names"),
                )
                result.append(info)
            except Exception as e:
                logger.warning(f"Failed to parse model info for {current_model_id}: {e}")
                continue

        return result

    def get_model_info(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        version: int | None = None,
    ) -> ModelInfo | None:
        """Get detailed information about a model.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            version: Specific version to get info for. If None, gets latest.

        Returns:
            ModelInfo if found, None otherwise.
        """
        # Use versioned loading
        model_data, loaded_version = self._storage.load_model_versioned(
            model_id=model_id,
            platform_info=platform_info.model_dump(),
            prediction_type=prediction_type,
            version=version,
        )

        if model_data is None:
            return None

        metadata = model_data.get("metadata", {})
        platform_dict = metadata.get("platform_info", {})

        return ModelInfo(
            model_id=metadata.get("model_id", model_id),
            platform_info=PlatformInfo(**platform_dict),
            prediction_type=metadata.get("prediction_type", prediction_type),
            samples_count=metadata.get("samples_count", 0),
            last_trained=metadata.get("saved_at", ""),
            feature_names=metadata.get("feature_names"),
        )

    def model_exists(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> bool:
        """Check if a model exists (any version).

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            True if at least one version exists, False otherwise.
        """
        # Check if any version exists (versioned or legacy)
        latest = self._storage.get_latest_version(
            model_id=model_id,
            platform_info=platform_info.model_dump(),
            prediction_type=prediction_type,
        )
        return latest is not None

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict_with_predictor(
        self,
        predictor: BasePredictor,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a prediction using a predictor instance.

        Args:
            predictor: Trained predictor instance.
            features: Feature dictionary.

        Returns:
            Prediction result dictionary.

        Raises:
            ValidationError: If required features are missing.
            PredictionError: If prediction fails.
        """
        # Get expected features from model
        expected_features = predictor.feature_names

        if expected_features is None:
            raise PredictionError("Predictor has no feature_names - not trained?")

        # Filter extra features (auto-filter)
        filtered = predictor.filter_features_for_prediction(features, expected_features)

        # Validate required features present
        missing = set(expected_features) - set(filtered.keys())
        if missing:
            raise ValidationError(
                f"Missing required features: {sorted(missing)}. "
                f"Model expects: {sorted(expected_features)}"
            )

        # Warn about extra features (debug level)
        extra = set(features.keys()) - set(expected_features)
        if extra:
            logger.debug(f"Ignoring extra features: {sorted(extra)}")

        # Make prediction
        try:
            return predictor.predict(filtered)
        except Exception as e:
            raise PredictionError(f"Prediction failed: {e}") from e

    # -------------------------------------------------------------------------
    # Cache Control
    # -------------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
        logger.info("Model cache cleared")

    def invalidate_cache(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> None:
        """Invalidate a specific model in the cache.

        Removes all cached versions of the model (versioned keys like
        '{base_key}__v{timestamp}').

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
        """
        base_key = self.generate_model_key(model_id, platform_info, prediction_type)
        with self._lock:
            # Use prefix invalidation to remove all versioned cache entries
            self._cache.invalidate_prefix(base_key)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, max_size, hits, misses).
        """
        return self._cache.get_stats()

    # -------------------------------------------------------------------------
    # Storage Control
    # -------------------------------------------------------------------------

    def get_storage_info(self) -> dict[str, Any]:
        """Get storage directory information.

        Returns:
            Dictionary with storage info.
        """
        return self._storage.get_storage_info()

    def generate_model_key(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> str:
        """Generate model key string.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            Model key string.
        """
        return self._storage.generate_model_key(
            model_id=model_id,
            platform_info=platform_info.model_dump(),
            prediction_type=prediction_type,
        )

    # -------------------------------------------------------------------------
    # Version Management
    # -------------------------------------------------------------------------

    def get_model_versions(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> list[int]:
        """Get all available versions for a model configuration.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            List of unix timestamps (versions), sorted descending (newest first).
        """
        return self._storage.list_versions(
            model_id=model_id,
            platform_info=platform_info.model_dump(),
            prediction_type=prediction_type,
        )

    def get_latest_version(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> int | None:
        """Get the latest version timestamp for a model configuration.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            Unix timestamp of latest version, or None if no versions exist.
        """
        return self._storage.get_latest_version(
            model_id=model_id,
            platform_info=platform_info.model_dump(),
            prediction_type=prediction_type,
        )

    def get_version_info(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> dict[str, Any]:
        """Get comprehensive version information for a model configuration.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

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
        return self._storage.get_version_info(
            model_id=model_id,
            platform_info=platform_info.model_dump(),
            prediction_type=prediction_type,
        )

    def delete_model_version(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        version: int,
    ) -> bool:
        """Delete a specific version of a model.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            version: Version to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            # Invalidate cache for this version
            base_key = self.generate_model_key(model_id, platform_info, prediction_type)
            cache_key = f"{base_key}__v{version}"
            self._cache.invalidate(cache_key)

            # Delete from storage
            return self._storage.delete_version(
                model_id=model_id,
                platform_info=platform_info.model_dump(),
                prediction_type=prediction_type,
                version=version,
            )

    # -------------------------------------------------------------------------
    # Inference Pipeline API
    # -------------------------------------------------------------------------

    def apply_preprocess_pipeline(
        self,
        features: dict[str, Any],
        preprocess_config: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Apply per-feature preprocessing chain.

        Each feature in the config is processed by its corresponding chain
        of preprocessors in order. Features not in the config are passed
        through unchanged.

        Args:
            features: Raw feature dictionary.
            preprocess_config: Maps feature_name to ordered list of preprocessor names.
                Format: {"feature_name": ["preprocessor_0", "preprocessor_1", ...]}

        Returns:
            Processed features dictionary.

        Example:
            >>> processed = low.apply_preprocess_pipeline(
            ...     features={"text": "hello", "num": 42},
            ...     preprocess_config={
            ...         "text": ["normalize", "tokenize", "embed"],  # Applied in order
            ...         # "num" not in config - kept as-is
            ...     },
            ... )
        """
        if not preprocess_config:
            return features.copy()

        processed = features.copy()

        for feature_name, preprocessor_chain in preprocess_config.items():
            if feature_name not in processed:
                # Skip features not in input
                continue

            value = processed[feature_name]

            for preprocessor_name in preprocessor_chain:
                preprocessor = self._preprocessors_registry.get_preprocessor(
                    preprocessor_name
                )

                # Apply preprocessor - returns (output_dict, should_remove_original)
                result, should_remove = preprocessor([value])

                # Handle remove_origin flag
                if should_remove and feature_name in processed:
                    del processed[feature_name]

                # Merge preprocessor output into processed features
                processed.update(result)

                # Update value for next preprocessor in chain
                # (first output feature becomes input to next)
                if result:
                    value = next(iter(result.values()))

        return processed

    def train_predictor_with_pipeline(
        self,
        features_list: list[dict[str, Any]],
        prediction_type: str,
        config: dict[str, Any] | None = None,
        preprocess_config: dict[str, list[str]] | None = None,
    ) -> BasePredictor:
        """Train predictor with preprocessing pipeline.

        Applies the preprocessing pipeline to each training sample before
        training the predictor.

        Args:
            features_list: Training data with runtime_ms field.
            prediction_type: Type of predictor to train.
            config: Optional training configuration.
            preprocess_config: Per-feature preprocessor chains.
                Format: {"feature_name": ["preprocessor_0", "preprocessor_1", ...]}

        Returns:
            Trained predictor instance.

        Raises:
            ValidationError: If training data is invalid or insufficient.
            TrainingError: If training fails.
        """
        if not preprocess_config:
            return self.train_predictor(features_list, prediction_type, config)

        # Apply preprocessing to each sample
        processed_list = []
        for sample in features_list:
            # Separate runtime_ms from features
            runtime_ms = sample.get("runtime_ms")
            features = {k: v for k, v in sample.items() if k != "runtime_ms"}

            # Apply preprocessing pipeline
            processed_features = self.apply_preprocess_pipeline(
                features, preprocess_config
            )

            # Recombine with runtime_ms
            processed_sample = {**processed_features, "runtime_ms": runtime_ms}
            processed_list.append(processed_sample)

        return self.train_predictor(processed_list, prediction_type, config)

    def predict_with_pipeline(
        self,
        predictor: BasePredictor,
        features: dict[str, Any],
        preprocess_config: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Make prediction with preprocessing pipeline.

        Applies the preprocessing pipeline to features before making
        a prediction with the predictor.

        Args:
            predictor: Trained predictor instance.
            features: Raw feature dictionary.
            preprocess_config: Per-feature preprocessor chains.
                Format: {"feature_name": ["preprocessor_0", "preprocessor_1", ...]}

        Returns:
            Prediction result dictionary.

        Raises:
            ValidationError: If required features are missing after preprocessing.
            PredictionError: If prediction fails.
        """
        if preprocess_config:
            features = self.apply_preprocess_pipeline(features, preprocess_config)

        return self.predict_with_predictor(predictor, features)

    # -------------------------------------------------------------------------
    # V2 Preprocessing Pipeline API
    # -------------------------------------------------------------------------

    def apply_preprocess_pipeline_v2(
        self,
        features: dict[str, Any],
        chain: PreprocessorChainV2 | None,
    ) -> dict[str, Any]:
        """Apply V2 preprocessing chain to features.

        Args:
            features: Raw feature dictionary.
            chain: V2 preprocessing chain. If None, returns features unchanged.

        Returns:
            Processed features dictionary.

        Example:
            >>> chain = (PreprocessorChainV2(name="pipeline")
            ...     .add(MultiplyPreprocessor("w", "h", "pixels"))
            ...     .add(RemoveFeaturePreprocessor(["w", "h"])))
            >>> result = low.apply_preprocess_pipeline_v2(
            ...     {"w": 10, "h": 20}, chain)
        """
        if chain is None:
            return features.copy()

        return chain.transform(features)

    def train_predictor_with_pipeline_v2(
        self,
        features_list: list[dict[str, Any]],
        prediction_type: str,
        config: dict[str, Any] | None = None,
        chain: PreprocessorChainV2 | None = None,
    ) -> BasePredictor:
        """Train predictor with V2 preprocessing chain.

        Applies the V2 chain to each training sample before training.

        Args:
            features_list: Training data with runtime_ms field.
            prediction_type: Type of predictor to train.
            config: Optional training configuration.
            chain: V2 preprocessing chain. If None, trains without preprocessing.

        Returns:
            Trained predictor instance.

        Raises:
            ValidationError: If training data is invalid or insufficient.
            TrainingError: If training fails.
        """
        if chain is None:
            return self.train_predictor(features_list, prediction_type, config)

        # Apply V2 chain to each sample
        processed_list = []
        for sample in features_list:
            # Separate runtime_ms from features
            runtime_ms = sample.get("runtime_ms")
            features = {k: v for k, v in sample.items() if k != "runtime_ms"}

            # Apply V2 chain
            processed_features = chain.transform(features)

            # Recombine with runtime_ms
            processed_sample = {**processed_features, "runtime_ms": runtime_ms}
            processed_list.append(processed_sample)

        return self.train_predictor(processed_list, prediction_type, config)

    def predict_with_pipeline_v2(
        self,
        predictor: BasePredictor,
        features: dict[str, Any],
        chain: PreprocessorChainV2 | None = None,
    ) -> dict[str, Any]:
        """Make prediction with V2 preprocessing chain.

        Args:
            predictor: Trained predictor instance.
            features: Raw feature dictionary.
            chain: V2 preprocessing chain. If None, predicts without preprocessing.

        Returns:
            Prediction result dictionary.

        Raises:
            ValidationError: If required features are missing after preprocessing.
            PredictionError: If prediction fails.
        """
        if chain is not None:
            features = chain.transform(features)

        return self.predict_with_predictor(predictor, features)


# =============================================================================
# PredictorCore - High-Level API
# =============================================================================


class PredictorCore:
    """High-level API for prediction and training.

    This class provides a simple interface using the accumulator pattern
    for training data collection. Use this for straightforward workflows.

    The accumulator pattern:
    1. Call `collect()` multiple times to accumulate training samples
    2. Call `train()` to train on accumulated data
    3. Call `predict()` to make predictions

    Attributes:
        low_level: Underlying PredictorLowLevel instance.

    Example:
        >>> core = PredictorCore()
        >>> # Collect samples
        >>> for i in range(10):
        ...     core.collect("model", platform, "quantile", {"x": i}, i*10)
        >>> # Train
        >>> result = core.train("model", platform, "quantile")
        >>> # Predict
        >>> pred = core.predict("model", platform, "quantile", {"x": 5})
    """

    def __init__(
        self,
        low_level: PredictorLowLevel | None = None,
        storage_dir: str | None = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize high-level predictor API.

        Args:
            low_level: Low-level API instance. Creates new if None.
            storage_dir: Override storage directory (only if low_level is None).
            max_workers: Max workers for async operations.
        """
        if low_level is not None:
            self._low_level = low_level
        else:
            self._low_level = PredictorLowLevel(storage_dir=storage_dir)

        self._accumulated: dict[str, list[CollectedSample]] = {}
        self._feature_schemas: dict[str, list[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # Accumulator Pattern
    # -------------------------------------------------------------------------

    def collect(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        features: dict[str, Any],
        runtime_ms: float,
    ) -> None:
        """Add a sample to the accumulator.

        The first sample collected defines the feature schema.
        Subsequent samples must have the same features.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            features: Feature dictionary (without runtime_ms).
            runtime_ms: Measured runtime in milliseconds.

        Raises:
            ValidationError: If features are inconsistent with schema.
        """
        key = self._make_accumulator_key(model_id, platform_info, prediction_type)

        with self._lock:
            # First sample defines feature schema
            if key not in self._feature_schemas:
                self._feature_schemas[key] = sorted(features.keys())
                self._accumulated[key] = []

            # Validate feature consistency
            expected_features = set(self._feature_schemas[key])
            provided_features = set(features.keys())

            if provided_features != expected_features:
                missing = expected_features - provided_features
                extra = provided_features - expected_features
                raise ValidationError(
                    f"Sample has inconsistent features. "
                    f"Expected: {sorted(expected_features)}, "
                    f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
                )

            # Add sample
            sample = CollectedSample(
                features=features,
                runtime_ms=runtime_ms,
            )
            self._accumulated[key].append(sample)

    def get_collected_count(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> int:
        """Get number of collected samples for a model.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            Number of collected samples.
        """
        key = self._make_accumulator_key(model_id, platform_info, prediction_type)
        with self._lock:
            return len(self._accumulated.get(key, []))

    def clear_collected(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> None:
        """Clear accumulated samples for a model.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
        """
        key = self._make_accumulator_key(model_id, platform_info, prediction_type)
        with self._lock:
            self._accumulated.pop(key, None)
            self._feature_schemas.pop(key, None)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        config: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Train on accumulated data.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            config: Optional training configuration.

        Returns:
            TrainingResult with training metadata.

        Raises:
            ValidationError: If no data collected or insufficient samples.
            TrainingError: If training fails.
        """
        key = self._make_accumulator_key(model_id, platform_info, prediction_type)

        with self._lock:
            samples = self._accumulated.get(key, [])

            # Validate data exists
            if not samples:
                raise ValidationError(
                    f"No samples collected for model '{model_id}'. "
                    f"Use collect() to add training data first."
                )

            # Validate minimum samples
            if len(samples) < MIN_TRAINING_SAMPLES:
                raise ValidationError(
                    f"Insufficient training data. Got {len(samples)} samples, "
                    f"but minimum {MIN_TRAINING_SAMPLES} samples required."
                )

            # Convert to features_list format
            features_list = [
                {**sample.features, "runtime_ms": sample.runtime_ms}
                for sample in samples
            ]

        # Train using low-level API
        try:
            predictor = self._low_level.train_predictor(
                features_list=features_list,
                prediction_type=prediction_type,
                config=config,
            )

            # Save model and get version
            version = self._low_level.save_model(
                model_id=model_id,
                platform_info=platform_info,
                prediction_type=prediction_type,
                predictor=predictor,
            )

            # Clear accumulator after successful training
            with self._lock:
                self._accumulated.pop(key, None)
                self._feature_schemas.pop(key, None)

            # Get version ISO string
            from datetime import timezone
            version_iso = datetime.fromtimestamp(version, tz=timezone.utc).isoformat()

            return TrainingResult(
                success=True,
                model_id=model_id,
                platform_info=platform_info,
                prediction_type=prediction_type,
                samples_trained=len(features_list),
                training_metadata={
                    **predictor.get_model_state(),
                    "version": version,
                    "version_iso": version_iso,
                },
                message=f"Successfully trained model with {len(features_list)} samples (version {version})",
            )

        except (ValidationError, TrainingError):
            raise
        except Exception as e:
            raise TrainingError(f"Training failed: {e}") from e

    async def train_async(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        config: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Train on accumulated data asynchronously.

        Non-blocking training suitable for large datasets.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            config: Optional training configuration.

        Returns:
            TrainingResult with training metadata.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.train(
                model_id=model_id,
                platform_info=platform_info,
                prediction_type=prediction_type,
                config=config,
            ),
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        features: dict[str, Any],
    ) -> PredictionResult:
        """Make a single prediction.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            features: Feature dictionary.

        Returns:
            PredictionResult with prediction values.

        Raises:
            ModelNotFoundError: If model does not exist.
            ValidationError: If features are invalid.
            PredictionError: If prediction fails.
        """
        # Load predictor
        predictor = self._low_level.load_model(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
        )

        # Make prediction
        result = self._low_level.predict_with_predictor(predictor, features)

        return PredictionResult(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
            result=result,
        )

    def batch_predict(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        features_list: list[dict[str, Any]],
    ) -> list[PredictionResult]:
        """Make multiple predictions.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            features_list: List of feature dictionaries.

        Returns:
            List of PredictionResult objects.
        """
        # Load predictor once
        predictor = self._low_level.load_model(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
        )

        # Make predictions
        results = []
        for features in features_list:
            result = self._low_level.predict_with_predictor(predictor, features)
            results.append(
                PredictionResult(
                    model_id=model_id,
                    platform_info=platform_info,
                    prediction_type=prediction_type,
                    result=result,
                )
            )

        return results

    # -------------------------------------------------------------------------
    # Inference Pipeline
    # -------------------------------------------------------------------------

    def inference_pipeline(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        features: dict[str, Any],
        preprocess_config: dict[str, list[str]] | None = None,
    ) -> PredictionResult:
        """Define and execute an inference pipeline.

        Combines model loading, preprocessing, and prediction into a single
        operation. Use this for streamlined inference workflows.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            features: Raw feature dictionary.
            preprocess_config: Per-feature preprocessor chains.
                Format: {"feature_name": ["preprocessor_0", "preprocessor_1", ...]}
                Each feature is processed by its chain in order.

        Returns:
            PredictionResult with prediction values.

        Raises:
            ModelNotFoundError: If model does not exist.
            ValidationError: If features are invalid.
            PredictionError: If prediction fails.

        Example:
            >>> result = core.inference_pipeline(
            ...     model_id="my_model",
            ...     platform_info=platform,
            ...     prediction_type="quantile",
            ...     features={"sentence": "Hello world", "batch_size": 32},
            ...     preprocess_config={
            ...         "sentence": ["tokenizer", "semantic_encoder"],
            ...         # batch_size has no preprocessors - passed through as-is
            ...     },
            ... )
        """
        # Load predictor
        predictor = self._low_level.load_model(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
        )

        # Make prediction with pipeline preprocessing
        result = self._low_level.predict_with_pipeline(
            predictor=predictor,
            features=features,
            preprocess_config=preprocess_config,
        )

        return PredictionResult(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
            result=result,
        )

    def inference_pipeline_v2(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        features: dict[str, Any],
        chain: PreprocessorChainV2 | None = None,
    ) -> PredictionResult:
        """Execute inference pipeline with V2 preprocessing chain.

        Loads model, applies V2 chain, and makes prediction in a single
        operation. Use this for streamlined V2 inference workflows.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            features: Raw feature dictionary.
            chain: V2 preprocessing chain. If None, predicts without preprocessing.

        Returns:
            PredictionResult with prediction values.

        Raises:
            ModelNotFoundError: If model does not exist.
            ValidationError: If features are invalid.
            PredictionError: If prediction fails.

        Example:
            >>> chain = (PreprocessorChainV2(name="pipeline")
            ...     .add(MultiplyPreprocessor("w", "h", "pixels")))
            >>> result = core.inference_pipeline_v2(
            ...     model_id="my_model",
            ...     platform_info=platform,
            ...     prediction_type="quantile",
            ...     features={"w": 10, "h": 20, "batch_size": 32},
            ...     chain=chain,
            ... )
        """
        # Load predictor
        predictor = self._low_level.load_model(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
        )

        # Make prediction with V2 pipeline preprocessing
        result = self._low_level.predict_with_pipeline_v2(
            predictor=predictor,
            features=features,
            chain=chain,
        )

        return PredictionResult(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
            result=result,
        )

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _make_accumulator_key(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> str:
        """Generate key for accumulator dictionary.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            Accumulator key string.
        """
        return (
            f"{model_id}__"
            f"{platform_info.software_name}-{platform_info.software_version}__"
            f"{platform_info.hardware_name}__"
            f"{prediction_type}"
        )

    def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        self._low_level.clear_cache()
