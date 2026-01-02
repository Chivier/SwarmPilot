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
    ) -> None:
        """Save a predictor to storage.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            predictor: Trained predictor instance.
        """
        model_key = self.generate_model_key(model_id, platform_info, prediction_type)

        # Get predictor state
        predictor_state = predictor.get_model_state()

        # Prepare metadata
        metadata = {
            "model_id": model_id,
            "platform_info": platform_info.model_dump(),
            "prediction_type": prediction_type,
            "samples_count": predictor_state.get("samples_count", 0),
            "feature_names": predictor_state.get("feature_names", []),
            "saved_at": datetime.now().isoformat(),
        }

        # Save to storage
        with self._lock:
            self._storage.save_model(model_key, predictor_state, metadata)

            # Invalidate cache entry if exists
            self._cache.invalidate(model_key)

        logger.info(f"Saved model: {model_key}")

    def load_model(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> BasePredictor:
        """Load a predictor from storage.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            Loaded predictor instance.

        Raises:
            ModelNotFoundError: If model does not exist.
        """
        model_key = self.generate_model_key(model_id, platform_info, prediction_type)

        with self._lock:
            # Check cache first
            cached = self._cache.get(model_key)
            if cached is not None:
                predictor, _ = cached
                logger.debug(f"Cache hit for model: {model_key}")
                return predictor

            # Load from storage
            model_data = self._storage.load_model(model_key)
            if model_data is None:
                raise ModelNotFoundError(f"Model not found: {model_key}")

            # Create predictor instance
            if prediction_type not in PREDICTOR_CLASSES:
                raise ValidationError(f"Invalid prediction_type: {prediction_type}")

            predictor_class = PREDICTOR_CLASSES[prediction_type]
            predictor = predictor_class()
            predictor.load_model_state(model_data["predictor_state"])

            # Cache it
            self._cache.put(model_key, predictor, prediction_type)

            logger.debug(f"Loaded model from storage: {model_key}")
            return predictor

    def delete_model(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> bool:
        """Delete a model from storage.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            True if deleted, False if not found.
        """
        model_key = self.generate_model_key(model_id, platform_info, prediction_type)

        with self._lock:
            # Invalidate cache
            self._cache.invalidate(model_key)

            # Delete from storage
            result = self._storage.delete_model(model_key)

        if result:
            logger.info(f"Deleted model: {model_key}")
        return result

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
    ) -> ModelInfo | None:
        """Get detailed information about a model.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            ModelInfo if found, None otherwise.
        """
        model_key = self.generate_model_key(model_id, platform_info, prediction_type)
        model_data = self._storage.load_model(model_key)

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
        """Check if a model exists.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            True if model exists, False otherwise.
        """
        model_key = self.generate_model_key(model_id, platform_info, prediction_type)
        return self._storage.model_exists(model_key)

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

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
        """
        model_key = self.generate_model_key(model_id, platform_info, prediction_type)
        with self._lock:
            self._cache.invalidate(model_key)

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

            # Save model
            self._low_level.save_model(
                model_id=model_id,
                platform_info=platform_info,
                prediction_type=prediction_type,
                predictor=predictor,
            )

            # Clear accumulator after successful training
            with self._lock:
                self._accumulated.pop(key, None)
                self._feature_schemas.pop(key, None)

            return TrainingResult(
                success=True,
                model_id=model_id,
                platform_info=platform_info,
                prediction_type=prediction_type,
                samples_trained=len(features_list),
                training_metadata=predictor.get_model_state(),
                message=f"Successfully trained model with {len(features_list)} samples",
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
    # Inference Pipeline (TODO)
    # -------------------------------------------------------------------------

    def inference_pipeline(self, *args: Any, **kwargs: Any) -> Any:
        """Define an inference pipeline.

        TODO: Implement inference pipeline functionality.
        """
        raise NotImplementedError("inference_pipeline is not yet implemented")

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
