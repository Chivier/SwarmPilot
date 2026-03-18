"""Library-style API for the predictor module.

This module provides a two-level API for using the predictor without HTTP:

- **PredictorLowLevel**: Full control over model management, storage, and cache.
- **PredictorCore**: Simple high-level API with accumulator pattern for training.

Adapted for non-versioned storage (Agnetify branch).

Example:
    >>> from swarmpilot.predictor.api.core import PredictorCore
    >>> from swarmpilot.predictor.models import PlatformInfo
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

from swarmpilot.errors import (
    ModelNotFoundError,
    PredictionError,
    PredictorValidationError as ValidationError,
    TrainingError,
)
from swarmpilot.predictor.api.cache import ModelCache
from swarmpilot.predictor.models import (
    CollectedSample,
    ModelInfo,
    PlatformInfo,
    PredictionResult,
    TrainingResult,
)
from swarmpilot.predictor.predictor.base import BasePredictor
from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor
from swarmpilot.predictor.preprocessor.chain_v2 import PreprocessorChainV2
from swarmpilot.predictor.preprocessor.preprocessors_registry import (
    PreprocessorsRegistry,
)
from swarmpilot.predictor.storage.model_storage import ModelStorage
from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()

# =============================================================================
# Constants
# =============================================================================


PREDICTOR_CLASSES: dict[str, type[BasePredictor]] = {
    "expect_error": ExpectErrorPredictor,
    "quantile": QuantilePredictor,
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

    Adapted for non-versioned ModelStorage (Agnetify branch).

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
        storage_dir: str = "models",
        cache_size: int = 100,
    ) -> None:
        """Initialize low-level predictor API.

        Args:
            storage_dir: Directory for model storage.
            cache_size: Maximum number of models to cache in memory.
        """
        self._storage = ModelStorage(storage_dir=storage_dir)
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
        if prediction_type not in PREDICTOR_CLASSES:
            raise ValidationError(
                f"Invalid prediction_type: {prediction_type}. "
                f"Must be one of: {list(PREDICTOR_CLASSES.keys())}"
            )

        if len(features_list) < MIN_TRAINING_SAMPLES:
            raise ValidationError(
                f"Insufficient training data. Got {len(features_list)} samples, "
                f"but minimum {MIN_TRAINING_SAMPLES} samples required."
            )

        predictor_class = PREDICTOR_CLASSES[prediction_type]
        predictor = predictor_class()

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
    ) -> None:
        """Save a predictor to non-versioned storage.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            predictor: Trained predictor instance.
            samples_count: Number of samples used for training.
            extra_metadata: Additional metadata to store.
        """
        predictor_state = predictor.get_model_state()

        actual_samples_count = samples_count
        if actual_samples_count is None:
            actual_samples_count = predictor_state.get("samples_count", 0)

        metadata = {
            "model_id": model_id,
            "platform_info": platform_info.model_dump(),
            "prediction_type": prediction_type,
            "samples_count": actual_samples_count,
            "feature_names": predictor_state.get("feature_names", []),
            "saved_at": datetime.now().isoformat(),
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        model_key = self.generate_model_key(
            model_id, platform_info, prediction_type
        )

        with self._lock:
            self._storage.save_model(model_key, predictor_state, metadata)
            self._cache.invalidate(model_key)

        logger.info(f"Saved model {model_id} as {model_key}")

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
        model_key = self.generate_model_key(
            model_id, platform_info, prediction_type
        )

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

            if prediction_type not in PREDICTOR_CLASSES:
                raise ValidationError(
                    f"Invalid prediction_type: {prediction_type}"
                )

            predictor_class = PREDICTOR_CLASSES[prediction_type]
            predictor = predictor_class()
            predictor.load_model_state(model_data["predictor_state"])

            self._cache.put(model_key, predictor, prediction_type)

            logger.debug(f"Loaded model: {model_key}")
            return predictor

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
        model_key = self.generate_model_key(
            model_id, platform_info, prediction_type
        )
        return self._storage.model_exists(model_key)

    def get_model_info(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
    ) -> ModelInfo:
        """Get metadata about a stored model.

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.

        Returns:
            ModelInfo with model metadata.

        Raises:
            ModelNotFoundError: If model does not exist.
        """
        model_key = self.generate_model_key(
            model_id, platform_info, prediction_type
        )
        model_data = self._storage.load_model(model_key)
        if model_data is None:
            raise ModelNotFoundError(f"Model not found: {model_key}")

        metadata = model_data.get("metadata", {})
        return ModelInfo(
            model_id=metadata.get("model_id", model_id),
            platform_info=platform_info,
            prediction_type=metadata.get("prediction_type", prediction_type),
            samples_count=metadata.get("samples_count", 0),
            last_trained=model_data.get("saved_at", ""),
            feature_names=metadata.get("feature_names"),
        )

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
        expected_features = predictor.feature_names

        if expected_features is None:
            raise PredictionError(
                "Predictor has no feature_names - not trained?"
            )

        filtered = predictor.filter_features_for_prediction(
            features, expected_features
        )

        missing = set(expected_features) - set(filtered.keys())
        if missing:
            raise ValidationError(
                f"Missing required features: {sorted(missing)}. "
                f"Model expects: {sorted(expected_features)}"
            )

        extra = set(features.keys()) - set(expected_features)
        if extra:
            logger.debug(f"Ignoring extra features: {sorted(extra)}")

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
        model_key = self.generate_model_key(
            model_id, platform_info, prediction_type
        )
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
    # V1 Inference Pipeline API
    # -------------------------------------------------------------------------

    def apply_preprocess_pipeline(
        self,
        features: dict[str, Any],
        preprocess_config: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Apply per-feature preprocessing chain (V1).

        Args:
            features: Raw feature dictionary.
            preprocess_config: Maps feature_name to ordered list of
                preprocessor names.

        Returns:
            Processed features dictionary.
        """
        if not preprocess_config:
            return features.copy()

        processed = features.copy()

        for feature_name, preprocessor_chain in preprocess_config.items():
            if feature_name not in processed:
                continue

            value = processed[feature_name]

            for preprocessor_name in preprocessor_chain:
                preprocessor = self._preprocessors_registry.get_preprocessor(
                    preprocessor_name
                )

                result, should_remove = preprocessor([value])

                if should_remove and feature_name in processed:
                    del processed[feature_name]

                processed.update(result)

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
        """Train predictor with V1 preprocessing pipeline.

        Args:
            features_list: Training data with runtime_ms field.
            prediction_type: Type of predictor to train.
            config: Optional training configuration.
            preprocess_config: Per-feature preprocessor chains.

        Returns:
            Trained predictor instance.
        """
        if not preprocess_config:
            return self.train_predictor(features_list, prediction_type, config)

        processed_list = []
        for sample in features_list:
            runtime_ms = sample.get("runtime_ms")
            features = {k: v for k, v in sample.items() if k != "runtime_ms"}

            processed_features = self.apply_preprocess_pipeline(
                features, preprocess_config
            )

            processed_sample = {
                **processed_features,
                "runtime_ms": runtime_ms,
            }
            processed_list.append(processed_sample)

        return self.train_predictor(processed_list, prediction_type, config)

    def predict_with_pipeline(
        self,
        predictor: BasePredictor,
        features: dict[str, Any],
        preprocess_config: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Make prediction with V1 preprocessing pipeline.

        Args:
            predictor: Trained predictor instance.
            features: Raw feature dictionary.
            preprocess_config: Per-feature preprocessor chains.

        Returns:
            Prediction result dictionary.
        """
        if preprocess_config:
            features = self.apply_preprocess_pipeline(
                features, preprocess_config
            )

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
            chain: V2 preprocessing chain. If None, returns copy.

        Returns:
            Processed features dictionary.
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

        Args:
            features_list: Training data with runtime_ms field.
            prediction_type: Type of predictor to train.
            config: Optional training configuration.
            chain: V2 preprocessing chain. If None, no preprocessing.

        Returns:
            Trained predictor instance.
        """
        if chain is None:
            return self.train_predictor(features_list, prediction_type, config)

        processed_list = []
        for sample in features_list:
            runtime_ms = sample.get("runtime_ms")
            features = {k: v for k, v in sample.items() if k != "runtime_ms"}

            processed_features = chain.transform(features)

            processed_sample = {
                **processed_features,
                "runtime_ms": runtime_ms,
            }
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
            chain: V2 preprocessing chain. If None, no preprocessing.

        Returns:
            Prediction result dictionary.
        """
        if chain is not None:
            features = chain.transform(features)

        return self.predict_with_predictor(predictor, features)


# =============================================================================
# PredictorCore - High-Level API
# =============================================================================


class PredictorCore:
    """High-level API for prediction and training.

    Uses the accumulator pattern for training data collection.

    Example:
        >>> core = PredictorCore()
        >>> for i in range(10):
        ...     core.collect("model", platform, "quantile", {"x": i}, i*10)
        >>> result = core.train("model", platform, "quantile")
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
            storage_dir: Override storage directory.
            max_workers: Max workers for async operations.
        """
        if low_level is not None:
            self._low_level = low_level
        else:
            self._low_level = PredictorLowLevel(
                storage_dir=storage_dir or "models"
            )

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

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            features: Feature dictionary (without runtime_ms).
            runtime_ms: Measured runtime in milliseconds.

        Raises:
            ValidationError: If features are inconsistent with schema.
        """
        key = self._make_accumulator_key(
            model_id, platform_info, prediction_type
        )

        with self._lock:
            if key not in self._feature_schemas:
                self._feature_schemas[key] = sorted(features.keys())
                self._accumulated[key] = []

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
        key = self._make_accumulator_key(
            model_id, platform_info, prediction_type
        )
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
        key = self._make_accumulator_key(
            model_id, platform_info, prediction_type
        )
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
            ValidationError: If no data collected or insufficient.
            TrainingError: If training fails.
        """
        key = self._make_accumulator_key(
            model_id, platform_info, prediction_type
        )

        with self._lock:
            samples = self._accumulated.get(key, [])

            if not samples:
                raise ValidationError(
                    f"No samples collected for model '{model_id}'. "
                    f"Use collect() to add training data first."
                )

            if len(samples) < MIN_TRAINING_SAMPLES:
                raise ValidationError(
                    f"Insufficient training data. "
                    f"Got {len(samples)} samples, "
                    f"but minimum {MIN_TRAINING_SAMPLES} required."
                )

            features_list = [
                {**sample.features, "runtime_ms": sample.runtime_ms}
                for sample in samples
            ]

        try:
            predictor = self._low_level.train_predictor(
                features_list=features_list,
                prediction_type=prediction_type,
                config=config,
            )

            self._low_level.save_model(
                model_id=model_id,
                platform_info=platform_info,
                prediction_type=prediction_type,
                predictor=predictor,
            )

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
                message=(
                    f"Successfully trained model with "
                    f"{len(features_list)} samples"
                ),
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
        """
        predictor = self._low_level.load_model(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
        )

        result = self._low_level.predict_with_predictor(predictor, features)

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

        Args:
            model_id: Model identifier.
            platform_info: Platform information.
            prediction_type: Type of prediction.
            features: Raw feature dictionary.
            chain: V2 preprocessing chain.

        Returns:
            PredictionResult with prediction values.
        """
        predictor = self._low_level.load_model(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
        )

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
            f"{platform_info.software_name}-"
            f"{platform_info.software_version}__"
            f"{platform_info.hardware_name}__"
            f"{prediction_type}"
        )

    def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        self._low_level.clear_cache()
