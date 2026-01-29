"""Training client using direct library imports.

Shares ModelStorage and ModelCache instances with PredictorClient
so that trained models are immediately visible to predictions.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from src.clients._predictor_lib import (
    PREDICTOR_CLASSES,
    ModelCache,
    ModelStorage,
    PreprocessorsRegistry,
)
from src.clients.models import TrainingSample


class TrainingClient:
    """Training client using direct library imports.

    Shares ModelStorage and ModelCache with PredictorClient
    so that trained models are immediately visible to predictions.
    """

    def __init__(
        self,
        storage: ModelStorage,
        cache: ModelCache,
        batch_size: int = 100,
        min_samples: int = 10,
        prediction_types: list[str] | None = None,
    ):
        """Initialize training client.

        Args:
            storage: Shared ModelStorage instance.
            cache: Shared ModelCache instance.
            batch_size: Samples to batch before training.
            min_samples: Minimum samples required for training.
            prediction_types: Prediction types to train.
        """
        self._storage = storage
        self._cache = cache
        self._preprocessors_registry = PreprocessorsRegistry()
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.prediction_types = prediction_types or [
            "expect_error",
            "quantile",
        ]

        self._samples_buffer: list[TrainingSample] = []

        logger.info(
            f"TrainingClient initialized "
            f"(batch_size={batch_size}, min_samples={min_samples}, "
            f"types={self.prediction_types})"
        )

    def add_sample(
        self,
        model_id: str,
        platform_info: dict[str, str],
        features: dict[str, Any],
        actual_runtime_ms: float,
    ) -> None:
        """Add a training sample to the buffer.

        Args:
            model_id: Model/tool identifier.
            platform_info: Platform information.
            features: Task features.
            actual_runtime_ms: Actual execution time in ms.
        """
        sample = TrainingSample(
            model_id=model_id,
            platform_info=platform_info,
            features=features,
            actual_runtime_ms=actual_runtime_ms,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._samples_buffer.append(sample)

        logger.debug(
            f"Added training sample for {model_id} on "
            f"{platform_info.get('hardware_name', '?')}: "
            f"{actual_runtime_ms:.2f}ms "
            f"(buffer size: {len(self._samples_buffer)})"
        )

    async def flush_if_ready(self) -> bool:
        """Flush buffer to train models if batch size reached.

        Returns:
            True if training was triggered, False otherwise.
        """
        if len(self._samples_buffer) >= self.batch_size:
            return await self.flush()
        return False

    async def flush(self, force: bool = False) -> bool:
        """Train models with all buffered samples.

        Args:
            force: If True, train even below min_samples threshold.

        Returns:
            True if all training succeeded, False otherwise.
        """
        if not self._samples_buffer:
            logger.debug("No training samples to flush")
            return False

        if not force and len(self._samples_buffer) < self.min_samples:
            logger.debug(
                f"Buffer has {len(self._samples_buffer)} samples, "
                f"below minimum of {self.min_samples}. Skipping training."
            )
            return False

        # Group samples by (model_id, platform_info)
        grouped_samples: dict[tuple[str, str], list[TrainingSample]] = defaultdict(list)
        for sample in self._samples_buffer:
            platform_key = json.dumps(sample.platform_info, sort_keys=True)
            key = (sample.model_id, platform_key)
            grouped_samples[key].append(sample)

        logger.info(
            f"Flushing {len(self._samples_buffer)} training samples "
            f"across {len(grouped_samples)} model-platform combinations"
        )

        success_count = 0
        failure_count = 0

        for (model_id, _platform_key), samples in grouped_samples.items():
            platform_info = samples[0].platform_info

            # Prepare features_list with runtime_ms included
            features_list = [
                {
                    **s.features,
                    "runtime_ms": s.actual_runtime_ms,
                }
                for s in samples
            ]

            for prediction_type in self.prediction_types:
                try:
                    self._train_model(
                        model_id=model_id,
                        platform_info=platform_info,
                        prediction_type=prediction_type,
                        features_list=features_list,
                    )
                    success_count += 1

                except Exception as e:
                    logger.error(
                        f"Failed to train {model_id} ({prediction_type}) "
                        f"on {platform_info.get('hardware_name', '?')}: {e}",
                        exc_info=True,
                    )
                    failure_count += 1

        self._samples_buffer.clear()

        logger.info(
            f"Training complete: {success_count} succeeded, " f"{failure_count} failed"
        )

        return failure_count == 0

    def _train_model(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str,
        features_list: list[dict[str, Any]],
    ) -> None:
        """Train a single model and save to storage.

        Args:
            model_id: Model identifier.
            platform_info: Platform info dict.
            prediction_type: Type of prediction model.
            features_list: Training data with runtime_ms.

        Raises:
            ValueError: If prediction type is invalid.
            Exception: On training or storage failure.
        """
        cls = PREDICTOR_CLASSES.get(prediction_type)
        if cls is None:
            raise ValueError(f"Invalid prediction type: {prediction_type}")

        logger.debug(
            f"Training {model_id} ({prediction_type}) on "
            f"{platform_info.get('hardware_name', '?')} "
            f"with {len(features_list)} samples"
        )

        predictor = cls()
        predictor.train(features_list=features_list, config={})

        # Save model
        model_key = self._storage.generate_model_key(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
        )

        predictor_state = predictor.get_model_state()
        metadata = {
            "model_id": model_id,
            "platform_info": platform_info,
            "prediction_type": prediction_type,
            "samples_count": len(features_list),
            "training_config": {},
        }
        self._storage.save_model(model_key, predictor_state, metadata)

        # Invalidate cache so predictor picks up new model
        self._cache.invalidate(model_key)

        logger.info(
            f"Successfully trained {model_id} ({prediction_type}) on "
            f"{platform_info.get('hardware_name', '?')} "
            f"with {len(features_list)} samples"
        )

    def get_buffer_size(self) -> int:
        """Get current number of buffered training samples."""
        return len(self._samples_buffer)

    def clear_buffer(self) -> None:
        """Clear all buffered training samples without training."""
        count = len(self._samples_buffer)
        self._samples_buffer.clear()
        logger.info(f"Cleared {count} training samples from buffer")

    async def close(self) -> None:
        """Close the training client (no-op for library mode)."""
        logger.info("TrainingClient closed")
