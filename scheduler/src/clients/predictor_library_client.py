"""Predictor client using PredictorLowLevel two-layer API.

Uses PredictorLowLevel for model management and V2 PreprocessorChainV2
for config-driven preprocessing. All preprocessor logic is delegated
to the PreprocessorChainBuilder.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from src.clients._predictor_lib import (
    PlatformInfo,
    PredictorLowLevel,
    generate_experiment_prediction,
    is_experiment_mode,
)
from src.clients.models import Prediction
from src.clients.preprocessor_config import PreprocessorChainBuilder
from src.config import PreprocessorConfig

if TYPE_CHECKING:
    from src.model import Instance


class PredictorClient:
    """Predictor client using PredictorLowLevel two-layer API.

    Delegates model management (load, save, cache) to PredictorLowLevel
    and preprocessor chain building to PreprocessorChainBuilder.
    """

    def __init__(
        self,
        storage_dir: str = "models",
        cache_max_size: int = 100,
        preprocessor_config: PreprocessorConfig | None = None,
    ):
        """Initialize predictor client.

        Args:
            storage_dir: Directory for model storage.
            cache_max_size: Maximum number of models to cache.
            preprocessor_config: Preprocessor pipeline configuration.
                If None, no preprocessors are applied.

        Raises:
            RuntimeError: If preprocessor_config has strict=True and
                a configured preprocessor model is unavailable.
        """
        self._low_level = PredictorLowLevel(
            storage_dir=storage_dir,
            cache_size=cache_max_size,
        )

        if preprocessor_config is None:
            preprocessor_config = PreprocessorConfig()

        self._chain_builder = PreprocessorChainBuilder(
            config_file=preprocessor_config.config_file,
            registry_v1=self._low_level._preprocessors_registry,
            strict=preprocessor_config.strict_validation,
        )

        logger.info(
            f"PredictorClient initialized "
            f"(storage_dir={storage_dir}, "
            f"cache_max_size={cache_max_size}, "
            f"preprocessor_rules={self._chain_builder.has_rules})"
        )

    def _predict_single_platform(
        self,
        model_id: str,
        metadata: dict[str, Any],
        platform_info_dict: dict[str, str],
        prediction_type: str,
        quantiles_list: list[float] | None,
        instance_ids: list[str],
    ) -> list[Prediction]:
        """Run prediction for a single platform, returning Predictions.

        Args:
            model_id: Model identifier.
            metadata: Task features.
            platform_info_dict: Platform info as dict.
            prediction_type: Prediction type.
            quantiles_list: Optional custom quantiles.
            instance_ids: IDs of all instances sharing this platform.

        Returns:
            List of Prediction objects for each instance.

        Raises:
            ValueError: On model-not-found or invalid features.
        """
        random.seed(42)
        np.random.seed(42)

        # Check experiment mode
        if is_experiment_mode(metadata, platform_info_dict):
            config = {}
            if quantiles_list is not None:
                config["quantiles"] = quantiles_list

            result = generate_experiment_prediction(
                prediction_type=prediction_type,
                features=metadata,
                config=config,
            )

            return self._parse_result(
                result,
                prediction_type,
                instance_ids,
                platform_info_dict,
            )

        # Normal mode: load model via PredictorLowLevel
        platform_model = PlatformInfo(**platform_info_dict)
        predictor = self._low_level.load_model(
            model_id=model_id,
            platform_info=platform_model,
            prediction_type=prediction_type,
        )

        # Build feature set with hardware info
        hardware_features = platform_model.extract_gpu_specs()
        all_features = metadata.copy()
        if hardware_features:
            all_features.update(hardware_features)

        # Apply V2 preprocessor chain (config-driven)
        chain = self._chain_builder.get_chain(model_id)
        result = self._low_level.predict_with_pipeline_v2(
            predictor=predictor,
            features=all_features,
            chain=chain,
        )

        return self._parse_result(
            result,
            prediction_type,
            instance_ids,
            platform_info_dict,
        )

    def _parse_result(
        self,
        result: dict[str, Any],
        prediction_type: str,
        instance_ids: list[str],
        platform_info_dict: dict[str, str],
    ) -> list[Prediction]:
        """Parse predictor result into Prediction objects.

        Args:
            result: Raw result dict from predictor.
            prediction_type: Type of prediction.
            instance_ids: Instance IDs to replicate for.
            platform_info_dict: Platform info for logging.

        Returns:
            List of Prediction objects.
        """
        predictions = []

        if prediction_type == "expect_error":
            predicted_time = result["expected_runtime_ms"]
            error_margin = result["error_margin_ms"]

            logger.info(
                f"Prediction (expect_error) for "
                f"{platform_info_dict.get('hardware_name', '?')}: "
                f"expected_runtime={predicted_time:.2f}ms, "
                f"error_margin={error_margin:.2f}ms "
                f"({len(instance_ids)} instances)"
            )

            for iid in instance_ids:
                predictions.append(
                    Prediction(
                        instance_id=iid,
                        predicted_time_ms=predicted_time,
                        confidence=None,
                        quantiles=None,
                        error_margin_ms=error_margin,
                    )
                )

        elif prediction_type == "quantile":
            quantiles_dict = result["quantiles"]
            quantiles = {float(k): v for k, v in quantiles_dict.items()}
            median = quantiles.get(0.5, next(iter(quantiles.values())))

            quantile_str = ", ".join(
                f"{k}={v:.2f}ms" for k, v in sorted(quantiles.items())
            )
            logger.info(
                f"Prediction (quantile) for "
                f"{platform_info_dict.get('hardware_name', '?')}: "
                f"quantiles={{{quantile_str}}} "
                f"({len(instance_ids)} instances)"
            )

            for iid in instance_ids:
                predictions.append(
                    Prediction(
                        instance_id=iid,
                        predicted_time_ms=median,
                        confidence=None,
                        quantiles=quantiles,
                    )
                )

        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        return predictions

    async def predict(
        self,
        model_id: str,
        metadata: dict[str, Any],
        instances: list[Instance],
        prediction_type: str = "quantile",
        quantiles: list[float] | None = None,
    ) -> list[Prediction]:
        """Get predictions for task execution time on multiple instances.

        Platform-based batching: instances sharing the same platform_info
        get a single prediction call, with results replicated.

        Args:
            model_id: Model/tool ID.
            metadata: Task metadata for prediction.
            instances: List of Instance objects.
            prediction_type: "expect_error" or "quantile".
            quantiles: Optional quantile levels.

        Returns:
            List of predictions for each instance.

        Raises:
            ValueError: Model not found or invalid features.
        """
        # Group instances by platform_info
        platform_to_instances: dict[str, list[Instance]] = defaultdict(list)
        for instance in instances:
            platform_key = json.dumps(instance.platform_info, sort_keys=True)
            platform_to_instances[platform_key].append(instance)

        logger.debug(
            f"Batching predictions: {len(instances)} instances across "
            f"{len(platform_to_instances)} unique platforms"
        )

        predictions: list[Prediction] = []

        for _platform_key, platform_instances in platform_to_instances.items():
            representative = platform_instances[0]
            instance_ids = [inst.instance_id for inst in platform_instances]

            try:
                logger.debug(
                    f"Requesting prediction for platform "
                    f"{representative.platform_info} "
                    f"(covers {len(platform_instances)} instances)"
                )

                batch_predictions = self._predict_single_platform(
                    model_id=model_id,
                    metadata=metadata,
                    platform_info_dict=representative.platform_info,
                    prediction_type=prediction_type,
                    quantiles_list=quantiles,
                    instance_ids=instance_ids,
                )
                predictions.extend(batch_predictions)

            except ValueError as e:
                error_msg = str(e)
                platform_info = representative.platform_info

                if "Model not found" in error_msg:
                    logger.error(
                        f"No trained model for {model_id} on platform "
                        f"{platform_info}. Task submission rejected."
                    )
                    raise ValueError(
                        f"No trained model for {model_id} on platform "
                        f"{platform_info.get('software_name', '?')}/"
                        f"{platform_info.get('hardware_name', '?')}. "
                        f"Please train the model first or use "
                        f"experiment mode."
                    ) from e
                elif "Invalid features" in error_msg or "Invalid request" in error_msg:
                    logger.error(f"Invalid features for prediction: {error_msg}")
                    raise ValueError(f"Invalid task metadata: {error_msg}") from e
                else:
                    logger.error(f"Prediction validation error: {error_msg}")
                    raise

        return predictions

    async def health_check(self) -> bool:
        """Check if predictor is healthy.

        Returns:
            Always True for library mode (no network dependency).
        """
        return True

    async def close(self) -> None:
        """Close the predictor client and cleanup resources."""
        self._low_level.clear_cache()
        logger.info("PredictorClient closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
