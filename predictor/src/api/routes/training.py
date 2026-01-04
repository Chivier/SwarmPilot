"""Training endpoint for model training.

Provides endpoints for:
- /collect: Collect individual training samples
- /train: Train model using collected + request data
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import HTTPException

from src.api import dependencies
from src.api.core import ValidationError
from src.api.routes.helpers import handle_library_exception
from src.models import CollectRequest
from src.models import CollectResponse
from src.models import TrainingRequest
from src.models import TrainingResponse
from src.utils.logging import get_logger

logger = get_logger()

router = APIRouter()


# =============================================================================
# /collect - Collect Training Samples
# =============================================================================


@router.post("/collect", response_model=CollectResponse, tags=["Training"])
async def collect_sample(request: CollectRequest):
    """Collect a single training sample for later training.

    Samples are accumulated until /train is called. This endpoint
    allows incremental data collection without immediate training.

    Args:
        request: CollectRequest with model_id, features, and runtime_ms.

    Returns:
        CollectResponse with the total number of samples collected.

    Raises:
        HTTPException: If validation fails.
    """
    try:
        dependencies.predictor_core.collect(
            model_id=request.model_id,
            platform_info=request.platform_info,
            prediction_type=request.prediction_type,
            features=request.features,
            runtime_ms=request.runtime_ms,
        )

        count = dependencies.predictor_core.get_collected_count(
            request.model_id, request.platform_info, request.prediction_type
        )

        return CollectResponse(
            status="success",
            samples_collected=count,
            message=f"Sample collected. Total: {count} samples.",
        )

    except ValidationError as e:
        raise handle_library_exception(e, f"collect model_id={request.model_id}")
    except Exception as e:
        raise handle_library_exception(e, f"collect model_id={request.model_id}")


# =============================================================================
# /train - Train Model
# =============================================================================


@router.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest):
    """Train or update a model.

    Combines collected samples (from /collect) with features_list from request
    and trains a model. If preprocess_config is provided, applies per-feature
    preprocessing chains.

    Args:
        request: TrainingRequest with model_id, features, and config.

    Returns:
        TrainingResponse with training status and model key.

    Raises:
        HTTPException: If validation fails or training errors occur.
    """
    from src.api.core import TrainingError, ValidationError

    try:
        # Combine collected samples with request data
        all_features = list(request.features_list) if request.features_list else []

        # Get collected samples from accumulator
        collected_count = dependencies.predictor_core.get_collected_count(
            request.model_id, request.platform_info, request.prediction_type
        )

        if collected_count > 0:
            # Get accumulated samples and convert to features_list format
            key = dependencies.predictor_core._make_accumulator_key(
                request.model_id, request.platform_info, request.prediction_type
            )
            collected_samples = dependencies.predictor_core._accumulated.get(key, [])
            for sample in collected_samples:
                all_features.append({
                    **sample.features,
                    "runtime_ms": sample.runtime_ms,
                })

            # Clear collected after adding to training
            dependencies.predictor_core.clear_collected(
                request.model_id, request.platform_info, request.prediction_type
            )

        # Determine preprocessor config to use
        # Support both new preprocess_config and deprecated fields
        preprocess_config = request.preprocess_config
        if preprocess_config is None and request.enable_preprocessors:
            # Convert deprecated format to new format (backwards compatibility)
            # Old format: enable_preprocessors=["semantic"], preprocessor_mappings={"semantic": ["sentence"]}
            # Maps to: preprocess_config={"sentence": ["semantic"]}
            if request.preprocessor_mappings:
                preprocess_config = {}
                for prep_name, feature_keys in request.preprocessor_mappings.items():
                    for feature_key in feature_keys:
                        if feature_key not in preprocess_config:
                            preprocess_config[feature_key] = []
                        preprocess_config[feature_key].append(prep_name)

        # Train using library API with pipeline preprocessing
        predictor = dependencies.predictor_api.train_predictor_with_pipeline(
            features_list=all_features,
            prediction_type=request.prediction_type,
            config=request.training_config,
            preprocess_config=preprocess_config,
        )

        # Save using library API
        dependencies.predictor_api.save_model(
            model_id=request.model_id,
            platform_info=request.platform_info,
            prediction_type=request.prediction_type,
            predictor=predictor,
            samples_count=len(all_features),
        )

        model_key = dependencies.predictor_api.generate_model_key(
            request.model_id, request.platform_info, request.prediction_type
        )

        return TrainingResponse(
            status="success",
            message=(
                f"Model trained with {len(all_features)} samples "
                f"({collected_count} collected + {len(request.features_list)} from request)"
            ),
            model_key=model_key,
            samples_trained=len(all_features),
        )

    except ValidationError as e:
        raise handle_library_exception(e, f"train model_id={request.model_id}")
    except TrainingError as e:
        raise handle_library_exception(e, f"train model_id={request.model_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise handle_library_exception(e, f"train model_id={request.model_id}")
