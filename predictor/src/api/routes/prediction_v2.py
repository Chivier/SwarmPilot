"""V2 Prediction endpoint with automatic chain application.

Provides the /v2/predict endpoint that automatically applies the preprocessing
chain stored with the model during training.

Key V2 Design:
- No preprocess_chain in prediction request
- Chain is loaded from model metadata and applied automatically
- Response includes chain_applied flag
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api import dependencies
from src.api.core import ModelNotFoundError
from src.api.core import PredictionError
from src.api.core import ValidationError
from src.api.routes.helpers import handle_library_exception
from src.models import PredictionRequestV2
from src.models import PredictionResponseV2
from src.preprocessor.chain_v2 import PreprocessorChainV2
from src.preprocessor.registry_v2 import PreprocessorsRegistryV2
from src.utils.logging import get_logger

logger = get_logger()

router = APIRouter(prefix="/v2", tags=["V2 Prediction"])

# Registry instance for creating chains from stored config
_registry = PreprocessorsRegistryV2()


def _load_chain_from_metadata(metadata: dict) -> PreprocessorChainV2 | None:
    """Load preprocessing chain from model metadata.

    Args:
        metadata: Model metadata dict.

    Returns:
        PreprocessorChainV2 or None if no chain stored.

    Raises:
        ValueError: If chain config is invalid.
    """
    chain_config = metadata.get("preprocess_chain_v2")
    if not chain_config:
        return None

    return _registry.create_chain_from_config(chain_config, chain_name="stored_chain")


@router.post("/predict", response_model=PredictionResponseV2)
async def predict_v2(request: PredictionRequestV2):
    """Make a prediction with automatic chain application.

    The preprocessing chain stored with the model during training is
    automatically applied to the input features. No chain can be passed
    in the request - this ensures training/inference consistency.

    Args:
        request: PredictionRequestV2 with model_id, features, etc.
            NOTE: preprocess_chain field is intentionally NOT accepted.

    Returns:
        PredictionResponseV2 with prediction result and chain_applied flag.

    Raises:
        HTTPException: If model not found or prediction fails.
    """
    try:
        # Generate model key for lookup
        model_key = dependencies.predictor_api.generate_model_key(
            request.model_id, request.platform_info, request.prediction_type
        )

        # Load model data to get metadata (use versioned loading)
        model_data, version = dependencies.predictor_api._storage.load_model_versioned(
            model_id=request.model_id,
            platform_info=request.platform_info.model_dump(),
            prediction_type=request.prediction_type,
        )
        if model_data is None:
            raise ModelNotFoundError(f"Model not found: {model_key}")

        metadata = model_data.get("metadata", {})

        # Load chain from metadata
        chain = _load_chain_from_metadata(metadata)
        chain_applied = chain is not None

        # Append hardware info to features
        features = request.features.copy()
        hardware_features = request.platform_info.extract_gpu_specs()
        if hardware_features:
            features.update(hardware_features)

        # Apply chain if present
        if chain is not None:
            features = chain.transform(features)

        # Load predictor (possibly from cache)
        predictor = dependencies.predictor_api.load_model(
            model_id=request.model_id,
            platform_info=request.platform_info,
            prediction_type=request.prediction_type,
        )

        # Make prediction
        result = dependencies.predictor_api.predict_with_predictor(
            predictor=predictor,
            features=features,
        )

        logger.debug(
            f"V2 prediction: model={model_key}, chain_applied={chain_applied}"
        )

        return PredictionResponseV2(
            model_id=request.model_id,
            platform_info=request.platform_info,
            prediction_type=request.prediction_type,
            result=result,
            chain_applied=chain_applied,
        )

    except ModelNotFoundError as e:
        raise handle_library_exception(e, f"v2/predict model_id={request.model_id}")
    except ValidationError as e:
        raise handle_library_exception(e, f"v2/predict model_id={request.model_id}")
    except PredictionError as e:
        raise handle_library_exception(e, f"v2/predict model_id={request.model_id}")
    except ValueError as e:
        # Chain loading error (corrupted config)
        raise handle_library_exception(
            ValidationError(f"Invalid stored chain config: {e}"),
            f"v2/predict model_id={request.model_id}",
        )
    except Exception as e:
        raise handle_library_exception(e, f"v2/predict model_id={request.model_id}")
