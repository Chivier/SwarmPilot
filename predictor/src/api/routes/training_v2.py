"""V2 Training endpoints with preprocessing chain support.

Provides endpoints for:
- /v2/collect: Collect training samples with optional preprocessing
- /v2/train: Train model with preprocessing chain (stored with model)

Key V2 Design:
- Chain is set at training time and stored with model metadata
- Chain is validated before training begins
- Prediction automatically uses stored chain (no chain in request)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from src.api import dependencies
from src.api.core import TrainingError
from src.api.core import ValidationError
from src.api.routes.helpers import handle_library_exception
from src.models import ChainConfigV2
from src.models import CollectRequestV2
from src.models import CollectResponseV2
from src.models import TrainingRequestV2
from src.models import TrainingResponseV2
from src.preprocessor.chain_v2 import PreprocessorChainV2
from src.preprocessor.registry_v2 import PreprocessorsRegistryV2
from src.utils.logging import get_logger

logger = get_logger()

router = APIRouter(prefix="/v2", tags=["V2 Training"])

# Registry instance for creating chains from config
_registry = PreprocessorsRegistryV2()


def _build_chain_from_config(
    config: ChainConfigV2 | None,
    chain_name: str = "api_chain",
) -> PreprocessorChainV2 | None:
    """Build a PreprocessorChainV2 from ChainConfigV2.

    Args:
        config: Chain configuration from request.
        chain_name: Name for the created chain.

    Returns:
        PreprocessorChainV2 or None if no config/empty steps.

    Raises:
        ValueError: If preprocessor name is unknown.
    """
    if config is None or not config.steps:
        return None

    # Convert to registry format
    registry_config = [
        {"name": step.name, "params": step.params}
        for step in config.steps
    ]

    return _registry.create_chain_from_config(registry_config, chain_name)


def _hash_chain_config(config: ChainConfigV2 | None) -> str | None:
    """Compute hash of chain config for change detection.

    Args:
        config: Chain configuration.

    Returns:
        Hash string or None if no config.
    """
    if config is None or not config.steps:
        return None

    # Serialize to JSON for consistent hashing
    config_dict = [
        {"name": step.name, "params": step.params}
        for step in config.steps
    ]
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


def _validate_chain_on_sample(
    chain: PreprocessorChainV2,
    sample: dict[str, Any],
) -> None:
    """Validate chain by running on a sample.

    Args:
        chain: Chain to validate.
        sample: Sample to run chain on.

    Raises:
        ValidationError: If chain validation fails.
    """
    try:
        # Remove runtime_ms before validation
        features = {k: v for k, v in sample.items() if k != "runtime_ms"}
        chain.transform(features)
    except Exception as e:
        raise ValidationError(f"Chain validation failed: {e}") from e


# =============================================================================
# /v2/collect - Collect Training Samples
# =============================================================================


@router.post("/collect", response_model=CollectResponseV2)
async def collect_sample_v2(request: CollectRequestV2):
    """Collect a training sample with optional preprocessing.

    Samples are accumulated until /v2/train is called. If preprocess_chain
    is provided, features are preprocessed before storing.

    Args:
        request: CollectRequestV2 with features and optional chain.

    Returns:
        CollectResponseV2 with sample count.

    Raises:
        HTTPException: If validation fails.
    """
    try:
        # Build chain if provided
        chain = _build_chain_from_config(
            request.preprocess_chain, chain_name="collect_chain"
        )

        # Preprocess features if chain provided
        features = request.features.copy()
        if chain is not None:
            features = chain.transform(features)

        # Collect sample
        dependencies.predictor_core.collect(
            model_id=request.model_id,
            platform_info=request.platform_info,
            prediction_type=request.prediction_type,
            features=features,
            runtime_ms=request.runtime_ms,
        )

        count = dependencies.predictor_core.get_collected_count(
            request.model_id, request.platform_info, request.prediction_type
        )

        return CollectResponseV2(
            status="success",
            samples_collected=count,
            message=f"Sample collected. Total: {count} samples.",
        )

    except ValidationError as e:
        raise handle_library_exception(e, f"v2/collect model_id={request.model_id}")
    except ValueError as e:
        # Chain building error (unknown preprocessor)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Chain configuration error",
                "message": str(e),
            },
        )
    except Exception as e:
        raise handle_library_exception(e, f"v2/collect model_id={request.model_id}")


# =============================================================================
# /v2/train - Train Model with Chain
# =============================================================================


@router.post("/train", response_model=TrainingResponseV2)
async def train_model_v2(request: TrainingRequestV2):
    """Train a model with optional preprocessing chain.

    Chain Resolution:
    - Chain provided → use provided chain
    - No chain + existing model → use model's stored chain
    - No chain + new model → no preprocessing

    The chain is validated on the first sample before training begins.
    On success, the chain is stored with the model metadata.

    Args:
        request: TrainingRequestV2 with features and optional chain.

    Returns:
        TrainingResponseV2 with training status and chain_stored flag.

    Raises:
        HTTPException: If validation or training fails.
    """
    try:
        # Combine collected samples with request data
        all_features: list[dict[str, Any]] = []

        if request.features_list:
            all_features.extend(request.features_list)

        # Get collected samples from accumulator
        collected_count = dependencies.predictor_core.get_collected_count(
            request.model_id, request.platform_info, request.prediction_type
        )

        if collected_count > 0:
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

        if not all_features:
            raise ValidationError("No training data available (collected or request)")

        # Build chain from config
        chain = _build_chain_from_config(
            request.preprocess_chain, chain_name="train_chain"
        )

        # Validate chain before training
        if chain is not None and all_features:
            _validate_chain_on_sample(chain, all_features[0])

        # Train using low-level API with V2 pipeline
        predictor = dependencies.predictor_api.train_predictor_with_pipeline_v2(
            features_list=all_features,
            prediction_type=request.prediction_type,
            config=request.training_config,
            chain=chain,
        )

        # Get predictor state and prepare metadata
        predictor_state = predictor.get_model_state()

        # Prepare metadata with chain config
        chain_config_dict = None
        chain_hash = None
        if request.preprocess_chain is not None and request.preprocess_chain.steps:
            chain_config_dict = [
                {"name": step.name, "params": step.params}
                for step in request.preprocess_chain.steps
            ]
            chain_hash = _hash_chain_config(request.preprocess_chain)

        metadata = {
            "model_id": request.model_id,
            "platform_info": request.platform_info.model_dump(),
            "prediction_type": request.prediction_type,
            "samples_count": len(all_features),
            "feature_names": predictor_state.get("feature_names", []),
            "saved_at": datetime.now().isoformat(),
            # V2 fields
            "preprocess_chain_v2": chain_config_dict,
            "chain_hash": chain_hash,
        }

        # Save model with versioning
        model_key = dependencies.predictor_api.generate_model_key(
            request.model_id, request.platform_info, request.prediction_type
        )

        # Use versioned save to get version timestamp
        version = dependencies.predictor_api._storage.save_model_versioned(
            model_id=request.model_id,
            platform_info=request.platform_info.model_dump(),
            prediction_type=request.prediction_type,
            predictor_state=predictor_state,
            metadata=metadata,
        )
        dependencies.predictor_api._cache.invalidate(model_key)

        logger.info(
            f"V2 trained model: {model_key} version {version}, samples={len(all_features)}, "
            f"chain_stored={chain is not None}"
        )

        return TrainingResponseV2(
            status="success",
            message=(
                f"Model trained with {len(all_features)} samples "
                f"({collected_count} collected + {len(request.features_list or [])} from request), "
                f"version {version}"
            ),
            model_key=model_key,
            samples_trained=len(all_features),
            chain_stored=chain is not None,
        )

    except ValidationError as e:
        raise handle_library_exception(e, f"v2/train model_id={request.model_id}")
    except TrainingError as e:
        raise handle_library_exception(e, f"v2/train model_id={request.model_id}")
    except ValueError as e:
        # Chain building error (unknown preprocessor)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Chain configuration error",
                "message": str(e),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise handle_library_exception(e, f"v2/train model_id={request.model_id}")
