"""Training endpoint for model training."""

from __future__ import annotations

import traceback

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from src.api import dependencies
from src.models import TrainingRequest
from src.models import TrainingResponse
from src.predictor.decision_tree import DecisionTreePredictor
from src.predictor.expect_error import ExpectErrorPredictor
from src.predictor.linear_regression import LinearRegressionPredictor
from src.predictor.quantile import QuantilePredictor
from src.utils.logging import get_logger

logger = get_logger()

router = APIRouter()


@router.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest):
    """Train or update a model.

    Trains an MLP model on the provided features and runtime data.
    Supports both expect_error and quantile prediction types.

    Args:
        request: TrainingRequest with model_id, features, and config.

    Returns:
        TrainingResponse with training status and model key.

    Raises:
        HTTPException: If validation fails or training errors occur.
    """
    try:
        # Validate minimum samples
        if len(request.features_list) < 10:
            error_detail = {
                "error": "Insufficient training data",
                "message": (
                    f"Need at least 10 samples, got {len(request.features_list)}"
                ),
                "samples_provided": len(request.features_list),
                "minimum_required": 10,
            }
            dependencies._log_error(
                error_context=(
                    f"Training validation failed for model_id={request.model_id}"
                ),
                error_detail=error_detail,
                include_traceback=False,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail,
            )

        # Create appropriate predictor
        if request.prediction_type == "expect_error":
            predictor = ExpectErrorPredictor()
        elif request.prediction_type == "quantile":
            predictor = QuantilePredictor()
        elif request.prediction_type == "linear_regression":
            predictor = LinearRegressionPredictor()
        elif request.prediction_type == "decision_tree":
            predictor = DecisionTreePredictor()
        else:
            error_detail = {
                "error": "Invalid prediction type",
                "message": (
                    f"prediction_type must be 'expect_error', 'quantile', "
                    f"'linear_regression', or 'decision_tree', "
                    f"got '{request.prediction_type}'"
                ),
            }
            dependencies._log_error(
                error_context=(
                    f"Training failed - invalid prediction_type "
                    f"for model_id={request.model_id}"
                ),
                error_detail=error_detail,
                include_traceback=False,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail,
            )

        try:
            processed_features_list = []
            if request.enable_preprocessors:
                # Process each sample
                for features in request.features_list:
                    # Make a copy to avoid modifying the original
                    processed_features_dict = dict(features)

                    # Apply each preprocessor
                    for preprocessor_name in request.enable_preprocessors:
                        preprocessor = (
                            dependencies.preprocessors_registry.get_preprocessor(
                                preprocessor_name
                            )
                        )
                        target_feature_keys = request.preprocessor_mappings[
                            preprocessor_name
                        ]

                        # Validate all required features exist
                        assert all(
                            key in processed_features_dict
                            for key in target_feature_keys
                        ), f"Feature keys {target_feature_keys} not all found in features"

                        # Extract target feature values
                        target_feature_values = [
                            processed_features_dict[key]
                            for key in target_feature_keys
                        ]

                        # Apply preprocessor
                        processed_features, remove_origin = preprocessor(
                            target_feature_values
                        )

                        # Add processed features to the dict
                        for k, v in processed_features.items():
                            processed_features_dict[k] = v

                        # Remove original features if requested
                        if remove_origin:
                            for key in target_feature_keys:
                                del processed_features_dict[key]

                    processed_features_list.append(processed_features_dict)
            else:
                processed_features_list = request.features_list

        except Exception as e:
            error_detail = {
                "error": "Preprocessor error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=(
                    f"Preprocessing failed for model_id={request.model_id}"
                ),
                error_detail=error_detail,
                exception=e,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail,
            )

        # Train the model
        try:
            training_metadata = predictor.train(
                features_list=processed_features_list,
                config=request.training_config or {},
            )
        except ValueError as e:
            error_detail = {
                "error": "Training validation error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=(
                    f"Training validation error for model_id={request.model_id}"
                ),
                error_detail=error_detail,
                exception=e,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail,
            )
        except Exception as e:
            error_detail = {
                "error": "Training failed",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=f"Training failed for model_id={request.model_id}",
                error_detail=error_detail,
                exception=e,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_detail,
            )

        # Generate model key with prediction_type
        model_key = dependencies.storage.generate_model_key(
            model_id=request.model_id,
            platform_info=request.platform_info.model_dump(),
            prediction_type=request.prediction_type,
        )

        # Save model
        try:
            predictor_state = predictor.get_model_state()
            metadata = {
                "model_id": request.model_id,
                "platform_info": request.platform_info.model_dump(),
                "prediction_type": request.prediction_type,
                "samples_count": len(request.features_list),
                "training_config": request.training_config,
            }
            dependencies.storage.save_model(model_key, predictor_state, metadata)

            # Invalidate cache for this model (it has been retrained)
            dependencies.model_cache.invalidate(model_key)
            logger.info(f"Invalidated cache for retrained model: {model_key}")

        except Exception as e:
            error_detail = {
                "error": "Model save failed",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=(
                    f"Model save failed for model_id={request.model_id}, "
                    f"model_key={model_key}"
                ),
                error_detail=error_detail,
                exception=e,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_detail,
            )

        return TrainingResponse(
            status="success",
            message=(
                f"Model trained successfully with "
                f"{len(request.features_list)} samples"
            ),
            model_key=model_key,
            samples_trained=len(request.features_list),
        )

    except HTTPException:
        raise
    except Exception as e:
        error_detail = {
            "error": "Unexpected error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context=(
                f"Unexpected error in /train endpoint "
                f"for model_id={request.model_id}"
            ),
            error_detail=error_detail,
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        )
