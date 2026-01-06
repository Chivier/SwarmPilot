"""WebSocket prediction endpoint.

Provides real-time prediction via WebSocket connections.
"""

from __future__ import annotations

import json
import traceback

from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from src.api import dependencies
from src.api.core import ModelNotFoundError, PredictionError, ValidationError
from src.models import PredictionRequest
from src.models import PredictionResponse
from src.utils.experiment import generate_experiment_prediction
from src.utils.experiment import is_experiment_mode
from src.utils.logging import get_logger

logger = get_logger()


async def websocket_predict(websocket: WebSocket):
    """WebSocket endpoint for real-time runtime predictions.

    Accepts PredictionRequest JSON messages and returns PredictionResponse JSON.
    Keeps connection open for multiple prediction requests.

    Args:
        websocket: The WebSocket connection instance.
    """
    await websocket.accept()

    try:
        while True:
            # Receive JSON message from client
            try:
                data = await websocket.receive_text()
                request_data = json.loads(data)
            except json.JSONDecodeError as e:
                error_detail = {
                    "error": "Invalid JSON",
                    "message": f"Failed to parse JSON: {str(e)}",
                    "traceback": traceback.format_exc(),
                }
                dependencies._log_error(
                    error_context="WebSocket - JSON parsing error",
                    error_detail=error_detail,
                    exception=e,
                )
                await websocket.send_json(error_detail)
                continue
            except Exception as e:
                error_detail = {
                    "error": "Receive error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                dependencies._log_error(
                    error_context="WebSocket - receive error",
                    error_detail=error_detail,
                    exception=e,
                )
                await websocket.send_json(error_detail)
                continue

            # Validate and parse request
            try:
                request = PredictionRequest(**request_data)
            except Exception as e:
                error_detail = {
                    "error": "Invalid request",
                    "message": f"Failed to validate request: {str(e)}",
                    "traceback": traceback.format_exc(),
                }
                dependencies._log_error(
                    error_context="WebSocket - request validation error",
                    error_detail=error_detail,
                    exception=e,
                )
                await websocket.send_json(error_detail)
                continue

            try:
                # Check if experiment mode
                if is_experiment_mode(
                    request.features, request.platform_info.model_dump()
                ):
                    response = _handle_experiment_mode(request)
                    await websocket.send_json(response.model_dump())
                    continue

                # Normal mode: use library API for prediction
                # Append hardware info to features
                features = request.features.copy()
                hardware_features = request.platform_info.extract_gpu_specs()
                if hardware_features:
                    features.update(hardware_features)

                # Determine preprocessor config to use
                preprocess_config = request.preprocess_config
                if preprocess_config is None and request.enable_preprocessors:
                    if request.preprocessor_mappings:
                        preprocess_config = {}
                        for prep_name, feature_keys in request.preprocessor_mappings.items():
                            for feature_key in feature_keys:
                                if feature_key not in preprocess_config:
                                    preprocess_config[feature_key] = []
                                preprocess_config[feature_key].append(prep_name)

                # Use inference_pipeline for combined load + preprocess + predict
                result = dependencies.predictor_core.inference_pipeline(
                    model_id=request.model_id,
                    platform_info=request.platform_info,
                    prediction_type=request.prediction_type,
                    features=features,
                    preprocess_config=preprocess_config,
                )

                response = PredictionResponse(
                    model_id=result.model_id,
                    platform_info=result.platform_info,
                    prediction_type=result.prediction_type,
                    result=result.result,
                )

                await websocket.send_json(response.model_dump())

            except ModelNotFoundError as e:
                error_detail = {
                    "error": "Model not found",
                    "message": str(e),
                    "model_id": request.model_id,
                    "platform_info": request.platform_info.model_dump(),
                }
                dependencies._log_error(
                    error_context=f"WebSocket - model not found for model_id={request.model_id}",
                    error_detail=error_detail,
                    include_traceback=False,
                )
                await websocket.send_json(error_detail)
            except ValidationError as e:
                error_detail = {
                    "error": "Validation error",
                    "message": str(e),
                }
                dependencies._log_error(
                    error_context=f"WebSocket - validation error for model_id={request.model_id}",
                    error_detail=error_detail,
                    include_traceback=False,
                )
                await websocket.send_json(error_detail)
            except PredictionError as e:
                error_detail = {
                    "error": "Prediction failed",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                dependencies._log_error(
                    error_context=f"WebSocket - prediction failed for model_id={request.model_id}",
                    error_detail=error_detail,
                    exception=e,
                )
                await websocket.send_json(error_detail)
            except Exception as e:
                error_detail = {
                    "error": "Unexpected error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                dependencies._log_error(
                    error_context="WebSocket - unexpected error in request processing",
                    error_detail=error_detail,
                    exception=e,
                )
                await websocket.send_json(error_detail)

    except WebSocketDisconnect:
        # Client disconnected, close gracefully
        logger.debug("WebSocket client disconnected")
        pass
    except Exception as e:
        # Unexpected error, try to send error message before closing
        error_detail = {
            "error": "WebSocket error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="WebSocket - connection-level error",
            error_detail=error_detail,
            exception=e,
        )
        try:
            await websocket.send_json(error_detail)
        except Exception:
            pass
        finally:
            await websocket.close()


def _handle_experiment_mode(request: PredictionRequest) -> PredictionResponse:
    """Handle experiment mode prediction (synthetic data).

    Args:
        request: PredictionRequest with experiment mode features.

    Returns:
        PredictionResponse with synthetic prediction.

    Raises:
        ValueError: If experiment mode prediction fails.
    """
    logger.debug("request is experiment request")

    # Pass custom quantiles to experiment mode if provided
    config = {}
    if request.quantiles is not None:
        config["quantiles"] = request.quantiles

    result = generate_experiment_prediction(
        prediction_type=request.prediction_type,
        features=request.features,
        config=config,
    )

    return PredictionResponse(
        model_id=request.model_id,
        platform_info=request.platform_info,
        prediction_type=request.prediction_type,
        result=result,
    )
