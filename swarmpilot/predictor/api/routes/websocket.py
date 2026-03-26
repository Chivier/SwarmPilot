"""WebSocket prediction endpoint."""

from __future__ import annotations

import json
import traceback

from fastapi import WebSocket, WebSocketDisconnect

from swarmpilot.predictor.api import dependencies
from swarmpilot.predictor.models import PredictionRequest, PredictionResponse
from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor
from swarmpilot.predictor.utils.experiment import (
    generate_experiment_prediction,
    is_experiment_mode,
)
from swarmpilot.predictor.utils.logging import get_logger

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
                    "message": f"Failed to parse JSON: {e!s}",
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
                    "message": f"Failed to validate request: {e!s}",
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
                    # Generate synthetic prediction
                    logger.debug("request is experiment request")
                    try:
                        # Pass custom quantiles to experiment mode if provided
                        config = {}
                        if request.quantiles is not None:
                            config["quantiles"] = request.quantiles

                        result = generate_experiment_prediction(
                            prediction_type=request.prediction_type,
                            features=request.features,
                            config=config,
                        )

                        response = PredictionResponse(
                            model_id=request.model_id,
                            platform_info=request.platform_info,
                            prediction_type=request.prediction_type,
                            result=result,
                        )

                        await websocket.send_json(response.model_dump())
                        continue

                    except ValueError as e:
                        error_detail = {
                            "error": "Experiment mode error",
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        }
                        dependencies._log_error(
                            error_context=(
                                f"WebSocket - experiment mode error "
                                f"for model_id={request.model_id}"
                            ),
                            error_detail=error_detail,
                            exception=e,
                        )
                        await websocket.send_json(error_detail)
                        continue

                # Normal mode: load model and predict
                model_key = dependencies.storage.generate_model_key(
                    model_id=request.model_id,
                    platform_info=request.platform_info.model_dump(),
                    prediction_type=request.prediction_type,
                )

                # Try to get predictor from cache
                cached_result = dependencies.model_cache.get(model_key)

                if cached_result is not None:
                    # Cache hit - use cached predictor
                    predictor, stored_prediction_type = cached_result

                    # Validate prediction type matches
                    if stored_prediction_type != request.prediction_type:
                        error_detail = {
                            "error": "Prediction type mismatch",
                            "message": (
                                f"Model was trained with prediction_type="
                                f"'{stored_prediction_type}', but request has "
                                f"'{request.prediction_type}'"
                            ),
                            "model_prediction_type": stored_prediction_type,
                            "request_prediction_type": request.prediction_type,
                        }
                        dependencies._log_error(
                            error_context=(
                                f"WebSocket - prediction type mismatch (cached) "
                                f"for model_id={request.model_id}"
                            ),
                            error_detail=error_detail,
                            include_traceback=False,
                        )
                        await websocket.send_json(error_detail)
                        continue
                else:
                    # Cache miss - load model from storage
                    model_data = dependencies.storage.load_model(model_key)
                    if model_data is None:
                        error_detail = {
                            "error": "Model not found",
                            "message": (
                                f"No trained model found for "
                                f"model_id='{request.model_id}' "
                                f"with given platform_info"
                            ),
                            "model_id": request.model_id,
                            "platform_info": request.platform_info.model_dump(),
                            "model_key": model_key,
                        }
                        dependencies._log_error(
                            error_context=(
                                f"WebSocket - model not found "
                                f"for model_id={request.model_id}"
                            ),
                            error_detail=error_detail,
                            include_traceback=False,
                        )
                        await websocket.send_json(error_detail)
                        continue

                    # Validate prediction type matches
                    stored_prediction_type = model_data["metadata"].get(
                        "prediction_type"
                    )
                    if stored_prediction_type != request.prediction_type:
                        error_detail = {
                            "error": "Prediction type mismatch",
                            "message": (
                                f"Model was trained with prediction_type="
                                f"'{stored_prediction_type}', but request has "
                                f"'{request.prediction_type}'"
                            ),
                            "model_prediction_type": stored_prediction_type,
                            "request_prediction_type": request.prediction_type,
                        }
                        dependencies._log_error(
                            error_context=(
                                f"WebSocket - prediction type mismatch (storage) "
                                f"for model_id={request.model_id}"
                            ),
                            error_detail=error_detail,
                            include_traceback=False,
                        )
                        await websocket.send_json(error_detail)
                        continue

                    # Create predictor and load state
                    if request.prediction_type == "expect_error":
                        predictor = ExpectErrorPredictor()
                    elif request.prediction_type == "quantile":
                        predictor = QuantilePredictor()
                    else:
                        error_detail = {
                            "error": "Invalid prediction type",
                            "message": (
                                f"prediction_type must be 'expect_error' "
                                f"or 'quantile', "
                                f"got '{request.prediction_type}'"
                            ),
                        }
                        dependencies._log_error(
                            error_context=(
                                f"WebSocket - invalid prediction_type "
                                f"for model_id={request.model_id}"
                            ),
                            error_detail=error_detail,
                            include_traceback=False,
                        )
                        await websocket.send_json(error_detail)
                        continue

                    predictor.load_model_state(model_data["predictor_state"])

                    # Cache the loaded predictor
                    dependencies.model_cache.put(
                        model_key, predictor, stored_prediction_type
                    )
                    logger.info(f"Loaded and cached model (WebSocket): {model_key}")

                # Make prediction
                try:
                    result = predictor.predict(request.features)

                    response = PredictionResponse(
                        model_id=request.model_id,
                        platform_info=request.platform_info,
                        prediction_type=request.prediction_type,
                        result=result,
                    )

                    await websocket.send_json(response.model_dump())

                except ValueError as e:
                    # Feature validation errors
                    error_detail = {
                        "error": "Invalid features",
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    dependencies._log_error(
                        error_context=(
                            f"WebSocket - feature validation error "
                            f"for model_id={request.model_id}"
                        ),
                        error_detail=error_detail,
                        exception=e,
                    )
                    await websocket.send_json(error_detail)
                except Exception as e:
                    error_detail = {
                        "error": "Prediction failed",
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    dependencies._log_error(
                        error_context=(
                            f"WebSocket - prediction failed "
                            f"for model_id={request.model_id}"
                        ),
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
