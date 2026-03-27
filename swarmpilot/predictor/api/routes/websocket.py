"""WebSocket prediction endpoint."""

from __future__ import annotations

import json
import traceback

import pydantic
from fastapi import WebSocket, WebSocketDisconnect

from swarmpilot.predictor.api import dependencies
from swarmpilot.predictor.api.services.prediction_service import (
    PredictionServiceError,
    execute_prediction,
)
from swarmpilot.predictor.models import PredictionRequest
from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()


async def _receive_request(websocket: WebSocket) -> PredictionRequest | None:
    """Receive and validate one PredictionRequest from the WebSocket."""
    try:
        data = await websocket.receive_text()
        request_data = json.loads(data)
    except json.JSONDecodeError as exc:
        error_detail = {
            "error": "Invalid JSON",
            "message": f"Failed to parse JSON: {exc!s}",
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="WebSocket - JSON parsing error",
            error_detail=error_detail,
            exception=exc,
        )
        await websocket.send_json(error_detail)
        return None
    except (RuntimeError, ConnectionError) as exc:
        error_detail = {
            "error": "Receive error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="WebSocket - receive error",
            error_detail=error_detail,
            exception=exc,
        )
        await websocket.send_json(error_detail)
        return None

    try:
        return PredictionRequest(**request_data)
    except (pydantic.ValidationError, TypeError) as exc:
        error_detail = {
            "error": "Invalid request",
            "message": f"Failed to validate request: {exc!s}",
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="WebSocket - request validation error",
            error_detail=error_detail,
            exception=exc,
        )
        await websocket.send_json(error_detail)
        return None


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
            request = await _receive_request(websocket)
            if request is None:
                continue

            try:
                response = execute_prediction(request)
                await websocket.send_json(response.model_dump())
            except PredictionServiceError as exc:
                dependencies._log_error(
                    error_context=(
                        f"WebSocket - prediction service error for model_id={request.model_id}"
                    ),
                    error_detail=exc.error_detail,
                    exception=exc,
                )
                await websocket.send_json(exc.error_detail)
            except ValueError as exc:
                error_detail = {
                    "error": "Invalid features",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                dependencies._log_error(
                    error_context=(
                        f"WebSocket - feature validation error for model_id={request.model_id}"
                    ),
                    error_detail=error_detail,
                    exception=exc,
                )
                await websocket.send_json(error_detail)
            except Exception as exc:
                error_detail = {
                    "error": "Prediction failed",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                dependencies._log_error(
                    error_context=(
                        f"WebSocket - prediction failed for model_id={request.model_id}"
                    ),
                    error_detail=error_detail,
                    exception=exc,
                )
                await websocket.send_json(error_detail)
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as exc:
        error_detail = {
            "error": "WebSocket error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="WebSocket - connection-level error",
            error_detail=error_detail,
            exception=exc,
        )
        try:
            await websocket.send_json(error_detail)
        except RuntimeError:
            pass
        finally:
            await websocket.close()
