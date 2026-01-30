"""Mock Predictor Server for E2E Testing.

This module provides a FastAPI mock predictor service that returns
the task's sleep_time (in seconds) converted to milliseconds as the
predicted runtime. This allows testing the scheduler's prediction-based
scheduling without a real ML model.

Prediction Logic:
- Extracts 'sleep_time' from request.features
- Returns sleep_time * 1000 as the median (p50) prediction
- Adds variance for other quantiles (p90=+10%, p95=+15%, p99=+19%)

Endpoints:
- POST /predict - Returns prediction based on task metadata
- GET /health - Health check endpoint

Environment Variables:
- PREDICTOR_PORT: Port to run on (default: 8002)
- PREDICTOR_LOG_LEVEL: Log level (default: INFO)

Usage:
    # Start the server
    uv run python examples/llm_cluster/mock_predictor_server.py

    # Or via uvicorn
    uv run uvicorn examples.llm_cluster.mock_predictor_server:app \
        --host 0.0.0.0 --port 8002
"""

import os
from typing import Any

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field


# Configuration from environment
PREDICTOR_PORT = int(os.environ.get("PREDICTOR_PORT", "8002"))
LOG_LEVEL = os.environ.get("PREDICTOR_LOG_LEVEL", "INFO")


# FastAPI Application
app = FastAPI(
    title="Mock Predictor Server",
    description="Returns sleep_time as predicted runtime for E2E testing",
    version="1.0.0",
)


# Request/Response Models
class PredictRequest(BaseModel):
    """Prediction request from scheduler."""

    model_id: str = Field(..., description="Model identifier")
    platform_info: dict[str, str] = Field(
        ..., description="Platform information (hardware, software)"
    )
    prediction_type: str = Field(
        default="quantile", description="Prediction type: 'quantile' or 'expect_error'"
    )
    features: dict[str, Any] = Field(
        default_factory=dict, description="Task metadata including sleep_time"
    )
    quantiles: list[float] | None = Field(
        default=None, description="Custom quantile levels (optional)"
    )
    enable_preprocessors: list[str] | None = Field(
        default=None, description="Preprocessors to enable (ignored in mock)"
    )
    preprocessor_mappings: dict[str, list[str]] | None = Field(
        default=None, description="Preprocessor mappings (ignored in mock)"
    )


class QuantileResult(BaseModel):
    """Quantile prediction result."""

    quantiles: dict[str, float]


class ExpectErrorResult(BaseModel):
    """Expected error prediction result."""

    expected_runtime_ms: float
    error_margin_ms: float


class PredictResponse(BaseModel):
    """Prediction response."""

    result: QuantileResult | ExpectErrorResult


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    predictor_type: str = "mock"


# Statistics tracking
_request_count = 0
_total_predictions = 0


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate prediction based on task metadata.

    Extracts sleep_time from features and returns it as predicted runtime.
    For quantile predictions, adds variance for p90, p95, p99.

    Args:
        request: Prediction request with model_id, platform_info, features

    Returns:
        Prediction response with quantiles or expect_error result
    """
    global _request_count, _total_predictions
    _request_count += 1

    # Extract sleep_time from features (default to 1.0 if not provided)
    sleep_time = request.features.get("sleep_time", 1.0)
    if isinstance(sleep_time, str):
        try:
            sleep_time = float(sleep_time)
        except ValueError:
            sleep_time = 1.0

    # Convert seconds to milliseconds
    base_ms = float(sleep_time) * 1000.0

    logger.debug(
        f"Prediction request #{_request_count}: model={request.model_id}, "
        f"sleep_time={sleep_time}s, base_ms={base_ms}ms, type={request.prediction_type}"
    )

    if request.prediction_type == "quantile":
        # Default quantiles if not specified
        quantile_levels = request.quantiles or [0.5, 0.9, 0.95, 0.99]

        # Generate quantile predictions with variance
        quantiles = {}
        for q in quantile_levels:
            if q <= 0.5:
                quantiles[str(q)] = base_ms
            elif q <= 0.9:
                quantiles[str(q)] = base_ms * 1.1  # +10%
            elif q <= 0.95:
                quantiles[str(q)] = base_ms * 1.15  # +15%
            else:
                quantiles[str(q)] = base_ms * 1.19  # +19%

        _total_predictions += 1
        logger.debug(f"Quantile prediction: {quantiles}")

        return PredictResponse(result=QuantileResult(quantiles=quantiles))

    elif request.prediction_type == "expect_error":
        # Expected value with error margin
        expected_runtime_ms = base_ms
        error_margin_ms = base_ms * 0.2  # 20% error margin

        _total_predictions += 1
        logger.debug(
            f"Expect-error prediction: expected={expected_runtime_ms}ms, "
            f"error_margin={error_margin_ms}ms"
        )

        return PredictResponse(
            result=ExpectErrorResult(
                expected_runtime_ms=expected_runtime_ms,
                error_margin_ms=error_margin_ms,
            )
        )

    else:
        # Unknown prediction type - default to quantile
        logger.warning(
            f"Unknown prediction type: {request.prediction_type}, using quantile"
        )
        quantiles = {
            "0.5": base_ms,
            "0.9": base_ms * 1.1,
            "0.95": base_ms * 1.15,
            "0.99": base_ms * 1.19,
        }
        return PredictResponse(result=QuantileResult(quantiles=quantiles))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status indicating the mock predictor is running
    """
    return HealthResponse(status="healthy", predictor_type="mock")


@app.get("/stats")
async def stats() -> dict[str, Any]:
    """Get mock predictor statistics.

    Returns:
        Statistics about prediction requests
    """
    return {
        "request_count": _request_count,
        "total_predictions": _total_predictions,
    }


if __name__ == "__main__":
    logger.info(f"Starting Mock Predictor Server on port {PREDICTOR_PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PREDICTOR_PORT,
        log_level=LOG_LEVEL.lower(),
    )
