"""Predictor management endpoints for the scheduler.

Exposes training, prediction, and model listing through the
embedded predictor / training library clients.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from swarmpilot.scheduler.models.requests import (
    PredictorPredictRequest,
    PredictorTrainRequest,
)
from swarmpilot.scheduler.models.responses import (
    PredictorModelsResponse,
    PredictorPredictResponse,
    PredictorStatusResponse,
    PredictorTrainResponse,
)

router = APIRouter(prefix="/predictor", tags=["predictor"])

# Minimum number of samples required before auto-switching the
# scheduling strategy to ``probabilistic`` after training.
MIN_TRAINING_SAMPLES = 10


def _get_clients() -> tuple[Any, Any | None]:
    """Return the global predictor and training clients.

    Deferred import avoids circular dependency with ``api.py``
    which creates both clients at module-load time.

    Returns:
        Tuple of (predictor_client, training_client).
    """
    from swarmpilot.scheduler.api import (
        predictor_client,
        training_client,
    )

    return predictor_client, training_client


def _maybe_switch_to_probabilistic(
    samples_trained: int,
) -> str | None:
    """Switch strategy to ``probabilistic`` if enough samples.

    When training produces at least ``MIN_TRAINING_SAMPLES``
    samples the global scheduling strategy is replaced with
    a fresh ``ProbabilisticSchedulingStrategy`` instance,
    mirroring the mechanism used by ``POST /v1/strategy/set``.

    Args:
        samples_trained: Number of samples consumed during
            the training run.

    Returns:
        The name of the active strategy after the check, or
        ``None`` if no switch was attempted (too few samples).
    """
    if samples_trained < MIN_TRAINING_SAMPLES:
        return None

    import swarmpilot.scheduler.api as api_module
    from swarmpilot.scheduler.algorithms import get_strategy

    current_name = api_module.scheduling_strategy.__class__.__name__
    if current_name == "ProbabilisticSchedulingStrategy":
        logger.info(
            "[auto-switch] Already using probabilistic "
            "strategy; skipping switch."
        )
        return "probabilistic"

    try:
        new_strategy = get_strategy(
            strategy_name="probabilistic",
            predictor_client=api_module.predictor_client,
            instance_registry=api_module.instance_registry,
            target_quantile=(
                api_module.config.scheduling.probabilistic_quantile
            ),
        )
        # Preserve the worker queue manager reference.
        wqm = getattr(
            api_module.scheduling_strategy,
            "_worker_queue_manager",
            None,
        )
        if wqm is not None:
            new_strategy.set_worker_queue_manager(wqm)

        api_module.scheduling_strategy = new_strategy
        logger.success(
            "[auto-switch] Switched scheduling strategy to "
            "'probabilistic' after successful training "
            f"({samples_trained} samples)."
        )
        return "probabilistic"
    except (ValueError, ImportError, RuntimeError):
        logger.opt(exception=True).error(
            "[auto-switch] Failed to switch strategy to "
            "'probabilistic'; keeping current strategy."
        )
        return None


# ------------------------------------------------------------------
# POST /v1/predictor/train
# ------------------------------------------------------------------


@router.post("/train", response_model=PredictorTrainResponse)
async def train_model(
    request: PredictorTrainRequest,
) -> PredictorTrainResponse:
    """Trigger training for a model.

    Uses the embedded training client to flush buffered samples
    and train the specified model.  If no training client is
    configured (auto-training disabled) the endpoint returns an
    error.

    Args:
        request: Training request with model_id and options.

    Returns:
        PredictorTrainResponse with training outcome.

    Raises:
        HTTPException: If training client is not available or
            training fails.
    """
    _, training_client = _get_clients()

    if training_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Training client not configured. Set TRAINING_ENABLE_AUTO=true."
            ),
        )

    try:
        success = await training_client.flush(force=True)
        samples = training_client.get_buffer_size()

        # Auto-switch to probabilistic strategy when training
        # succeeds with enough samples.
        strategy_name: str | None = None
        if success:
            strategy_name = _maybe_switch_to_probabilistic(
                samples,
            )

        return PredictorTrainResponse(
            success=success,
            model_id=request.model_id,
            samples_trained=samples,
            message=(
                "Training completed successfully"
                if success
                else "No samples available for training"
            ),
            strategy=strategy_name,
        )
    except Exception as exc:
        logger.opt(exception=True).error(
            f"Training failed for model_id={request.model_id}: {exc}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {exc}",
        ) from exc


# ------------------------------------------------------------------
# POST /v1/predictor/retrain
# ------------------------------------------------------------------


@router.post("/retrain", response_model=PredictorTrainResponse)
async def retrain_model(
    request: PredictorTrainRequest,
) -> PredictorTrainResponse:
    """Force retrain a model.

    Identical to ``/train`` but always forces the flush regardless
    of minimum-sample thresholds.

    Args:
        request: Training request with model_id and options.

    Returns:
        PredictorTrainResponse with training outcome.

    Raises:
        HTTPException: If training client is not available or
            training fails.
    """
    # Reuse the train endpoint; force=True is always used there.
    return await train_model(request)


# ------------------------------------------------------------------
# GET /v1/predictor/status/{model_id}
# ------------------------------------------------------------------


@router.get(
    "/status/{model_id}",
    response_model=PredictorStatusResponse,
)
async def get_model_status(
    model_id: str,
) -> PredictorStatusResponse:
    """Get predictor status for a model.

    Returns information about samples collected and any trained
    model entries stored on disk for the given ``model_id``.

    Args:
        model_id: Identifier of the model.

    Returns:
        PredictorStatusResponse with model status.

    Raises:
        HTTPException: If the model has never been seen.
    """
    predictor_client, training_client = _get_clients()

    storage = predictor_client._low_level._storage

    # Collect trained-model entries for this model_id
    all_models = storage.list_models()
    matched = [m for m in all_models if m.get("model_id") == model_id]

    # Count buffered samples for this model_id
    buffer_count = 0
    if training_client is not None:
        buffer_count = sum(
            1 for s in training_client._samples_buffer if s.model_id == model_id
        )

    if not matched and buffer_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )

    return PredictorStatusResponse(
        success=True,
        model_id=model_id,
        samples_collected=buffer_count,
        models=matched,
    )


# ------------------------------------------------------------------
# POST /v1/predictor/predict
# ------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictorPredictResponse,
)
async def predict(
    request: PredictorPredictRequest,
) -> PredictorPredictResponse:
    """Make a manual prediction (not via scheduling).

    Delegates to the embedded predictor library client to run
    a single-platform prediction.

    Args:
        request: Prediction request with model_id, platform_info,
            features, and prediction_type.

    Returns:
        PredictorPredictResponse with prediction results.

    Raises:
        HTTPException: If model not found or prediction fails.
    """
    predictor_client, _ = _get_clients()

    try:
        result = predictor_client._predict_single_platform(
            model_id=request.model_id,
            metadata=request.features,
            platform_info_dict=request.platform_info,
            prediction_type=request.prediction_type,
            quantiles_list=None,
            instance_ids=["manual"],
        )

        pred = result[0]

        quantiles_out: dict[str, float] | None = None
        if pred.quantiles:
            quantiles_out = {str(k): v for k, v in pred.quantiles.items()}

        return PredictorPredictResponse(
            success=True,
            model_id=request.model_id,
            expected_runtime_ms=pred.predicted_time_ms,
            error_margin_ms=pred.error_margin_ms,
            quantiles=quantiles_out,
        )

    except ValueError as exc:
        error_msg = str(exc)
        if "Model not found" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        ) from exc
    except Exception as exc:
        logger.opt(exception=True).error(
            f"Prediction failed for model_id={request.model_id}: {exc}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


# ------------------------------------------------------------------
# GET /v1/predictor/models
# ------------------------------------------------------------------


@router.get(
    "/models",
    response_model=PredictorModelsResponse,
)
async def list_models() -> PredictorModelsResponse:
    """List all models with predictor data.

    Returns:
        PredictorModelsResponse with a list of trained model
        metadata entries.
    """
    predictor_client, _ = _get_clients()

    storage = predictor_client._low_level._storage
    all_models = storage.list_models()

    return PredictorModelsResponse(
        success=True,
        models=all_models,
    )
