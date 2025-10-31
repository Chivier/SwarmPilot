"""
FastAPI application for runtime prediction service.

All API endpoints are implemented in this module.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any
import traceback

from .models import (
    TrainingRequest, TrainingResponse,
    PredictionRequest, PredictionResponse,
    ModelListResponse, ModelMetadata,
    HealthResponse, ErrorResponse
)
from .storage.model_storage import ModelStorage
from .predictor.expect_error import ExpectErrorPredictor
from .predictor.quantile import QuantilePredictor
from .utils.experiment import is_experiment_mode, generate_experiment_prediction


# Initialize storage
storage = ModelStorage(storage_dir="models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    storage_info = storage.get_storage_info()
    print(f"Storage initialized: {storage_info['storage_dir']}")
    print(f"Found {storage_info['model_count']} existing models")
    yield
    # Shutdown (cleanup if needed)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Runtime Predictor Service",
    description="MLP-based runtime prediction with expect/error and quantile regression",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service health status and checks if storage is accessible.
    """
    try:
        storage_info = storage.get_storage_info()

        if not storage_info['is_accessible']:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "reason": f"Storage directory not accessible: {storage_info['storage_dir']}"
                }
            )

        return HealthResponse(status="healthy")

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "reason": f"Health check failed: {str(e)}"
            }
        )


@app.get("/list", response_model=ModelListResponse, tags=["Models"])
async def list_models():
    """
    List all trained models with their metadata.

    Returns information about all models stored in the system.
    """
    try:
        models_data = storage.list_models()

        # Convert to ModelMetadata objects
        models = [
            ModelMetadata(
                model_id=m['model_id'],
                platform_info=m['platform_info'],
                prediction_type=m['prediction_type'],
                samples_count=m['samples_count'],
                last_trained=m['last_trained']
            )
            for m in models_data
        ]

        return ModelListResponse(models=models)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest):
    """
    Train or update a model.

    Trains an MLP model on the provided features and runtime data.
    Supports both expect_error and quantile prediction types.
    """
    try:
        # Validate minimum samples
        if len(request.features_list) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Insufficient training data",
                    "message": f"Need at least 10 samples, got {len(request.features_list)}",
                    "samples_provided": len(request.features_list),
                    "minimum_required": 10
                }
            )

        # Create appropriate predictor
        if request.prediction_type == "expect_error":
            predictor = ExpectErrorPredictor()
        elif request.prediction_type == "quantile":
            predictor = QuantilePredictor()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid prediction type",
                    "message": f"prediction_type must be 'expect_error' or 'quantile', got '{request.prediction_type}'"
                }
            )

        # Train the model
        try:
            training_metadata = predictor.train(
                features_list=request.features_list,
                config=request.training_config or {}
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Training validation error",
                    "message": str(e)
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Training failed",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            )

        # Generate model key
        model_key = storage.generate_model_key(
            model_id=request.model_id,
            platform_info=request.platform_info.model_dump()
        )

        # Save model
        try:
            predictor_state = predictor.get_model_state()
            metadata = {
                'model_id': request.model_id,
                'platform_info': request.platform_info.model_dump(),
                'prediction_type': request.prediction_type,
                'samples_count': len(request.features_list),
                'training_config': request.training_config
            }
            storage.save_model(model_key, predictor_state, metadata)

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Model save failed",
                    "message": str(e)
                }
            )

        return TrainingResponse(
            status="success",
            message=f"Model trained successfully with {len(request.features_list)} samples",
            model_key=model_key,
            samples_trained=len(request.features_list)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Unexpected error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a runtime prediction.

    Returns prediction based on trained model or experiment mode.
    Supports both expect_error and quantile prediction types.
    """
    try:
        # Check if experiment mode
        if is_experiment_mode(request.features, request.platform_info.model_dump()):
            # Generate synthetic prediction
            try:
                result = generate_experiment_prediction(
                    prediction_type=request.prediction_type,
                    features=request.features,
                    config={}
                )

                return PredictionResponse(
                    model_id=request.model_id,
                    platform_info=request.platform_info,
                    prediction_type=request.prediction_type,
                    result=result
                )

            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Experiment mode error",
                        "message": str(e)
                    }
                )

        # Normal mode: load model and predict
        model_key = storage.generate_model_key(
            model_id=request.model_id,
            platform_info=request.platform_info.model_dump()
        )

        # Load model
        model_data = storage.load_model(model_key)
        if model_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Model not found",
                    "message": f"No trained model found for model_id='{request.model_id}' with given platform_info",
                    "model_id": request.model_id,
                    "platform_info": request.platform_info.model_dump(),
                    "model_key": model_key
                }
            )

        # Validate prediction type matches
        stored_prediction_type = model_data['metadata'].get('prediction_type')
        if stored_prediction_type != request.prediction_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Prediction type mismatch",
                    "message": f"Model was trained with prediction_type='{stored_prediction_type}', but request has '{request.prediction_type}'",
                    "model_prediction_type": stored_prediction_type,
                    "request_prediction_type": request.prediction_type
                }
            )

        # Create predictor and load state
        if request.prediction_type == "expect_error":
            predictor = ExpectErrorPredictor()
        elif request.prediction_type == "quantile":
            predictor = QuantilePredictor()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid prediction type",
                    "message": f"prediction_type must be 'expect_error' or 'quantile', got '{request.prediction_type}'"
                }
            )

        predictor.load_model_state(model_data['predictor_state'])

        # Make prediction
        try:
            result = predictor.predict(request.features)

            return PredictionResponse(
                model_id=request.model_id,
                platform_info=request.platform_info,
                prediction_type=request.prediction_type,
                result=result
            )

        except ValueError as e:
            # Feature validation errors
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid features",
                    "message": str(e)
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Prediction failed",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Unexpected error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
