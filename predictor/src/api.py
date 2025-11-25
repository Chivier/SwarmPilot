"""
FastAPI application for runtime prediction service.

All API endpoints are implemented in this module.
"""

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Tuple
import traceback
import json
import random
import numpy as np
from collections import OrderedDict
import threading

from .models import (
    TrainingRequest, TrainingResponse,
    PredictionRequest, PredictionResponse,
    ModelListResponse, ModelMetadata,
    HealthResponse
)
from .storage.model_storage import ModelStorage
from .predictor.expect_error import ExpectErrorPredictor
from .predictor.quantile import QuantilePredictor
from .predictor.linear_regression import LinearRegressionPredictor
from .predictor.decision_tree import DecisionTreePredictor
from .utils.experiment import is_experiment_mode, generate_experiment_prediction
from .config import get_config
from .utils.logging import get_logger, setup_logging
from .preprocessor.preprocessors_registry import PreprocessorsRegistry

logger = get_logger()


# Initialize storage (will use config when available, otherwise default)
def get_storage() -> ModelStorage:
    """Get ModelStorage instance using current configuration."""
    config = get_config()
    return ModelStorage(storage_dir=config.storage_dir)


storage = get_storage()
preprocessors_registry = PreprocessorsRegistry()


class ModelCache:
    """
    Thread-safe LRU cache for loaded prediction models.

    Caches predictor instances to avoid reloading models from disk on every prediction.
    Uses OrderedDict for LRU eviction policy.
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize model cache.

        Args:
            max_size: Maximum number of models to cache (default: 100)
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Tuple[Any, str]] = OrderedDict()  # key -> (predictor, prediction_type)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, model_key: str) -> Optional[Tuple[Any, str]]:
        """
        Get cached predictor for model_key.

        Args:
            model_key: The model identifier

        Returns:
            Tuple of (predictor, prediction_type) if cached, None otherwise
        """
        with self._lock:
            if model_key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(model_key)
                self._hits += 1
                predictor, pred_type = self._cache[model_key]
                logger.debug(f"Cache hit for model_key={model_key} (hits={self._hits}, misses={self._misses})")
                return (predictor, pred_type)
            else:
                self._misses += 1
                logger.debug(f"Cache miss for model_key={model_key} (hits={self._hits}, misses={self._misses})")
                return None

    def put(self, model_key: str, predictor: Any, prediction_type: str) -> None:
        """
        Cache a predictor instance.

        Args:
            model_key: The model identifier
            predictor: The predictor instance to cache
            prediction_type: Type of prediction ('expect_error' or 'quantile')
        """
        with self._lock:
            # If already exists, move to end
            if model_key in self._cache:
                self._cache.move_to_end(model_key)

            self._cache[model_key] = (predictor, prediction_type)

            # Evict oldest if cache is full
            if len(self._cache) > self.max_size:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                logger.debug(f"Evicted model_key={evicted_key} from cache (size={len(self._cache)})")

            logger.debug(f"Cached model_key={model_key}, cache size={len(self._cache)}")

    def invalidate(self, model_key: str) -> None:
        """
        Remove a model from cache (e.g., after retraining).

        Args:
            model_key: The model identifier to remove
        """
        with self._lock:
            if model_key in self._cache:
                del self._cache[model_key]
                logger.debug(f"Invalidated model_key={model_key} from cache")

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cleared model cache")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2)
            }


# Initialize model cache
model_cache = ModelCache(max_size=100)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("FastAPI application starting up")
    storage_info = storage.get_storage_info()
    logger.info(f"Storage initialized: {storage_info['storage_dir']}")
    logger.info(f"Found {storage_info['model_count']} existing models")

    yield

    # Shutdown (cleanup if needed)
    logger.info("FastAPI application shutting down")


# Initialize FastAPI app with lifespan
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    config = get_config()
    return FastAPI(
        title=config.app_name,
        description="MLP-based runtime prediction with expect/error and quantile regression",
        version=config.app_version,
        lifespan=lifespan
    )


app = create_app()


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


@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """
    Get model cache statistics.

    Returns information about cache performance including hit rate,
    current size, and total hits/misses.
    """
    try:
        stats = model_cache.get_stats()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "cache_stats": stats,
                "message": "Cache statistics retrieved successfully"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to get cache stats",
                "message": str(e)
            }
        )


@app.post("/cache/clear", tags=["Cache"])
async def clear_cache():
    """
    Clear all cached models.

    Useful for freeing memory or forcing model reloads.
    This does not affect stored models on disk.
    """
    try:
        model_cache.clear()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Model cache cleared successfully"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to clear cache",
                "message": str(e)
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
        elif request.prediction_type == "linear_regression":
            predictor = LinearRegressionPredictor()
        elif request.prediction_type == "decision_tree":
            predictor = DecisionTreePredictor()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid prediction type",
                    "message": f"prediction_type must be 'expect_error', 'quantile', 'linear_regression', or 'decision_tree', got '{request.prediction_type}'"
                }
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
                        preprocessor = preprocessors_registry.get_preprocessor(preprocessor_name)
                        target_feature_keys = request.preprocessor_mappings[preprocessor_name]

                        # Validate all required features exist
                        assert all(key in processed_features_dict for key in target_feature_keys), \
                            f"Feature keys {target_feature_keys} not all found in features"

                        # Extract target feature values
                        target_feature_values = [processed_features_dict[key] for key in target_feature_keys]

                        # Apply preprocessor
                        processed_features, remove_origin = preprocessor(target_feature_values)

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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Preprocessor error",
                    "message": str(e)
                }
            )
        # Train the model
        try:
            training_metadata = predictor.train(
                features_list=processed_features_list,
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

        # Generate model key with prediction_type
        model_key = storage.generate_model_key(
            model_id=request.model_id,
            platform_info=request.platform_info.model_dump(),
            prediction_type=request.prediction_type
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

            # Invalidate cache for this model (it has been retrained)
            model_cache.invalidate(model_key)
            logger.info(f"Invalidated cache for retrained model: {model_key}")

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
    random.seed(42)
    np.random.seed(42)
    try:
        # Check if experiment mode
        if is_experiment_mode(request.features, request.platform_info.model_dump()):
            # Generate synthetic prediction
            logger.debug(f"Got experiment mode request, raw request: {request}")
            try:
                # Pass custom quantiles to experiment mode if provided
                config = {}
                if request.quantiles is not None:
                    config['quantiles'] = request.quantiles

                result = generate_experiment_prediction(
                    prediction_type=request.prediction_type,
                    features=request.features,
                    config=config
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
            platform_info=request.platform_info.model_dump(),
            prediction_type=request.prediction_type
        )

        # Try to get predictor from cache
        cached_result = model_cache.get(model_key)

        if cached_result is not None:
            # Cache hit - use cached predictor
            predictor, stored_prediction_type = cached_result

            # Validate prediction type matches
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
        else:
            # Cache miss - load model from storage
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
            elif request.prediction_type == "linear_regression":
                predictor = LinearRegressionPredictor()
            elif request.prediction_type == "decision_tree":
                predictor = DecisionTreePredictor()
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Invalid prediction type",
                        "message": f"prediction_type must be 'expect_error', 'quantile', 'linear_regression', or 'decision_tree', got '{request.prediction_type}'"
                    }
                )

            predictor.load_model_state(model_data['predictor_state'])

            # Cache the loaded predictor
            model_cache.put(model_key, predictor, stored_prediction_type)
            logger.info(f"Loaded and cached model: {model_key}")

        # Make prediction
        try:
            # Append hardware info
            hardware_features = request.platform_info.extract_gpu_specs()
            all_features = request.features.copy()
            if hardware_features:
                for key, value in hardware_features.items():
                    all_features[key] = value

            # Start preprocessing if enabled
            if request.enable_preprocessors:
                for preprocessor_name in request.enable_preprocessors:
                    preprocessor = preprocessors_registry.get_preprocessor(preprocessor_name)
                    target_feature_keys = request.preprocessor_mappings[preprocessor_name]

                    # Validate all required features exist
                    assert all(key in all_features for key in target_feature_keys), \
                        f"Feature keys {target_feature_keys} not all found in features"

                    # Extract target feature values
                    target_feature_values = [all_features[key] for key in target_feature_keys]

                    # Apply preprocessor
                    processed_features, remove_origin = preprocessor(target_feature_values)

                    # Add processed features
                    for k, v in processed_features.items():
                        all_features[k] = v

                    # Remove original features if requested
                    if remove_origin:
                        for key in target_feature_keys:
                            del all_features[key]

            result = predictor.predict(all_features)

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
            logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Unexpected error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time runtime predictions.

    Accepts PredictionRequest JSON messages and returns PredictionResponse JSON.
    Keeps connection open for multiple prediction requests.
    """
    await websocket.accept()

    try:
        while True:
            # Receive JSON message from client
            try:
                data = await websocket.receive_text()
                request_data = json.loads(data)
            except json.JSONDecodeError as e:
                await websocket.send_json({
                    "error": "Invalid JSON",
                    "message": f"Failed to parse JSON: {str(e)}"
                })
                continue
            except Exception as e:
                await websocket.send_json({
                    "error": "Receive error",
                    "message": str(e)
                })
                continue

            # Validate and parse request
            try:
                request = PredictionRequest(**request_data)
            except Exception as e:
                await websocket.send_json({
                    "error": "Invalid request",
                    "message": f"Failed to validate request: {str(e)}"
                })
                continue

            try:
                # Check if experiment mode
                if is_experiment_mode(request.features, request.platform_info.model_dump()):
                    # Generate synthetic prediction
                    logger.debug("request is experiment request")
                    try:
                        # Pass custom quantiles to experiment mode if provided
                        config = {}
                        if request.quantiles is not None:
                            config['quantiles'] = request.quantiles

                        result = generate_experiment_prediction(
                            prediction_type=request.prediction_type,
                            features=request.features,
                            config=config
                        )

                        response = PredictionResponse(
                            model_id=request.model_id,
                            platform_info=request.platform_info,
                            prediction_type=request.prediction_type,
                            result=result
                        )

                        await websocket.send_json(response.model_dump())
                        continue

                    except ValueError as e:
                        await websocket.send_json({
                            "error": "Experiment mode error",
                            "message": str(e)
                        })
                        continue

                # Normal mode: load model and predict
                model_key = storage.generate_model_key(
                    model_id=request.model_id,
                    platform_info=request.platform_info.model_dump(),
                    prediction_type=request.prediction_type
                )

                # Try to get predictor from cache
                cached_result = model_cache.get(model_key)

                if cached_result is not None:
                    # Cache hit - use cached predictor
                    predictor, stored_prediction_type = cached_result

                    # Validate prediction type matches
                    if stored_prediction_type != request.prediction_type:
                        await websocket.send_json({
                            "error": "Prediction type mismatch",
                            "message": f"Model was trained with prediction_type='{stored_prediction_type}', but request has '{request.prediction_type}'",
                            "model_prediction_type": stored_prediction_type,
                            "request_prediction_type": request.prediction_type
                        })
                        continue
                else:
                    # Cache miss - load model from storage
                    model_data = storage.load_model(model_key)
                    if model_data is None:
                        await websocket.send_json({
                            "error": "Model not found",
                            "message": f"No trained model found for model_id='{request.model_id}' with given platform_info",
                            "model_id": request.model_id,
                            "platform_info": request.platform_info.model_dump(),
                            "model_key": model_key
                        })
                        continue

                    # Validate prediction type matches
                    stored_prediction_type = model_data['metadata'].get('prediction_type')
                    if stored_prediction_type != request.prediction_type:
                        await websocket.send_json({
                            "error": "Prediction type mismatch",
                            "message": f"Model was trained with prediction_type='{stored_prediction_type}', but request has '{request.prediction_type}'",
                            "model_prediction_type": stored_prediction_type,
                            "request_prediction_type": request.prediction_type
                        })
                        continue

                    # Create predictor and load state
                    if request.prediction_type == "expect_error":
                        predictor = ExpectErrorPredictor()
                    elif request.prediction_type == "quantile":
                        predictor = QuantilePredictor()
                    elif request.prediction_type == "linear_regression":
                        predictor = LinearRegressionPredictor()
                    elif request.prediction_type == "decision_tree":
                        predictor = DecisionTreePredictor()
                    else:
                        await websocket.send_json({
                            "error": "Invalid prediction type",
                            "message": f"prediction_type must be 'expect_error', 'quantile', 'linear_regression', or 'decision_tree', got '{request.prediction_type}'"
                        })
                        continue

                    predictor.load_model_state(model_data['predictor_state'])

                    # Cache the loaded predictor
                    model_cache.put(model_key, predictor, stored_prediction_type)
                    logger.info(f"Loaded and cached model (WebSocket): {model_key}")

                # Make prediction
                try:
                    result = predictor.predict(request.features)

                    response = PredictionResponse(
                        model_id=request.model_id,
                        platform_info=request.platform_info,
                        prediction_type=request.prediction_type,
                        result=result
                    )

                    await websocket.send_json(response.model_dump())

                except ValueError as e:
                    # Feature validation errors
                    await websocket.send_json({
                        "error": "Invalid features",
                        "message": str(e)
                    })
                except Exception as e:
                    await websocket.send_json({
                        "error": "Prediction failed",
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    })

            except Exception as e:
                await websocket.send_json({
                    "error": "Unexpected error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                })

    except WebSocketDisconnect:
        # Client disconnected, close gracefully
        pass
    except Exception as e:
        # Unexpected error, try to send error message before closing
        try:
            await websocket.send_json({
                "error": "WebSocket error",
                "message": str(e)
            })
        except:
            pass
        finally:
            await websocket.close()


if __name__ == "__main__":
    import uvicorn
    from .config import get_config

    # Initialize logging when running directly
    config = get_config()
    setup_logging(
        log_dir=config.log_dir,
        log_level=config.log_level
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
