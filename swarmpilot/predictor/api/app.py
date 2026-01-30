"""FastAPI application factory and configuration."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from swarmpilot.predictor.api.dependencies import storage
from swarmpilot.predictor.config import get_config
from swarmpilot.predictor.utils.logging import get_logger
from swarmpilot.predictor.utils.logging import setup_logging

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events.

    Args:
        app: The FastAPI application instance.

    Yields:
        None after startup, cleanup runs on shutdown.
    """
    # Startup
    logger.info("FastAPI application starting up")
    storage_info = storage.get_storage_info()
    logger.info(f"Storage initialized: {storage_info['storage_dir']}")
    logger.info(f"Found {storage_info['model_count']} existing models")

    yield

    # Shutdown (cleanup if needed)
    logger.info("FastAPI application shutting down")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    config = get_config()
    application = FastAPI(
        title=config.app_name,
        description=(
            "MLP-based runtime prediction with expect/error and quantile regression"
        ),
        version=config.app_version,
        lifespan=lifespan,
    )

    # Register routes
    _register_routes(application)

    return application


def _register_routes(application: FastAPI) -> None:
    """Register all API routes.

    Args:
        application: The FastAPI application instance.
    """
    from swarmpilot.predictor.api.routes import cache
    from swarmpilot.predictor.api.routes import health
    from swarmpilot.predictor.api.routes import models
    from swarmpilot.predictor.api.routes import prediction
    from swarmpilot.predictor.api.routes import training
    from swarmpilot.predictor.api.routes import websocket

    # Include routers
    application.include_router(health.router)
    application.include_router(cache.router, prefix="/cache")
    application.include_router(models.router)
    application.include_router(training.router)
    application.include_router(prediction.router)

    # Add WebSocket endpoint
    application.add_api_websocket_route("/ws/predict", websocket.websocket_predict)


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Initialize logging when running directly
    run_config = get_config()
    setup_logging(
        log_dir=run_config.log_dir,
        log_level=run_config.log_level,
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
