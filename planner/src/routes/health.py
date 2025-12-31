"""Health check and service info endpoints."""

from datetime import UTC, datetime

from fastapi import APIRouter

from .. import __version__
from ..available_instance_store import get_available_instance_store

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancing.

    Returns:
        Health status with timestamp
    """
    return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}


@router.get("/info")
async def service_info():
    """Get service information and capabilities.

    Returns:
        Service metadata including version and supported algorithms
    """
    return {
        "service": "planner",
        "version": __version__,
        "algorithms": ["simulated_annealing", "integer_programming"],
        "objective_methods": [
            "relative_error",
            "ratio_difference",
            "weighted_squared",
        ],
        "description": "Model deployment optimization service",
        "available_instances": get_available_instance_store().available_instances,
    }
