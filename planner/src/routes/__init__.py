"""HTTP route handlers for the Planner service.

This package organizes FastAPI routes by domain:
- health: Health check and service info
- planning: Optimization planning endpoints
- instance: Instance registration and migration info
- target: Target distribution and throughput endpoints
- scheduler_compat: Scheduler-compatible dummy endpoints
- timeline: Timeline tracking endpoints

Note: Complex deployment endpoints (/deploy, /deploy/migration) remain in
api.py due to their heavy state manipulation requirements.
"""

from fastapi import APIRouter

from .health import router as health_router
from .instance import router as instance_router
from .planning import router as planning_router
from .scheduler_compat import router as scheduler_compat_router
from .target import router as target_router
from .timeline import router as timeline_router


def create_api_router() -> APIRouter:
    """Create the main API router with all sub-routers included.

    Returns:
        APIRouter with all domain-specific routers included
    """
    api_router = APIRouter()
    api_router.include_router(health_router)
    api_router.include_router(planning_router)
    api_router.include_router(instance_router)
    api_router.include_router(target_router)
    api_router.include_router(scheduler_compat_router)
    api_router.include_router(timeline_router)
    return api_router


__all__ = [
    "create_api_router",
    "health_router",
    "planning_router",
    "instance_router",
    "target_router",
    "scheduler_compat_router",
    "timeline_router",
]
