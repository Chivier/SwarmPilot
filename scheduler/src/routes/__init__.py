"""FastAPI route modules for the scheduler service.

This package organizes API endpoints into logical groups:
- instance: Instance management (register, remove, drain, list, info)
- task: Task submission and management
- callback: Task result callbacks
- websocket: WebSocket connections for real-time updates
- strategy: Scheduling strategy management
- health: Health check endpoints
"""

from fastapi import APIRouter

from .callback import router as callback_router
from .health import router as health_router
from .instance import router as instance_router
from .strategy import router as strategy_router
from .task import router as task_router
from .websocket import router as websocket_router

# Create a parent router that includes all sub-routers
api_router = APIRouter()

# Include all route modules with endpoints
api_router.include_router(callback_router)
api_router.include_router(health_router)
api_router.include_router(strategy_router)
api_router.include_router(websocket_router)

# Note: instance_router and task_router are placeholders
# Their endpoints remain in api.py for now

__all__ = [
    "api_router",
    "instance_router",
    "task_router",
    "callback_router",
    "websocket_router",
    "strategy_router",
    "health_router",
]
