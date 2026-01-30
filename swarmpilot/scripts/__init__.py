"""SwarmPilot PyLet deployment and registration scripts.

This module provides utilities for deploying model services via PyLet
and registering them with the SwarmPilot scheduler.

Migration Tasks:
- PYLET-001: Direct model deployment via PyLet
- PYLET-002: Model registration with scheduler
- PYLET-003: Signal handling for graceful shutdown
- PYLET-004: Health monitoring
"""

from swarmpilot.scripts.deploy import (
    MODEL_COMMANDS,
    deploy_model,
    wait_model_ready,
    wait_model_ready_sync,
)
from swarmpilot.scripts.health import (
    HEALTH_ENDPOINTS,
    HeartbeatSender,
    HealthMonitor,
    check_health_once,
    wait_for_health,
)
from swarmpilot.scripts.register import (
    deregister_from_scheduler,
    force_remove_from_scheduler,
    get_instance_info,
    register_with_scheduler,
)
from swarmpilot.scripts.signal_handler import (
    GracefulShutdown,
    create_shutdown_handler,
)

__all__ = [
    # Deployment (PYLET-001)
    "MODEL_COMMANDS",
    "deploy_model",
    "wait_model_ready",
    "wait_model_ready_sync",
    # Registration (PYLET-002)
    "get_instance_info",
    "register_with_scheduler",
    "deregister_from_scheduler",
    "force_remove_from_scheduler",
    # Signal Handling (PYLET-003)
    "GracefulShutdown",
    "create_shutdown_handler",
    # Health Monitoring (PYLET-004)
    "HEALTH_ENDPOINTS",
    "wait_for_health",
    "check_health_once",
    "HeartbeatSender",
    "HealthMonitor",
]

__version__ = "0.1.0"
