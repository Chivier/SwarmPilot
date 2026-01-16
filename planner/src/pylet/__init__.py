"""PyLet integration for SwarmPilot planner.

This package provides PyLet client integration for deploying and managing
model instances via the PyLet cluster management system.
"""

from src.pylet.client import (
    InstanceInfo,
    PartialDeploymentError,
    PartialDeploymentResult,
    PyLetClient,
    create_pylet_client,
    get_pylet_client,
)
from src.pylet.deployment_executor import (
    DeploymentAction,
    DeploymentExecutor,
    DeploymentPlan,
    ExecutionResult,
)
from src.pylet.deployment_service import (
    DeploymentServiceResult,
    PyLetDeploymentService,
    create_pylet_service,
    get_pylet_service,
    get_pylet_service_optional,
)
from src.pylet.instance_manager import (
    DeploymentResult,
    InstanceManager,
    ManagedInstance,
    ManagedInstanceStatus,
    create_instance_manager,
    get_instance_manager,
)
from src.pylet.migration_executor import (
    MigrationExecutor,
    MigrationOperation,
    MigrationPlan,
    MigrationResult,
    MigrationStatus,
)
from src.pylet.scheduler_client import (
    RegistrationInfo,
    SchedulerClient,
    create_scheduler_client,
    get_scheduler_client,
)

__all__ = [
    # PyLet client
    "InstanceInfo",
    "PartialDeploymentError",
    "PartialDeploymentResult",
    "PyLetClient",
    "create_pylet_client",
    "get_pylet_client",
    # Deployment executor
    "DeploymentAction",
    "DeploymentExecutor",
    "DeploymentPlan",
    "ExecutionResult",
    # Instance manager
    "DeploymentResult",
    "InstanceManager",
    "ManagedInstance",
    "ManagedInstanceStatus",
    "create_instance_manager",
    "get_instance_manager",
    # Migration executor
    "MigrationExecutor",
    "MigrationOperation",
    "MigrationPlan",
    "MigrationResult",
    "MigrationStatus",
    # Scheduler client
    "RegistrationInfo",
    "SchedulerClient",
    "create_scheduler_client",
    "get_scheduler_client",
    # Deployment service
    "DeploymentServiceResult",
    "PyLetDeploymentService",
    "create_pylet_service",
    "get_pylet_service",
    "get_pylet_service_optional",
]
