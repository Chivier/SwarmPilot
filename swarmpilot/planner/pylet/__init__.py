"""PyLet integration for SwarmPilot planner.

This package provides PyLet client integration for deploying and managing
model instances via the PyLet cluster management system.
"""

from swarmpilot.planner.pylet.client import (
    InstanceInfo,
    PartialDeploymentError,
    PartialDeploymentResult,
    PyLetClient,
    create_pylet_client,
    get_pylet_client,
)
from swarmpilot.planner.pylet.deployment_executor import (
    DeploymentAction,
    DeploymentExecutor,
    DeploymentPlan,
    ExecutionResult,
)
from swarmpilot.planner.pylet.deployment_service import (
    DeploymentServiceResult,
    PyLetDeploymentService,
    create_pylet_service,
    get_pylet_service,
    get_pylet_service_optional,
)
from swarmpilot.planner.pylet.instance_manager import (
    DeploymentResult,
    InstanceManager,
    ManagedInstance,
    ManagedInstanceStatus,
    create_instance_manager,
    get_instance_manager,
)
from swarmpilot.planner.pylet.migration_executor import (
    MigrationExecutor,
    MigrationOperation,
    MigrationPlan,
    MigrationResult,
    MigrationStatus,
)
from swarmpilot.planner.pylet.scheduler_client import (
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
