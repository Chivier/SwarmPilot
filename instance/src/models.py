"""
Core data models for Instance Service
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InstanceStatus(str, Enum):
    """Instance status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    BUSY = "busy"
    ERROR = "error"


class RestartStatus(str, Enum):
    """Restart operation status enumeration"""
    PENDING = "pending"
    DRAINING = "draining"
    EXTRACTING_TASKS = "extracting_tasks"
    WAITING_RUNNING_TASK = "waiting_running_task"
    STOPPING_MODEL = "stopping_model"
    DEREGISTERING = "deregistering"
    STARTING_MODEL = "starting_model"
    REGISTERING = "registering"
    COMPLETED = "completed"
    FAILED = "failed"

class DeregisterStatus(str, Enum):
    """Restart operation status enumeration"""
    PENDING = "pending"
    DRAINING = "draining"
    EXTRACTING_TASKS = "extracting_tasks"
    WAITING_RUNNING_TASK = "waiting_running_task"
    DEREGISTERING = "deregistering"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Hot-Standby Port System Models
# =============================================================================

class PortRole(str, Enum):
    """
    Role of a port in the hot-standby system.

    PRIMARY: Currently active, serving inference requests
    STANDBY: Hot-standby, ready to take over on restart
    """
    PRIMARY = "primary"
    STANDBY = "standby"


class PortState(str, Enum):
    """
    State of an individual port's process lifecycle.

    State transitions:
    UNINITIALIZED -> STARTING -> HEALTHY -> STOPPING -> STOPPED
                           |-> FAILED (can happen from STARTING)
    STOPPED can transition back to STARTING (for restart scenarios)
    """
    UNINITIALIZED = "uninitialized"  # No process has been started
    STARTING = "starting"             # Process spawned, waiting for health
    HEALTHY = "healthy"               # Process healthy, ready to serve
    STOPPING = "stopping"             # Graceful shutdown in progress
    STOPPED = "stopped"               # Process terminated
    FAILED = "failed"                 # Process failed to start or crashed


@dataclass
class PortInfo:
    """
    Information about a single port/process pair in the hot-standby system.

    Tracks everything needed to manage one model process instance.
    """
    port: int
    role: PortRole
    state: PortState = PortState.UNINITIALIZED
    process: Optional[asyncio.subprocess.Process] = None
    started_at: Optional[str] = None
    last_health_check: Optional[str] = None
    health_check_failures: int = 0
    error_message: Optional[str] = None

    # Retry tracking during standby startup
    startup_attempts: int = 0
    max_startup_attempts: int = 3

    def is_ready(self) -> bool:
        """Check if port is ready to serve traffic."""
        return self.state == PortState.HEALTHY

    def can_start(self) -> bool:
        """Check if port can be started (not already running)."""
        return self.state in (PortState.UNINITIALIZED, PortState.STOPPED, PortState.FAILED)

    def reset_for_restart(self):
        """Reset state for a new start attempt."""
        self.state = PortState.UNINITIALIZED
        self.process = None
        self.started_at = None
        self.last_health_check = None
        self.health_check_failures = 0
        self.error_message = None
        self.startup_attempts += 1


@dataclass
class RuntimeStandbyConfig:
    """
    Runtime configuration for standby (hot-standby) behavior.

    This stores the effective standby configuration for a running model,
    combining environment variable defaults with API request overrides.
    Used by SubprocessManager to control standby behavior during model lifecycle.
    """
    enabled: bool = True
    port_offset: int = 1000
    max_retries: int = 3
    initial_delay: float = 5.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    restart_delay: int = 30
    health_check_timeout: int = 600
    traditional_restart_delay: int = 30

    @classmethod
    def from_config_and_overrides(
        cls,
        config,
        standby_enabled: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> "RuntimeStandbyConfig":
        """
        Create RuntimeStandbyConfig from environment config and API overrides.

        Args:
            config: The global config object with environment variable defaults
            standby_enabled: Override for standby enabled (from API request)
            overrides: Dict of config overrides (from API standby_config)

        Returns:
            RuntimeStandbyConfig with merged values
        """
        # Start with environment defaults
        instance = cls(
            enabled=config.standby_enabled,
            port_offset=config.standby_port_offset,
            max_retries=config.hot_standby_max_retries,
            initial_delay=config.hot_standby_initial_delay,
            max_delay=config.hot_standby_max_delay,
            backoff_multiplier=config.hot_standby_backoff_multiplier,
            restart_delay=config.standby_restart_delay,
            health_check_timeout=config.backup_health_check_timeout,
            traditional_restart_delay=config.traditional_restart_delay,
        )

        # Override with API request values
        if standby_enabled is not None:
            instance.enabled = standby_enabled

        if overrides:
            if "port_offset" in overrides:
                instance.port_offset = overrides["port_offset"]
            if "max_retries" in overrides:
                instance.max_retries = overrides["max_retries"]
            if "initial_delay" in overrides:
                instance.initial_delay = overrides["initial_delay"]
            if "max_delay" in overrides:
                instance.max_delay = overrides["max_delay"]
            if "backoff_multiplier" in overrides:
                instance.backoff_multiplier = overrides["backoff_multiplier"]
            if "restart_delay" in overrides:
                instance.restart_delay = overrides["restart_delay"]
            if "health_check_timeout" in overrides:
                instance.health_check_timeout = overrides["health_check_timeout"]
            if "traditional_restart_delay" in overrides:
                instance.traditional_restart_delay = overrides["traditional_restart_delay"]

        return instance


@dataclass
class DualPortState:
    """
    Container for managing the two-port hot-standby system.

    Key invariants:
    1. Exactly one port should have role=PRIMARY when system is active
    2. At most one port should have role=STANDBY
    3. The active_port property returns the port currently serving traffic
    4. After a swap, roles change but port objects remain

    Port naming convention:
    - port_a: Uses base model_port (e.g., instance_port + 1000)
    - port_b: Uses backup port (e.g., instance_port + 2000)
    """
    port_a: PortInfo
    port_b: PortInfo
    _primary_port_name: str = "port_a"  # "port_a" or "port_b"

    @property
    def primary(self) -> PortInfo:
        """Get the current primary (active) port."""
        return self.port_a if self._primary_port_name == "port_a" else self.port_b

    @property
    def standby(self) -> PortInfo:
        """Get the current standby port."""
        return self.port_b if self._primary_port_name == "port_a" else self.port_a

    @property
    def active_port(self) -> int:
        """
        Get the port number currently serving traffic.

        This is what external code should use for inference/health checks.
        """
        return self.primary.port

    def swap_roles(self):
        """
        Swap PRIMARY and STANDBY roles between ports.

        This is the core operation for hot-switching:
        - The standby becomes primary
        - The old primary becomes standby

        Precondition: standby must be HEALTHY before calling
        """
        old_primary = self.primary
        old_standby = self.standby

        old_primary.role = PortRole.STANDBY
        old_standby.role = PortRole.PRIMARY

        self._primary_port_name = "port_b" if self._primary_port_name == "port_a" else "port_a"

    def get_port_by_role(self, role: PortRole) -> PortInfo:
        """Get port by its current role."""
        if role == PortRole.PRIMARY:
            return self.primary
        return self.standby


class Task(BaseModel):
    """Task data model"""
    task_id: str = Field(..., description="Unique identifier for this task")
    model_id: str = Field(..., description="Model/tool ID to use for this task")
    task_input: Dict[str, Any] = Field(..., description="Model-specific input data")
    status: TaskStatus = Field(default=TaskStatus.QUEUED, description="Current task status")
    callback_url: Optional[str] = Field(None, description="Optional callback URL for task result")

    # Timestamps
    submitted_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    enqueue_time: float = Field(
        default_factory=time.time,
        description="Unix timestamp when task was enqueued, used for priority queue ordering"
    )

    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def mark_started(self):
        """Mark task as started"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def mark_completed(self, result: Dict[str, Any]):
        """Mark task as completed with result"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.result = result

    def mark_failed(self, error: str):
        """Mark task as failed with error"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.error = error


class ModelInfo(BaseModel):
    """Information about a running model"""
    model_id: str
    started_at: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    container_name: Optional[str] = None


class ModelRegistryEntry(BaseModel):
    """Model registry entry"""
    model_id: str
    name: str
    directory: str
    resource_requirements: Dict[str, Any]


class RestartOperation(BaseModel):
    """Tracks the state of a model restart operation"""
    operation_id: str = Field(..., description="Unique identifier for this restart operation")
    status: RestartStatus = Field(default=RestartStatus.PENDING, description="Current restart status")
    old_model_id: Optional[str] = None
    new_model_id: str
    new_parameters: Dict[str, Any] = Field(default_factory=dict)
    new_scheduler_url: Optional[str] = None

    # Timestamps
    initiated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    completed_at: Optional[str] = None

    # Progress tracking
    pending_tasks_at_start: int = 0
    pending_tasks_completed: int = 0
    redistributed_tasks_count: int = 0
    error: Optional[str] = None

    def update_status(self, new_status: RestartStatus, error: Optional[str] = None):
        """Update the operation status"""
        self.status = new_status
        if error:
            self.error = error
        if new_status in (RestartStatus.COMPLETED, RestartStatus.FAILED):
            self.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

class DeregisterOperation(BaseModel):
    """Tracks the state of a model deregister operation"""
    operation_id: str = Field(..., description="Unique identifier for this restart operation")
    status: DeregisterStatus = Field(default=DeregisterStatus.PENDING, description="Current deregister status")
    old_model_id: Optional[str] = None
    
    # Timestamps
    initiated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    completed_at: Optional[str] = None

    # Progress tracking
    pending_tasks_at_start: int = 0
    pending_tasks_completed: int = 0
    redistributed_tasks_count: int = 0
    error: Optional[str] = None

    def update_status(self, new_status: DeregisterStatus, error: Optional[str] = None):
        """Update the operation status"""
        self.status = new_status
        if error:
            self.error = error
        if new_status in (DeregisterStatus.COMPLETED, DeregisterStatus.FAILED):
            self.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")