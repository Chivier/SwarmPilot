"""Global application state for the Planner service.

This module centralizes all mutable global state that was previously scattered
throughout api.py. State is exposed at module level for backward compatibility.

Note: Tests access these variables via `api_module._stored_model_mapping` etc.
We maintain this pattern by re-exporting from the state module in api.py.
"""

import asyncio
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..models import DeploymentInput

# Global state for target distribution and model mapping
# These are set when /deploy or /deploy/migration is called
_stored_model_mapping: dict[str, int] | None = None
_stored_reverse_mapping: dict[int, str] | None = None
_current_target: list[float] | None = None

# Global state for auto-optimization
_submitted_models: set = set()  # Track which models have submitted targets
_auto_optimize_running: bool = False  # Flag to prevent concurrent optimizations
_stored_deployment_input: Optional["DeploymentInput"] = None  # Stored for auto-optimization
_auto_optimize_task: asyncio.Task | None = None  # Background task handle

# New state for event-driven optimization timing
_first_data_received: bool = False  # True after first /submit_target in current cycle
_first_migration_done: bool = False  # True after first /deploy/migration completed
_optimization_timer_start: float = 0.0  # Timestamp when optimization timer starts

# Throughput data storage for B matrix updates
# Structure: {instance_url: {model_id: capacity}}
_throughput_data: dict[str, dict[str, float]] = {}


def reset_state() -> None:
    """Reset all global state to initial values.

    Useful for testing and application restart scenarios.
    """
    global _stored_model_mapping, _stored_reverse_mapping, _current_target
    global _submitted_models, _auto_optimize_running, _stored_deployment_input
    global _auto_optimize_task, _first_data_received, _first_migration_done
    global _optimization_timer_start, _throughput_data

    _stored_model_mapping = None
    _stored_reverse_mapping = None
    _current_target = None
    _submitted_models = set()
    _auto_optimize_running = False
    _stored_deployment_input = None
    _auto_optimize_task = None
    _first_data_received = False
    _first_migration_done = False
    _optimization_timer_start = 0.0
    _throughput_data = {}
