"""State management package for the Planner service.

This package provides centralized global state management.
All state variables are re-exported here for convenience.
"""

from .app_state import (
    _auto_optimize_running,
    _auto_optimize_task,
    _current_target,
    _first_data_received,
    _first_migration_done,
    _optimization_timer_start,
    _stored_deployment_input,
    _stored_model_mapping,
    _stored_reverse_mapping,
    _submitted_models,
    _throughput_data,
    reset_state,
)

__all__ = [
    "_stored_model_mapping",
    "_stored_reverse_mapping",
    "_current_target",
    "_submitted_models",
    "_auto_optimize_running",
    "_stored_deployment_input",
    "_auto_optimize_task",
    "_first_data_received",
    "_first_migration_done",
    "_optimization_timer_start",
    "_throughput_data",
    "reset_state",
]
