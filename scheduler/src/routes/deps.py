"""Dependency injection for route handlers.

This module provides functions that return references to global service
instances. Using functions instead of direct imports avoids circular
import issues and makes testing easier.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..background_scheduler import BackgroundScheduler
    from ..central_queue import CentralTaskQueue
    from ..instance_registry import InstanceRegistry
    from ..predictor_client import PredictorClient
    from ..task_dispatcher import TaskDispatcher
    from ..task_registry import TaskRegistry
    from ..throughput_tracker import ThroughputTracker
    from ..training_client import TrainingClient
    from ..websocket_manager import ConnectionManager


def get_instance_registry() -> "InstanceRegistry":
    """Get the instance registry singleton."""
    from ..api import instance_registry

    return instance_registry


def get_task_registry() -> "TaskRegistry":
    """Get the task registry singleton."""
    from ..api import task_registry

    return task_registry


def get_websocket_manager() -> "ConnectionManager":
    """Get the websocket manager singleton."""
    from ..api import websocket_manager

    return websocket_manager


def get_predictor_client() -> "PredictorClient":
    """Get the predictor client singleton."""
    from ..api import predictor_client

    return predictor_client


def get_training_client() -> "TrainingClient | None":
    """Get the training client singleton (may be None)."""
    from ..api import training_client

    return training_client


def get_task_dispatcher() -> "TaskDispatcher":
    """Get the task dispatcher singleton."""
    from ..api import task_dispatcher

    return task_dispatcher


def get_background_scheduler() -> "BackgroundScheduler":
    """Get the background scheduler singleton."""
    from ..api import background_scheduler

    return background_scheduler


def get_central_queue() -> "CentralTaskQueue":
    """Get the central queue singleton."""
    from ..api import central_queue

    return central_queue


def get_scheduling_strategy():
    """Get the current scheduling strategy."""
    from ..api import scheduling_strategy

    return scheduling_strategy


def get_planner_reporter():
    """Get the planner reporter singleton (may be None)."""
    from ..api import planner_reporter

    return planner_reporter


def get_throughput_tracker() -> "ThroughputTracker | None":
    """Get the throughput tracker singleton (may be None)."""
    from ..api import throughput_tracker

    return throughput_tracker


def get_clearing_state():
    """Get the clearing in progress state and lock.

    Returns:
        Tuple of (_clearing_in_progress reference, _clearing_lock)
    """
    from ..api import _clearing_in_progress, _clearing_lock

    return _clearing_in_progress, _clearing_lock


def get_background_tasks():
    """Get the background tasks set."""
    from ..api import _background_tasks

    return _background_tasks
