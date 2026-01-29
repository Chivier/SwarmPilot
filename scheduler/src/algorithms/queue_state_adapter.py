"""Queue state adapter for PYLET-018: Queue-Aware Scheduling Integration.

This module provides adapter functions that convert scheduler-side queue state
(from WorkerQueueManager) to the format existing scheduling algorithms expect
(InstanceQueueExpectError).

The adapter pattern allows us to integrate scheduler-side queues without
modifying the existing algorithms - they continue to receive queue info in
their expected format.

Example:
    ```python
    # Get queue info for a single instance
    queue_info = get_queue_info_from_manager(
        worker_queue_manager=manager,
        instance_id="worker-1",
        avg_exec_time_ms=100.0,
    )

    # Get queue info for multiple instances
    all_queue_info = get_all_queue_info_from_manager(
        worker_queue_manager=manager,
        instance_ids=["worker-1", "worker-2"],
        avg_exec_time_ms=100.0,
    )
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.queue import InstanceQueueExpectError

if TYPE_CHECKING:
    from src.services.worker_queue_manager import WorkerQueueManager


# Default error margin as percentage of average execution time
ERROR_MARGIN_PERCENT = 0.2


def get_queue_info_from_manager(
    worker_queue_manager: WorkerQueueManager | None,
    instance_id: str,
    avg_exec_time_ms: float,
) -> InstanceQueueExpectError:
    """Convert scheduler-side queue state to algorithm format.

    This adapter function takes the queue state from WorkerQueueManager
    and converts it to the InstanceQueueExpectError format that existing
    scheduling algorithms expect.

    Args:
        worker_queue_manager: Manager with queue state. Can be None for
            graceful degradation.
        instance_id: Instance to query.
        avg_exec_time_ms: Average execution time per task in milliseconds.
            Used for wait time estimation and error margin calculation.

    Returns:
        Queue info in InstanceQueueExpectError format:
        - expected_time_ms: Estimated wait time based on queue depth
        - error_margin_ms: Uncertainty margin (20% of avg_exec_time_ms)

    Example:
        ```python
        queue_info = get_queue_info_from_manager(
            worker_queue_manager=manager,
            instance_id="worker-1",
            avg_exec_time_ms=100.0,
        )
        # queue_info.expected_time_ms = 500.0 (5 tasks * 100ms avg)
        # queue_info.error_margin_ms = 20.0 (20% of 100ms)
        ```
    """
    # Handle None manager gracefully
    if worker_queue_manager is None:
        return InstanceQueueExpectError(
            instance_id=instance_id,
            expected_time_ms=0.0,
            error_margin_ms=0.0,
        )

    # Get worker thread for this instance
    thread = worker_queue_manager.get_worker(instance_id)
    if thread is None:
        return InstanceQueueExpectError(
            instance_id=instance_id,
            expected_time_ms=0.0,
            error_margin_ms=0.0,
        )

    # Calculate estimated wait time from worker thread
    wait_time = thread.get_estimated_wait_time(avg_exec_time_ms)

    # Calculate error margin as 20% of average execution time
    error_margin = avg_exec_time_ms * ERROR_MARGIN_PERCENT

    return InstanceQueueExpectError(
        instance_id=instance_id,
        expected_time_ms=wait_time,
        error_margin_ms=error_margin,
    )


def get_all_queue_info_from_manager(
    worker_queue_manager: WorkerQueueManager | None,
    instance_ids: list[str],
    avg_exec_time_ms: float,
) -> dict[str, InstanceQueueExpectError]:
    """Get queue info for multiple instances.

    This is a batch version of get_queue_info_from_manager for efficiency
    when collecting queue state for all available instances.

    Args:
        worker_queue_manager: Manager with queue state. Can be None for
            graceful degradation.
        instance_ids: List of instance IDs to query.
        avg_exec_time_ms: Average execution time per task in milliseconds.

    Returns:
        Dictionary mapping instance_id to InstanceQueueExpectError.

    Example:
        ```python
        all_queue_info = get_all_queue_info_from_manager(
            worker_queue_manager=manager,
            instance_ids=["worker-1", "worker-2", "worker-3"],
            avg_exec_time_ms=100.0,
        )
        # Returns: {"worker-1": <queue_info>, "worker-2": <queue_info>, ...}
        ```
    """
    result = {}
    for instance_id in instance_ids:
        result[instance_id] = get_queue_info_from_manager(
            worker_queue_manager=worker_queue_manager,
            instance_id=instance_id,
            avg_exec_time_ms=avg_exec_time_ms,
        )
    return result
