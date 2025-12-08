"""
FastAPI application for the scheduler service.

This module defines all API endpoints for instance management, task scheduling,
and WebSocket connections for real-time task result delivery.
"""

from typing import Optional, Callable, List
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio
import math
import random
import numpy as np
import json
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Request
from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer
from pyinstrument.renderers.speedscope import SpeedscopeRenderer
from loguru import logger

from .http_error_logger import log_http_error

from .model import (
    # Instance models
    InstanceRegisterRequest,
    InstanceRegisterResponse,
    InstanceRemoveRequest,
    InstanceRemoveResponse,
    InstanceDrainRequest,
    InstanceDrainResponse,
    InstanceDrainStatusResponse,
    InstanceRedeployRequest,
    InstanceRedeployResponse,
    InstanceListResponse,
    InstanceInfoResponse,
    Instance,
    InstanceStatus,
    InstanceStats,
    InstanceQueueBase,
    InstanceQueueProbabilistic,
    InstanceQueueExpectError,
    # Task models
    TaskSubmitRequest,
    TaskSubmitResponse,
    TaskListResponse,
    TaskDetailResponse,
    TaskClearResponse,
    TaskResubmitRequest,
    TaskResubmitResponse,
    TaskUpdateMetadataRequest,
    TaskUpdateMetadataResponse,
    TaskUpdateMetadataResult,
    TaskRepredictResponse,
    TaskScheduleInfo,
    TaskScheduleInfoResponse,
    TaskStatus,
    TaskInfo,
    TaskSummary,
    TaskDetailInfo,
    TaskTimestamps,
    # Strategy models
    StrategyType,
    StrategySetRequest,
    StrategySetResponse,
    StrategyGetResponse,
    StrategyInfo,
    # Callback models
    TaskResultCallbackRequest,
    TaskResultCallbackResponse,
    # Health models
    HealthResponse,
    HealthErrorResponse,
    HealthStats,
    # WebSocket models
    WSSubscribeMessage,
    WSUnsubscribeMessage,
    WSAckMessage,
    WSTaskResultMessage,
    WSErrorMessage,
    WSPingMessage,
    WSPongMessage,
    WSMessageType,
    # Common models
    ErrorResponse,
)

from .instance_registry import InstanceRegistry
from .task_registry import TaskRegistry
from .websocket_manager import ConnectionManager
from .predictor_client import PredictorClient
from .scheduler import get_strategy
from .task_dispatcher import TaskDispatcher
from .background_scheduler import BackgroundScheduler
from .central_queue import CentralTaskQueue
from .planner_reporter import PlannerReporter
from .throughput_tracker import ThroughputTracker

# Import logger configuration to initialize loguru
from . import logger as logger_module  # noqa: F401

random.seed(42)
# ============================================================================
# Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown."""
    # Startup
    logger.info("Scheduler service starting up...")

    # Import config here to access global state
    from .config import config
    logger.info(f"Configuration: {config}")
    logger.info(f"Scheduling strategy: {config.scheduling.default_strategy}")
    logger.info(f"Auto-training enabled: {config.training.enable_auto_training}")

    # TODO: Load persisted state if needed
    # TODO: Connect to external services

    # Start central queue dispatcher
    await central_queue.start()
    logger.info("Central queue dispatcher started")

    # Start planner reporter (if configured)
    if planner_reporter:
        await planner_reporter.start()
        logger.info("Planner reporter started")

    logger.success("Scheduler service started successfully")

    yield

    # Shutdown
    logger.info("Scheduler service shutting down...")

    # Shutdown planner reporter first (if configured)
    if planner_reporter:
        await planner_reporter.shutdown()
        logger.debug("Planner reporter shutdown complete")

    # Shutdown central queue (wait for pending dispatches)
    await central_queue.shutdown()
    logger.debug("Central queue shutdown complete")

    # Shutdown background scheduler (wait for active tasks to complete)
    await background_scheduler.shutdown()
    logger.debug("Background scheduler shutdown complete")

    # Close HTTP clients
    await task_dispatcher.close()
    logger.debug("Task dispatcher closed")

    if training_client:
        await training_client.close()
        logger.debug("Training client closed")

    await predictor_client.close()
    logger.debug("Predictor client closed")

    # TODO: Persist state if needed
    logger.info("Scheduler service shutdown complete")


app = FastAPI(
    title="Scheduler API",
    description="Task scheduling and instance management service",
    version="1.0.0",
    lifespan=lifespan,
)

logger.info("Initializing Scheduler API")

@app.middleware("http")
async def profile_request(request: Request, call_next: Callable):
    """Profile the current request

    Taken from https://pyinstrument.readthedocs.io/en/latest/guide.html#profile-a-web-request-in-fastapi
    with small improvements.

    """
    # we map a profile type to a file extension, as well as a pyinstrument profile renderer
    profile_type_to_ext = {"html": "html", "speedscope": "speedscope.json"}
    profile_type_to_renderer = {
        "html": HTMLRenderer,
        "speedscope": SpeedscopeRenderer,
    }

    # if the `profile=true` HTTP query argument is passed, we profile the request
    if request.query_params.get("profile", False):

        # The default profile format is speedscope
        profile_type = request.query_params.get("profile_format", "speedscope")

        # we profile the request along with all additional middlewares, by interrupting
        # the program every 1ms1 and records the entire stack at that point
        with Profiler(interval=0.001, async_mode="enabled") as profiler:
            response = await call_next(request)

        # we dump the profiling into a file
        extension = profile_type_to_ext[profile_type]
        renderer = profile_type_to_renderer[profile_type]()
        with open(f"profile.{extension}", "w") as out:
            out.write(profiler.output(renderer=renderer))
        profiler.print()
        return response

    # Proceed without profiling
    return await call_next(request)

# Initialize configuration
from .config import config
from .training_client import TrainingClient

# Determine queue info type based on scheduling strategy
queue_info_type = "expect_error" if config.scheduling.default_strategy == "min_time" else "probabilistic"

# Global state - TODO: Consider dependency injection for better testability
instance_registry = InstanceRegistry(queue_info_type=queue_info_type)
task_registry = TaskRegistry()
websocket_manager = ConnectionManager()  # Client WebSocket for task result notifications

predictor_client = PredictorClient(
    predictor_url=config.predictor.url,
    timeout=config.predictor.timeout,
    max_retries=config.predictor.max_retries,
    retry_delay=config.predictor.retry_delay,
)

# Initialize training client
training_client = TrainingClient(
    predictor_url=config.predictor.url,
    batch_size=config.training.batch_size,
    min_samples=config.training.min_samples,
    prediction_types=config.training.prediction_types,
) if config.training.enable_auto_training else None

# Initialize task dispatcher
# Construct callback base URL from config
callback_base_url = f"http://{config.server.host}:{config.server.port}"
if config.server.host == "0.0.0.0":
    # For 0.0.0.0, use localhost for callbacks
    callback_base_url = f"http://localhost:{config.server.port}"

task_dispatcher = TaskDispatcher(
    task_registry=task_registry,
    instance_registry=instance_registry,
    websocket_manager=websocket_manager,
    training_client=training_client,
    callback_base_url=callback_base_url,
)

# Initialize scheduling strategy from configuration
# Note: strategy now receives predictor_client and instance_registry dependencies
scheduling_strategy = get_strategy(
    strategy_name=config.scheduling.default_strategy,
    predictor_client=predictor_client,
    instance_registry=instance_registry,
    target_quantile=config.scheduling.probabilistic_quantile,
)

# Initialize background scheduler for non-blocking task scheduling
background_scheduler = BackgroundScheduler(
    scheduling_strategy=scheduling_strategy,
    task_registry=task_registry,
    instance_registry=instance_registry,
    task_dispatcher=task_dispatcher,
    max_concurrent_scheduling=50,  # Limit concurrent scheduling operations
)

logger.info("Background scheduler initialized with max_concurrent=50")

# Initialize central task queue
central_queue = CentralTaskQueue(
    task_registry=task_registry,
    instance_registry=instance_registry,
    max_concurrent_dispatch=config.queue.max_concurrent_dispatch,
)

# Set up component references
central_queue.set_scheduling_strategy(scheduling_strategy)
central_queue.set_task_dispatcher(task_dispatcher)
task_dispatcher.set_central_queue(central_queue)

# Task clearing state - prevents new task submissions during clear operation
_clearing_in_progress = False
_clearing_lock = asyncio.Lock()

# Initialize throughput tracker and planner reporter (if configured)
throughput_tracker = None
planner_reporter = None
if config.planner_report.url and config.planner_report.interval > 0:
    # Create throughput tracker for planner reporting
    throughput_tracker = ThroughputTracker(
        window_size=config.planner_report.throughput_window_size
    )
    logger.info(
        f"Throughput tracker initialized with window_size={config.planner_report.throughput_window_size}"
    )

    # Wire throughput tracker to task dispatcher
    task_dispatcher.throughput_tracker = throughput_tracker

    # Create planner reporter with throughput tracker
    planner_reporter = PlannerReporter(
        task_registry=task_registry,
        planner_url=config.planner_report.url,
        interval=config.planner_report.interval,
        timeout=config.planner_report.timeout,
        throughput_tracker=throughput_tracker,
    )
    logger.info(
        f"Planner reporter initialized: URL={config.planner_report.url}, "
        f"interval={config.planner_report.interval}s"
    )
else:
    logger.debug(
        "Planner reporter disabled (PLANNER_URL not set or SCHEDULER_AUTO_REPORT=0)"
    )

# ============================================================================
# Task Redistribution Helper
# ============================================================================

async def _subtract_task_from_queue_info(
    instance_id: str,
    predicted_time_ms: float,
    predicted_error_margin_ms: Optional[float] = None,
    predicted_quantiles: Optional[dict] = None,
) -> None:
    """
    Subtract a task's predicted execution time from an instance's queue info.

    This should be called when a task is removed from an instance's queue
    (e.g., during work stealing when a task is fetched from a donor).

    Args:
        instance_id: ID of the instance to update
        predicted_time_ms: Predicted execution time of the task
        predicted_error_margin_ms: Predicted error margin (for expect_error strategy)
        predicted_quantiles: Predicted quantiles (for probabilistic strategy)
    """
    import math

    current_queue = await instance_registry.get_queue_info(instance_id)
    if not current_queue:
        logger.warning(f"No queue info found for instance {instance_id} during task subtraction")
        return

    if isinstance(current_queue, InstanceQueueExpectError):
        # Subtract expected time, keep error margin unchanged
        new_expected = max(0.0, current_queue.expected_time_ms - predicted_time_ms)
        updated_queue = InstanceQueueExpectError(
            instance_id=instance_id,
            expected_time_ms=new_expected,
            error_margin_ms=current_queue.error_margin_ms,
        )
        await instance_registry.update_queue_info(instance_id, updated_queue)
        logger.debug(
            f"Subtracted task time from queue (expect_error) for {instance_id}: "
            f"expected_time_ms={new_expected:.2f}"
        )

    elif isinstance(current_queue, InstanceQueueProbabilistic):
        if predicted_quantiles:
            # Monte Carlo subtraction
            num_samples = 1000
            random_percentiles = np.random.random(num_samples)

            # Sample from current queue distribution
            queue_samples = np.interp(
                random_percentiles,
                current_queue.quantiles,
                current_queue.values
            )

            # Sample from predicted task distribution
            task_quantiles = sorted(predicted_quantiles.keys())
            task_values = [predicted_quantiles[q] for q in task_quantiles]
            task_samples = np.interp(
                random_percentiles,
                task_quantiles,
                task_values
            )

            # Subtract task time from queue
            updated_samples = np.maximum(queue_samples - task_samples, 0.0)

            # Compute new quantiles from updated samples
            updated_values = [
                float(np.percentile(updated_samples, q * 100))
                for q in current_queue.quantiles
            ]
        else:
            # Fallback: simple subtraction for all quantiles
            updated_values = [
                max(0.0, v - predicted_time_ms)
                for v in current_queue.values
            ]

        updated_queue = InstanceQueueProbabilistic(
            instance_id=instance_id,
            quantiles=current_queue.quantiles,
            values=updated_values,
        )
        await instance_registry.update_queue_info(instance_id, updated_queue)
        logger.debug(
            f"Subtracted task time from queue (probabilistic) for {instance_id}: "
            f"values={[f'{v:.2f}' for v in updated_values]}"
        )


async def _add_task_to_queue_info(
    instance_id: str,
    predicted_time_ms: float,
    predicted_error_margin_ms: Optional[float] = None,
    predicted_quantiles: Optional[dict] = None,
) -> None:
    """
    Add a task's predicted execution time to an instance's queue info.

    This should be called when a task is added to an instance's queue
    (e.g., during work stealing when a task is submitted to a new instance).

    Args:
        instance_id: ID of the instance to update
        predicted_time_ms: Predicted execution time of the task
        predicted_error_margin_ms: Predicted error margin (for expect_error strategy)
        predicted_quantiles: Predicted quantiles (for probabilistic strategy)
    """
    import math

    current_queue = await instance_registry.get_queue_info(instance_id)
    if not current_queue:
        logger.warning(f"No queue info found for instance {instance_id} during task addition")
        return

    if isinstance(current_queue, InstanceQueueExpectError):
        # Add expected time
        new_expected = current_queue.expected_time_ms + predicted_time_ms
        # Add error margin using error accumulation formula
        task_error = predicted_error_margin_ms or 0.0
        new_error = math.sqrt(current_queue.error_margin_ms ** 2 + task_error ** 2)

        updated_queue = InstanceQueueExpectError(
            instance_id=instance_id,
            expected_time_ms=new_expected,
            error_margin_ms=new_error,
        )
        await instance_registry.update_queue_info(instance_id, updated_queue)
        logger.debug(
            f"Added task time to queue (expect_error) for {instance_id}: "
            f"expected_time_ms={new_expected:.2f}, error_margin_ms={new_error:.2f}"
        )

    elif isinstance(current_queue, InstanceQueueProbabilistic):
        if predicted_quantiles:
            # Monte Carlo addition
            num_samples = 1000
            random_percentiles = np.random.random(num_samples)

            # Sample from current queue distribution
            queue_samples = np.interp(
                random_percentiles,
                current_queue.quantiles,
                current_queue.values
            )

            # Sample from predicted task distribution
            task_quantiles = sorted(predicted_quantiles.keys())
            task_values = [predicted_quantiles[q] for q in task_quantiles]
            task_samples = np.interp(
                random_percentiles,
                task_quantiles,
                task_values
            )

            # Add task time to queue
            updated_samples = queue_samples + task_samples

            # Compute new quantiles from updated samples
            updated_values = [
                float(np.percentile(updated_samples, q * 100))
                for q in current_queue.quantiles
            ]
        else:
            # Fallback: simple addition for all quantiles
            updated_values = [
                v + predicted_time_ms
                for v in current_queue.values
            ]

        updated_queue = InstanceQueueProbabilistic(
            instance_id=instance_id,
            quantiles=current_queue.quantiles,
            values=updated_values,
        )
        await instance_registry.update_queue_info(instance_id, updated_queue)
        logger.debug(
            f"Added task time to queue (probabilistic) for {instance_id}: "
            f"values={[f'{v:.2f}' for v in updated_values]}"
        )


async def _redistribute_tasks_on_registration(
    instance_id: str,
) -> dict:
    """
    Redistribute tasks from other instances to a newly registered instance.

    When a new instance registers, this function "steals" pending tasks from
    other instances with the same model_id to load-balance across the cluster.
    After redistribution completes, the instance status is set to ACTIVE.

    This function is designed to run asynchronously via asyncio.create_task()
    so that registration returns immediately while work stealing proceeds in
    the background.

    Flow:
    1. Find instances with same model_id and pending_tasks > 0
    2. Randomly select up to half of the donor candidates
    3. From each selected instance, call GET /task/fetch to get 1 task
    4. Resubmit each fetched task to the new instance
    5. Update instance status to ACTIVE when done

    Args:
        instance_id: ID of the newly registered instance

    Returns:
        Dictionary with redistribution statistics:
        - donors_selected: Number of donor instances selected
        - tasks_fetched: Number of tasks successfully fetched
        - tasks_resubmitted: Number of tasks successfully resubmitted
        - failed_donors: List of donor instance IDs that failed
        - failed_resubmissions: List of task IDs that failed to resubmit
    """
    # Get the new instance from registry
    new_instance = await instance_registry.get(instance_id)
    if not new_instance:
        logger.error(f"[WorkStealing] Instance {instance_id} not found in registry")
        return {
            "donors_selected": 0,
            "tasks_fetched": 0,
            "tasks_resubmitted": 0,
            "failed_donors": [],
            "failed_resubmissions": [],
        }

    logger.info(
        f"[WorkStealing] Started for new instance {instance_id} (model={new_instance.model_id})"
    )

    try:
        # Step 1: Find donor candidates - all active instances with same model
        # We query instance directly via /task/fetch, not relying on scheduler's pending_tasks counter
        all_instances = await instance_registry.list_active(model_id=new_instance.model_id)
        donor_candidates = [
            inst for inst in all_instances
            if inst.instance_id != new_instance.instance_id
        ]

        if not donor_candidates:
            logger.info(
                f"[WorkStealing] No donor candidates for instance {instance_id} "
                f"(model={new_instance.model_id}) - no other active instances"
            )
            return {
                "donors_selected": 0,
                "tasks_fetched": 0,
                "tasks_resubmitted": 0,
                "failed_donors": [],
                "failed_resubmissions": [],
            }

        logger.info(
            f"[WorkStealing] Found {len(donor_candidates)} potential donors for "
            f"instance {instance_id}: [{', '.join(d.instance_id for d in donor_candidates)}]"
        )

        # Step 2: Calculate how many donors to select based on total active tasks in scheduler
        # Active tasks = PENDING + RUNNING (not COMPLETED or FAILED)
        pending_count = await task_registry.get_count_by_status(TaskStatus.PENDING)
        running_count = await task_registry.get_count_by_status(TaskStatus.RUNNING)
        completed_count = await task_registry.get_count_by_status(TaskStatus.COMPLETED)
        failed_count = await task_registry.get_count_by_status(TaskStatus.FAILED)
        total_tasks = await task_registry.get_total_count()
        total_active_tasks = pending_count + running_count

        logger.debug(
            f"[WorkStealing] Task counts for {instance_id}: "
            f"total={total_tasks}, pending={pending_count}, running={running_count}, "
            f"completed={completed_count}, failed={failed_count}, active={total_active_tasks}"
        )

        # Skip work-stealing if there are no active tasks
        if total_active_tasks == 0:
            logger.debug(
                f"[WorkStealing] Skipping for instance {instance_id} - no active tasks in scheduler"
            )
            return {
                "donors_selected": 0,
                "tasks_fetched": 0,
                "tasks_resubmitted": 0,
                "failed_donors": [],
                "failed_resubmissions": [],
            }

        # Include the new instance in the count (it has 0 tasks initially)
        total_instances = len(donor_candidates) + 1
        avg_tasks_per_instance = total_active_tasks / total_instances

        # Number of donors to select: min(ceil(avg), num_donors)
        num_donors_to_select = min(
            math.ceil(avg_tasks_per_instance),
            len(donor_candidates)
        )

        # Randomly select donors
        selected_donors = random.sample(donor_candidates, num_donors_to_select)

        logger.info(
            f"[WorkStealing] Selecting {num_donors_to_select} donors "
            f"(avg_tasks={avg_tasks_per_instance:.1f}, active_tasks={total_active_tasks}, "
            f"pending={pending_count}, running={running_count}): "
            f"[{', '.join(d.instance_id for d in selected_donors)}]"
        )

        # Step 3: Fetch tasks from donors in PARALLEL
        fetched_tasks = []
        failed_donors = []

        async def fetch_from_donor(donor, client):
            """Fetch a task from a single donor instance."""
            try:
                response = await client.get(f"{donor.endpoint}/task/fetch")
                response.raise_for_status()
                data = response.json()

                if data.get("exist"):
                    task_id = data["task_id"]

                    # Get prediction info from task_registry for queue update
                    task_record = await task_registry.get(task_id)
                    predicted_time_ms = None
                    predicted_error_margin_ms = None
                    predicted_quantiles = None
                    if task_record:
                        predicted_time_ms = task_record.predicted_time_ms
                        predicted_error_margin_ms = task_record.predicted_error_margin_ms
                        predicted_quantiles = task_record.predicted_quantiles

                    return {
                        "success": True,
                        "task_data": {
                            "task_id": task_id,
                            "model_id": data.get("model_id"),
                            "task_input": data.get("task_input"),
                            "enqueue_time": data.get("enqueue_time"),
                            "submitted_at": data.get("submitted_at"),
                            "donor_instance_id": donor.instance_id,
                            "predicted_time_ms": predicted_time_ms,
                            "predicted_error_margin_ms": predicted_error_margin_ms,
                            "predicted_quantiles": predicted_quantiles,
                        },
                        "donor": donor,
                    }
                else:
                    logger.debug(f"Donor {donor.instance_id} had no fetchable tasks")
                    return {"success": False, "has_task": False, "donor": donor}

            except httpx.HTTPStatusError as e:
                log_http_error(
                    e,
                    request_url=f"{donor.endpoint}/task/fetch",
                    request_method="GET",
                    context="task redistribution fetch",
                    extra={
                        "donor_id": donor.instance_id,
                        "new_instance_id": new_instance.instance_id
                    },
                )
                return {"success": False, "error": True, "donor": donor}
            except Exception as e:
                logger.warning(
                    f"Failed to fetch task from donor {donor.instance_id}: "
                    f"{type(e).__name__}: {e}"
                )
                return {"success": False, "error": True, "donor": donor}

        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            # Execute all fetch operations in parallel
            results = await asyncio.gather(
                *[fetch_from_donor(donor, client) for donor in selected_donors],
                return_exceptions=True
            )

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Unexpected exception during work stealing: {result}")
                    continue

                if result.get("success"):
                    task_data = result["task_data"]
                    donor = result["donor"]
                    fetched_tasks.append(task_data)
                    logger.info(
                        f"[WorkStealing] Stolen task {task_data['task_id']} from {donor.instance_id} → {instance_id}"
                    )

                    # Update donor's queue info by subtracting the stolen task's time
                    predicted_time_ms = task_data.get("predicted_time_ms")
                    if predicted_time_ms is not None:
                        await _subtract_task_from_queue_info(
                            instance_id=donor.instance_id,
                            predicted_time_ms=predicted_time_ms,
                            predicted_error_margin_ms=task_data.get("predicted_error_margin_ms"),
                            predicted_quantiles=task_data.get("predicted_quantiles"),
                        )
                        logger.debug(
                            f"Updated donor {donor.instance_id} queue_info after task steal "
                            f"(subtracted {predicted_time_ms:.2f}ms)"
                        )
                elif result.get("error"):
                    failed_donors.append(result["donor"].instance_id)

        if not fetched_tasks:
            logger.info(
                f"[WorkStealing] No tasks fetched from {len(selected_donors)} donors "
                f"for instance {instance_id}"
            )
            return {
                "donors_selected": len(selected_donors),
                "tasks_fetched": 0,
                "tasks_resubmitted": 0,
                "failed_donors": failed_donors,
                "failed_resubmissions": [],
            }

        # Step 4: Resubmit tasks to new instance
        resubmitted_count = 0
        failed_resubmissions = []
        # Use scheduler's callback URL for the resubmitted tasks
        redistribution_callback_url = f"{callback_base_url}/callback/task_result"

        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            for task_data in fetched_tasks:
                try:
                    # Prepare submission payload with preserved metadata
                    payload = {
                        "task_id": task_data["task_id"],
                        "model_id": task_data["model_id"] or new_instance.model_id,
                        "task_input": task_data["task_input"] or {},
                        "callback_url": redistribution_callback_url,  # Rewritten to scheduler's callback
                    }

                    # Preserve enqueue_time for priority ordering if available
                    if task_data.get("enqueue_time") is not None:
                        payload["enqueue_time"] = task_data["enqueue_time"]

                    # Submit to new instance
                    response = await client.post(
                        f"{new_instance.endpoint}/task/submit",
                        json=payload
                    )
                    response.raise_for_status()

                    submit_result = response.json()
                    if submit_result.get("success"):
                        # Increment new instance's pending count
                        await instance_registry.increment_pending(new_instance.instance_id)
                        resubmitted_count += 1

                        # Update new instance's queue info by adding the stolen task's time
                        predicted_time_ms = task_data.get("predicted_time_ms")
                        if predicted_time_ms is not None:
                            await _add_task_to_queue_info(
                                instance_id=new_instance.instance_id,
                                predicted_time_ms=predicted_time_ms,
                                predicted_error_margin_ms=task_data.get("predicted_error_margin_ms"),
                                predicted_quantiles=task_data.get("predicted_quantiles"),
                            )
                            logger.debug(
                                f"Updated new instance {new_instance.instance_id} queue_info "
                                f"after task steal (added {predicted_time_ms:.2f}ms)"
                            )

                        # Update task_registry to reflect new assigned instance
                        await task_registry.update_assigned_instance(
                            task_data["task_id"],
                            new_instance.instance_id
                        )

                        logger.debug(
                            f"Resubmitted task {task_data['task_id']} "
                            f"from {task_data['donor_instance_id']} to {new_instance.instance_id}"
                        )
                    else:
                        logger.warning(
                            f"New instance rejected task {task_data['task_id']}: "
                            f"{submit_result.get('message', 'Unknown error')}"
                        )
                        failed_resubmissions.append(task_data["task_id"])

                except httpx.HTTPStatusError as e:
                    log_http_error(
                        e,
                        request_url=f"{new_instance.endpoint}/task/submit",
                        request_method="POST",
                        request_body=payload,
                        context="task redistribution submit",
                        extra={
                            "task_id": task_data["task_id"],
                            "new_instance_id": new_instance.instance_id
                        },
                    )
                    failed_resubmissions.append(task_data["task_id"])
                except Exception as e:
                    logger.error(
                        f"Failed to resubmit task {task_data['task_id']} "
                        f"to {new_instance.instance_id}: {type(e).__name__}: {e}"
                    )
                    failed_resubmissions.append(task_data["task_id"])

        # Log work stealing summary
        stolen_from = {}
        for task_data in fetched_tasks:
            donor_id = task_data["donor_instance_id"]
            stolen_from[donor_id] = stolen_from.get(donor_id, 0) + 1

        summary_parts = [f"{donor}({count})" for donor, count in stolen_from.items()]
        logger.info(
            f"[WorkStealing] Summary for {instance_id}: "
            f"fetched={len(fetched_tasks)}, resubmitted={resubmitted_count}, "
            f"failed={len(failed_resubmissions)}, "
            f"from donors: [{', '.join(summary_parts)}]"
        )

        return {
            "donors_selected": len(selected_donors),
            "tasks_fetched": len(fetched_tasks),
            "tasks_resubmitted": resubmitted_count,
            "failed_donors": failed_donors,
            "failed_resubmissions": failed_resubmissions,
        }

    finally:
        # Step 5: Always update instance status to ACTIVE when redistribution completes
        # (whether successful or not - the instance should become active)
        try:
            await instance_registry.update_status(instance_id, InstanceStatus.ACTIVE)
            logger.info(f"[WorkStealing] Instance {instance_id} status → ACTIVE (work stealing complete)")
        except KeyError:
            logger.warning(f"[WorkStealing] Instance {instance_id} not found when updating status to ACTIVE")


# ============================================================================
# Instance Management Endpoints
# ============================================================================

@app.post("/instance/register", response_model=InstanceRegisterResponse)
async def register_instance(request: InstanceRegisterRequest):
    """
    Register a new instance to the scheduler.

    If the instance is already registered, returns success with the existing
    instance info (duplicate registration is silently ignored).

    Args:
        request: Instance registration details

    Returns:
        InstanceRegisterResponse with registration status and instance info
    """
    # Check if instance already exists - ignore duplicate registration
    existing_instance = await instance_registry.get(request.instance_id)
    if existing_instance:
        logger.warning(
            f"[register_instance] Duplicate registration ignored | "
            f"instance_id={request.instance_id} | model_id={existing_instance.model_id}"
        )
        return InstanceRegisterResponse(
            success=True,
            message="Instance already registered (duplicate registration ignored)",
            instance=existing_instance,
        )

    # Validate platform_info has required keys
    required_keys = {"software_name", "software_version", "hardware_name"}
    if not required_keys.issubset(request.platform_info.keys()):
        missing_keys = required_keys - request.platform_info.keys()
        error_msg = f"platform_info missing required keys: {missing_keys}"
        logger.error(f"[register_instance] {error_msg} | instance_id={request.instance_id}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg,
            },
        )

    # TODO: Optional - validate that the endpoint is reachable

    # Create Instance object with INITIALIZING status
    # The instance starts in INITIALIZING state and will be set to ACTIVE
    # after work stealing completes
    instance = Instance(
        instance_id=request.instance_id,
        model_id=request.model_id,
        endpoint=request.endpoint,
        platform_info=request.platform_info,
        status=InstanceStatus.INITIALIZING,
    )

    # Register instance (this also initializes queue info and stats)
    try:
        await instance_registry.register(instance)
        logger.info(
            f"Registered instance {request.instance_id} for model {request.model_id} "
            f"on {request.platform_info['hardware_name']} (status=INITIALIZING)"
        )

        # Set model_id for planner reporter (only first registration takes effect)
        if planner_reporter:
            planner_reporter.set_model_id(request.model_id)

        # Start async work stealing in the background
        # The redistribution function will set the instance status to ACTIVE when done
        asyncio.create_task(
            _redistribute_tasks_on_registration(
                instance_id=request.instance_id,
            )
        )
        logger.debug(f"Started async work stealing for instance {request.instance_id}")

    except ValueError as e:
        error_msg = str(e)
        logger.error(f"[register_instance] Failed to register instance | instance_id={request.instance_id} | error={error_msg}", exc_info=True)
        raise HTTPException(
            status_code=400, detail={"success": False, "error": error_msg}
        )

    return InstanceRegisterResponse(
        success=True,
        message="Instance registered successfully (work stealing in progress)",
        instance=instance,
    )


@app.post("/instance/remove", response_model=InstanceRemoveResponse)
async def remove_instance(request: InstanceRemoveRequest):
    """
    Safely remove an instance from the scheduler.

    The instance must be in DRAINING state with no pending tasks.
    Use /instance/drain first, then check /instance/drain/status before removing.

    Args:
        request: Instance removal request with instance_id

    Returns:
        InstanceRemoveResponse with removal status

    Raises:
        HTTPException 404: If instance not found
        HTTPException 400: If instance cannot be safely removed
    """
    # Check if instance exists
    if not await instance_registry.get(request.instance_id):
        error_msg = "Instance not found"
        logger.error(f"[remove_instance] {error_msg} | instance_id={request.instance_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    # Use safe_remove which validates draining state and pending tasks
    try:
        await instance_registry.safe_remove(request.instance_id)
        logger.info(f"Safely removed instance {request.instance_id}")
    except KeyError:
        error_msg = "Instance not found"
        logger.error(f"[remove_instance] Attempted to remove non-existent instance | instance_id={request.instance_id}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"[remove_instance] Cannot safely remove instance | instance_id={request.instance_id} | error={error_msg}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg,
                "hint": "Use /instance/drain first and wait for tasks to complete"
            }
        )

    return InstanceRemoveResponse(
        success=True,
        message="Instance removed successfully",
        instance_id=request.instance_id,
    )


@app.post("/instance/drain", response_model=InstanceDrainResponse)
async def drain_instance(request: InstanceDrainRequest):
    """
    Start draining an instance - stop assigning new tasks.

    The instance will continue processing existing tasks but will not receive
    new task assignments. Use /instance/drain/status to check when safe to remove.

    Args:
        request: Instance drain request with instance_id

    Returns:
        InstanceDrainResponse with drain status and pending task count

    Raises:
        HTTPException 404: If instance not found
        HTTPException 400: If instance is not in ACTIVE state
    """
    if not await instance_registry.get(request.instance_id):
        error_msg = "Instance not found"
        logger.error(f"[drain_instance] {error_msg} | instance_id={request.instance_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    try:
        instance = await instance_registry.start_draining(request.instance_id)
        drain_status = await instance_registry.get_drain_status(request.instance_id)

        # Estimate completion time based on queue info
        queue_info = await instance_registry.get_queue_info(request.instance_id)
        estimated_time = None
        if queue_info:
            if isinstance(queue_info, InstanceQueueExpectError):
                estimated_time = queue_info.expected_time_ms
            elif isinstance(queue_info, InstanceQueueProbabilistic):
                # Use median (0.5 quantile) if available
                if queue_info.quantiles and queue_info.values:
                    try:
                        median_idx = queue_info.quantiles.index(0.5)
                        estimated_time = queue_info.values[median_idx]
                    except (ValueError, IndexError):
                        # Fall back to first value if 0.5 quantile not available
                        estimated_time = queue_info.values[0] if queue_info.values else None

        logger.info(
            f"Started draining instance {request.instance_id} "
            f"with {drain_status['pending_tasks']} pending tasks"
        )

        return InstanceDrainResponse(
            success=True,
            message="Instance is now draining. No new tasks will be assigned.",
            instance_id=instance.instance_id,
            status=instance.status,
            pending_tasks=drain_status["pending_tasks"],
            running_tasks=0,
            estimated_completion_time_ms=estimated_time,
        )

    except ValueError as e:
        error_msg = str(e)
        logger.error(f"[drain_instance] Failed to drain instance | instance_id={request.instance_id} | error={error_msg}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": error_msg}
        )


@app.get("/instance/drain/status", response_model=InstanceDrainStatusResponse)
async def get_drain_status(instance_id: str = Query(..., description="ID of the instance to check")):
    """
    Check draining status of an instance.

    Returns whether the instance is ready for safe removal (all tasks completed).

    Args:
        instance_id: ID of the instance to check

    Returns:
        InstanceDrainStatusResponse with current status and whether safe to remove

    Raises:
        HTTPException 404: If instance not found
    """
    try:
        drain_status = await instance_registry.get_drain_status(instance_id)

        logger.debug(
            f"Drain status for {instance_id}: "
            f"status={drain_status['status']}, "
            f"pending={drain_status['pending_tasks']}, "
            f"can_remove={drain_status['can_remove']}"
        )

        return InstanceDrainStatusResponse(
            success=True,
            instance_id=instance_id,
            status=drain_status["status"],
            pending_tasks=drain_status["pending_tasks"],
            running_tasks=0,
            can_remove=drain_status["can_remove"],
            drain_initiated_at=drain_status.get("drain_initiated_at"),
        )

    except KeyError:
        error_msg = "Instance not found"
        logger.error(f"[get_drain_status] Attempted to check drain status for non-existent instance | instance_id={instance_id}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg}
        )


@app.post("/instance/redeploy/start", response_model=InstanceRedeployResponse)
async def start_instance_redeploy(request: InstanceRedeployRequest):
    """
    Start redeploying an instance.

    This initiates a graceful redeployment of an instance:
    1. Mark instance as REDEPLOYING (stops accepting new tasks)
    2. Request instance to extract pending tasks
    3. Redistribute extracted tasks to other active instances
    4. Return current running task info (if any)

    Args:
        request: Instance redeploy request with instance_id and optional parameters

    Returns:
        InstanceRedeployResponse with redistribution statistics and current task info

    Raises:
        HTTPException 404: If instance not found
        HTTPException 400: If instance is not in ACTIVE state
        HTTPException 500: If communication with instance fails
    """
    # Step 1: Validate instance exists and is in ACTIVE state
    instance = await instance_registry.get(request.instance_id)
    if not instance:
        error_msg = "Instance not found"
        logger.error(f"[start_instance_redeploy] {error_msg} | instance_id={request.instance_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    if instance.status != InstanceStatus.ACTIVE:
        error_msg = f"Instance must be in ACTIVE state, current state: {instance.status}"
        logger.error(f"[start_instance_redeploy] {error_msg} | instance_id={request.instance_id}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg
            }
        )

    logger.info(f"Starting redeployment for instance {request.instance_id}")

    # Step 2: Update instance status to REDEPLOYING in registry
    try:
        await instance_registry.update_status(request.instance_id, InstanceStatus.REDEPLOYING)
    except Exception as e:
        error_msg = f"Failed to update instance status: {str(e)}"
        logger.error(f"[start_instance_redeploy] {error_msg} | instance_id={request.instance_id}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": error_msg}
        )

    # Step 3: Communicate with instance to start redeployment and extract tasks
    returned_tasks = []
    current_task = None
    estimated_redeploy_time_ms = None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{instance.endpoint}/redeploy/start",
                json={
                    "reason": request.redeploy_reason or "Scheduler-initiated redeployment",
                    "target_model_id": request.target_model_id,
                }
            )
            response.raise_for_status()

            redeploy_data = response.json()
            returned_tasks = redeploy_data.get("returned_tasks", [])
            current_task = redeploy_data.get("current_task")
            estimated_redeploy_time_ms = redeploy_data.get("estimated_completion_ms")

            logger.info(
                f"Instance {request.instance_id} returned {len(returned_tasks)} pending tasks, "
                f"current_task: {current_task is not None}"
            )

    except httpx.HTTPStatusError as e:
        log_http_error(
            e,
            request_body={
                "reason": request.redeploy_reason or "Scheduler-initiated redeployment",
                "target_model_id": request.target_model_id,
            },
            context="instance redeploy start",
            extra={"instance_id": request.instance_id},
        )
        error_msg = f"Instance redeploy request failed: {e.response.text}"
        logger.error(f"[start_instance_redeploy] Instance redeploy request failed with status {e.response.status_code} | instance_id={request.instance_id} | error={error_msg}", exc_info=True)
        # Revert instance status
        await instance_registry.update_status(request.instance_id, InstanceStatus.ACTIVE)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": error_msg
            }
        )
    except Exception as e:
        log_http_error(
            e,
            request_url=f"{instance.endpoint}/redeploy/start",
            request_method="POST",
            request_body={
                "reason": request.redeploy_reason or "Scheduler-initiated redeployment",
                "target_model_id": request.target_model_id,
            },
            context="instance redeploy communication error",
            extra={"instance_id": request.instance_id},
        )
        error_msg = f"Failed to communicate with instance: {str(e)}"
        logger.error(f"[start_instance_redeploy] Failed to communicate with instance | instance_id={request.instance_id} | error={error_msg}", exc_info=True)
        # Revert instance status
        await instance_registry.update_status(request.instance_id, InstanceStatus.ACTIVE)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": error_msg
            }
        )

    # Step 4: Redistribute returned tasks to other instances
    redistributed_tasks = []
    failed_redistributions = []

    if returned_tasks:
        logger.info(f"Redistributing {len(returned_tasks)} tasks from {request.instance_id}")

        for task_data in returned_tasks:
            try:
                task_id = task_data["task_id"]
                model_id = task_data["model_id"]
                task_input = task_data["task_input"]
                enqueue_time = task_data.get("enqueue_time")
                metadata = task_data.get("metadata", {})

                # Use background scheduler to reassign task
                # The scheduler will use the current strategy to select the best instance
                success = await background_scheduler.reassign_task(
                    task_id=task_id,
                    model_id=model_id,
                    task_input=task_input,
                    enqueue_time=enqueue_time,
                    metadata=metadata,
                    exclude_instance_id=request.instance_id,  # Don't assign back to redeploying instance
                )

                if success:
                    redistributed_tasks.append(task_id)
                    logger.debug(f"Successfully redistributed task {task_id}")
                else:
                    failed_redistributions.append(task_id)
                    logger.warning(f"Failed to redistribute task {task_id}")

            except Exception as e:
                task_id = task_data.get("task_id", "unknown")
                failed_redistributions.append(task_id)
                logger.error(f"Error redistributing task {task_id}: {e}")

    # Step 5: Construct response
    response_message = f"Instance {request.instance_id} is now redeploying. "
    if returned_tasks:
        response_message += (
            f"Returned {len(returned_tasks)} pending tasks. "
            f"Successfully redistributed {len(redistributed_tasks)}, "
            f"failed {len(failed_redistributions)}."
        )
    else:
        response_message += "No pending tasks to redistribute."

    if current_task:
        response_message += f" Current task {current_task.get('task_id')} is still running."

    logger.info(response_message)

    return InstanceRedeployResponse(
        success=True,
        message=response_message,
        returned_tasks=returned_tasks,
        redistributed_tasks=redistributed_tasks,
        failed_redistributions=failed_redistributions,
        current_task=current_task,
        estimated_redeploy_time_ms=estimated_redeploy_time_ms,
    )


@app.post("/instance/redeploy/complete", response_model=InstanceRegisterResponse)
async def complete_instance_redeploy(request: InstanceRegisterRequest):
    """
    Complete instance redeployment and return to ACTIVE status.

    This endpoint is called after an instance finishes redeployment to:
    1. Update instance configuration (model_id, endpoint, platform_info)
    2. Transition instance status from REDEPLOYING to ACTIVE
    3. Reinitialize queue info and stats if needed

    Args:
        request: Instance registration details (may have updated values)

    Returns:
        InstanceRegisterResponse with updated instance info

    Raises:
        HTTPException 404: If instance not found
        HTTPException 400: If instance is not in REDEPLOYING state
    """
    # Check if instance exists
    existing_instance = await instance_registry.get(request.instance_id)
    if not existing_instance:
        error_msg = "Instance not found"
        logger.error(f"[complete_instance_redeploy] {error_msg} | instance_id={request.instance_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    # Verify instance is in REDEPLOYING state
    if existing_instance.status != InstanceStatus.REDEPLOYING:
        error_msg = f"Instance must be in REDEPLOYING state, current state: {existing_instance.status}"
        logger.error(f"[complete_instance_redeploy] {error_msg} | instance_id={request.instance_id}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg
            }
        )

    # Validate platform_info has required keys
    required_keys = {"software_name", "software_version", "hardware_name"}
    if not required_keys.issubset(request.platform_info.keys()):
        missing_keys = required_keys - request.platform_info.keys()
        error_msg = f"platform_info missing required keys: {missing_keys}"
        logger.error(f"[complete_instance_redeploy] {error_msg} | instance_id={request.instance_id}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg,
            },
        )

    # Update instance configuration
    existing_instance.model_id = request.model_id
    existing_instance.endpoint = request.endpoint
    existing_instance.platform_info = request.platform_info

    # Transition to ACTIVE status
    try:
        await instance_registry.update_status(request.instance_id, InstanceStatus.ACTIVE)

        # Reset stats if model changed (new model = fresh start)
        # Queue info will be updated as new tasks are scheduled
        stats = await instance_registry.get_stats(request.instance_id)
        if stats and stats.pending_tasks > 0:
            # Shouldn't have pending tasks after redeployment, but log if we do
            logger.warning(
                f"Instance {request.instance_id} has {stats.pending_tasks} pending tasks "
                f"after redeployment completion"
            )

        logger.info(
            f"Completed redeployment for instance {request.instance_id}. "
            f"New model: {request.model_id}, Status: ACTIVE"
        )

    except Exception as e:
        error_msg = f"Failed to complete redeployment: {str(e)}"
        logger.error(f"[complete_instance_redeploy] Failed to complete redeployment | instance_id={request.instance_id} | error={error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": error_msg}
        )

    return InstanceRegisterResponse(
        success=True,
        message="Instance redeployment completed successfully. Instance is now ACTIVE.",
        instance=existing_instance,
    )


@app.get("/instance/list", response_model=InstanceListResponse)
async def list_instances(model_id: Optional[str] = Query(None)):
    """
    List all registered instances with optional filtering.

    Args:
        model_id: Optional filter by model ID

    Returns:
        InstanceListResponse with list of instances
    """
    # Retrieve instances (with optional model_id filter)
    instances = await instance_registry.list_all(model_id=model_id)

    return InstanceListResponse(
        success=True,
        count=len(instances),
        instances=instances,
    )


@app.get("/instance/info", response_model=InstanceInfoResponse)
async def get_instance_info(instance_id: str = Query(...)):
    """
    Get detailed information about a specific instance.

    Args:
        instance_id: ID of the instance to query

    Returns:
        InstanceInfoResponse with instance details, queue info, and stats

    Raises:
        HTTPException 404: If instance not found
    """
    # Get instance
    instance = await instance_registry.get(instance_id)
    if not instance:
        error_msg = "Instance not found"
        logger.error(f"[get_instance_info] {error_msg} | instance_id={instance_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    # Get queue information
    queue_info = await instance_registry.get_queue_info(instance_id)
    if not queue_info:
        # Shouldn't happen, but handle gracefully
        error_msg = "Queue info not found"
        logger.error(f"[get_instance_info] {error_msg} | instance_id={instance_id}")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": error_msg},
        )

    # Get statistics
    stats = await instance_registry.get_stats(instance_id)
    if not stats:
        # Shouldn't happen, but handle gracefully
        error_msg = "Statistics not found"
        logger.error(f"[get_instance_info] {error_msg} | instance_id={instance_id}")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": error_msg},
        )

    return InstanceInfoResponse(
        success=True,
        instance=instance,
        queue_info=queue_info,
        stats=stats,
    )


# ============================================================================
# Task Management Endpoints
# ============================================================================

@app.post("/task/submit", response_model=TaskSubmitResponse)
async def submit_task(request: TaskSubmitRequest):
    """
    Submit a new task for execution.

    This endpoint returns immediately after creating the task record.
    Task scheduling (predictions, instance selection, dispatching) happens
    in the background to prevent blocking API responses.

    If no instance is currently available for the model_id, the task will
    be queued and wait for instances to be registered/allocated by the planner.

    Args:
        request: Task submission details

    Returns:
        TaskSubmitResponse with task in PENDING status

    Raises:
        HTTPException 400: If task with this ID already exists
        HTTPException 503: If task clear operation is in progress
    """
    # 0. Check if clear operation is in progress
    if _clearing_in_progress:
        error_msg = "Task clear operation in progress. Please retry."
        logger.error(f"[submit_task] {error_msg} | task_id={request.task_id}")
        raise HTTPException(
            status_code=503,
            detail={"success": False, "error": error_msg},
        )

    # 1. Validate that task doesn't already exist
    if await task_registry.get(request.task_id):
        error_msg = "Task with this ID already exists"
        logger.error(f"[submit_task] {error_msg} | task_id={request.task_id}")
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": error_msg},
        )

    # 2. Log instance availability (but don't reject if no instances)
    # Tasks will queue and wait for instances to be registered/available
    available_instances = await instance_registry.list_active(model_id=request.model_id)
    if not available_instances:
        logger.warning(
            f"[submit_task] No available instance for model_id: {request.model_id}, "
            f"task {request.task_id} will be queued and wait for instance allocation"
        )

    # 3. Create task record immediately with PENDING status
    # (No assigned_instance yet - will be set by background scheduler)
    try:
        task_record = await task_registry.create_task(
            task_id=request.task_id,
            model_id=request.model_id,
            task_input=request.task_input,
            metadata=request.metadata,
            assigned_instance="",  # Empty until background scheduling completes
            predicted_time_ms=None,
            predicted_error_margin_ms=None,
            predicted_quantiles=None,
        )
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"[submit_task] Failed to create task | task_id={request.task_id} | error={error_msg}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": error_msg}
        )

    # 4. Enqueue task to central queue (non-blocking)
    # Central queue will:
    # - Queue task for dispatch
    # - When instance capacity available:
    #   - Get predictions from predictor service
    #   - Select optimal instance
    #   - Update queue info (Monte Carlo sampling)
    #   - Assign instance to task
    #   - Dispatch task to instance
    queue_position = await central_queue.enqueue(
        task_id=request.task_id,
        model_id=request.model_id,
        task_input=request.task_input,
        metadata=request.metadata,
    )

    # 5. Return immediately with PENDING status
    task_info = TaskInfo(
        task_id=task_record.task_id,
        status=task_record.status,  # PENDING
        assigned_instance=task_record.assigned_instance,  # Empty string initially
        submitted_at=task_record.submitted_at,
    )

    return TaskSubmitResponse(
        success=True,
        message=f"Task queued at position {queue_position} and is being scheduled",
        task=task_info,
    )


@app.get("/task/list", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None),
    model_id: Optional[str] = Query(None),
    instance_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    List tasks with optional filtering and pagination.

    Args:
        status: Optional filter by task status
        model_id: Optional filter by model ID
        instance_id: Optional filter by assigned instance
        limit: Maximum number of tasks to return (1-1000)
        offset: Pagination offset

    Returns:
        TaskListResponse with paginated task list
    """
    # Retrieve tasks with filters and pagination
    tasks, total = await task_registry.list_all(
        status=status,
        model_id=model_id,
        instance_id=instance_id,
        limit=limit,
        offset=offset,
    )

    # Convert to TaskSummary objects
    task_summaries = [
        TaskSummary(
            task_id=task.task_id,
            model_id=task.model_id,
            status=task.status,
            assigned_instance=task.assigned_instance,
            submitted_at=task.submitted_at,
            completed_at=task.completed_at,
        )
        for task in tasks
    ]

    return TaskListResponse(
        success=True,
        count=len(task_summaries),
        total=total,
        offset=offset,
        limit=limit,
        tasks=task_summaries,
    )


@app.get("/task/info", response_model=TaskDetailResponse)
async def get_task_info(task_id: str = Query(...)):
    """
    Get detailed information about a specific task.

    Args:
        task_id: ID of the task to query

    Returns:
        TaskDetailResponse with complete task details

    Raises:
        HTTPException 404: If task not found
    """
    # Get task
    task = await task_registry.get(task_id)
    if not task:
        error_msg = "Task not found"
        logger.error(f"[get_task_info] {error_msg} | task_id={task_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    # Create detailed task info
    task_detail = TaskDetailInfo(
        task_id=task.task_id,
        model_id=task.model_id,
        status=task.status,
        assigned_instance=task.assigned_instance,
        task_input=task.task_input,
        metadata=task.metadata,
        result=task.result,
        error=task.error,
        timestamps=task.get_timestamps(),
        execution_time_ms=task.execution_time_ms,
    )

    return TaskDetailResponse(
        success=True,
        task=task_detail,
    )


@app.post("/task/clear", response_model=TaskClearResponse)
async def clear_tasks():
    """
    Clear all tasks from the scheduler and all registered instances.

    This endpoint:
    1. Sets clearing flag to prevent new task submissions
    2. Clears all tasks from central queue
    3. Clears all task records from the scheduler's registry
    4. Calls /task/clear on all registered instances to clear their task queues
    5. Resets the pending_tasks counter for all instances
    6. Clears the clearing flag

    Use with caution as this operation cannot be undone.

    Returns:
        TaskClearResponse with count of cleared tasks from scheduler
    """
    global _clearing_in_progress

    # Set clearing flag to prevent new task submissions
    async with _clearing_lock:
        if _clearing_in_progress:
            error_msg = "Clear operation already in progress"
            logger.error(f"[clear_tasks] {error_msg}")
            raise HTTPException(
                status_code=409,
                detail={"success": False, "error": error_msg},
            )
        _clearing_in_progress = True

    try:
        logger.warning("Starting task clear operation")

        # 1. Clear central queue first to prevent new dispatches
        queue_cleared = await central_queue.clear()
        logger.warning(f"Cleared {queue_cleared} tasks from central queue")

        # 2. Clear all tasks from scheduler registry
        cleared_count = await task_registry.clear_all()
        logger.warning(f"Cleared {cleared_count} tasks from scheduler registry")

        # 3. Get all registered instances
        all_instances = await instance_registry.list_all()

        # Clear tasks from each instance in parallel
        async def clear_instance_tasks(client: httpx.AsyncClient, instance):
            """Helper function to clear tasks from a single instance."""
            try:
                # Call instance's /task/clear endpoint
                response = await client.post(f"{instance.endpoint}/task/clear")
                response.raise_for_status()
                result = response.json()

                logger.info(
                    f"Cleared {result.get('cleared_count', {}).get('total', 0)} tasks "
                    f"from instance {instance.instance_id}"
                )
                return {
                    "instance_id": instance.instance_id,
                    "success": True,
                    "cleared": result.get("cleared_count", {}).get("total", 0)
                }
            except Exception as e:
                log_http_error(
                    e,
                    request_url=f"{instance.endpoint}/task/clear",
                    request_method="POST",
                    context="task clear on instance",
                    extra={"instance_id": instance.instance_id},
                )
                logger.warning(
                    f"Failed to clear tasks from instance {instance.instance_id}: {e}"
                )
                return {
                    "instance_id": instance.instance_id,
                    "success": False,
                    "error": str(e)
                }

        instance_clear_results = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Execute all clear requests in parallel
            instance_clear_results = await asyncio.gather(
                *[clear_instance_tasks(client, instance) for instance in all_instances],
                return_exceptions=False
            )

        # 4. Reset pending_tasks counter for all instances to maintain consistency
        reset_count = await instance_registry.reset_all_pending_tasks()
        logger.info(f"Reset pending_tasks counter for {reset_count} instance(s)")

        # Log summary
        successful_clears = sum(1 for r in instance_clear_results if r["success"])
        total_instance_tasks = sum(
            r.get("cleared", 0) for r in instance_clear_results if r["success"]
        )
        logger.warning(
            f"Task clear operation complete: scheduler={cleared_count}, queue={queue_cleared}, "
            f"instances={successful_clears}/{len(all_instances)} ({total_instance_tasks} tasks)"
        )

        return TaskClearResponse(
            success=True,
            message=f"Successfully cleared {cleared_count} scheduler task(s), "
                    f"{queue_cleared} queued task(s), and tasks from "
                    f"{successful_clears}/{len(all_instances)} instance(s)",
            cleared_count=cleared_count,
        )
    finally:
        # Clear the clearing flag to allow new task submissions
        async with _clearing_lock:
            _clearing_in_progress = False
        logger.info("Task clear operation finished, accepting new submissions")


@app.post("/task/resubmit", response_model=TaskResubmitResponse)
async def resubmit_task(request: TaskResubmitRequest):
    """
    Resubmit a task for rescheduling during instance migration.

    This endpoint is called by instances during migration to return a task
    to the scheduler for reassignment to another instance.

    The task will be:
    1. Reset to PENDING status with cleared result/error
    2. Removed from the original instance's pending count
    3. Rescheduled to a new instance in the background

    Args:
        request: Task resubmit request with task_id and original_instance_id

    Returns:
        TaskResubmitResponse with success status

    Raises:
        HTTPException 404: If task not found
        HTTPException 400: If task is in invalid state (COMPLETED or FAILED)
        HTTPException 404: If original instance not found
    """
    # 1. Validate task exists
    task = await task_registry.get(request.task_id)
    if not task:
        error_msg = "Task not found"
        logger.error(f"[resubmit_task] {error_msg} | task_id={request.task_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    # 2. Record previous state for logging
    previous_status = task.status
    previous_instance = task.assigned_instance

    # 3. Validate task status - can only resubmit PENDING or RUNNING tasks
    if previous_status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        error_msg = f"Cannot resubmit task in {previous_status} state. Only PENDING or RUNNING tasks can be resubmitted."
        logger.error(f"[resubmit_task] {error_msg} | task_id={request.task_id} | status={previous_status}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg,
            },
        )

    # 4. Validate original instance exists (for statistics update)
    original_instance = await instance_registry.get(request.original_instance_id)
    if not original_instance:
        error_msg = "Original instance not found"
        logger.error(f"[resubmit_task] {error_msg} | task_id={request.task_id} | original_instance_id={request.original_instance_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    # 5. Decrement pending tasks count on original instance
    try:
        await instance_registry.decrement_pending(request.original_instance_id)
        logger.debug(f"Decremented pending count for instance {request.original_instance_id}")
    except Exception as e:
        logger.warning(f"Failed to decrement pending count for {request.original_instance_id}: {e}")
        # Continue anyway - the task should still be resubmitted

    # 6. Reset task for resubmission
    try:
        await task_registry.reset_for_resubmit(request.task_id)
        logger.debug(f"Reset task {request.task_id} for resubmission")
    except KeyError:
        error_msg = "Task not found"
        logger.error(f"[resubmit_task] Task not found during reset | task_id={request.task_id}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    # 7. Resubmit task to central queue with original submission time
    # Parse ISO format string to datetime, then convert to timestamp for queue ordering
    enqueue_time = None
    if task.submitted_at:
        from datetime import datetime
        dt = datetime.fromisoformat(task.submitted_at.replace("Z", "+00:00"))
        enqueue_time = dt.timestamp()

    queue_position = await central_queue.enqueue(
        task_id=request.task_id,
        model_id=task.model_id,
        task_input=task.task_input,
        metadata=task.metadata or {},
        enqueue_time=enqueue_time,
    )

    logger.info(
        f"Successfully resubmitted task {request.task_id} to queue at position {queue_position} "
        f"(previous status: {previous_status}, previous instance: {previous_instance}, "
        f"original submission time preserved)"
    )

    return TaskResubmitResponse(
        success=True,
        message=f"Task {request.task_id} resubmitted successfully for rescheduling",
    )


@app.post("/task/update_metadata", response_model=TaskUpdateMetadataResponse)
async def update_task_metadata(request: TaskUpdateMetadataRequest):
    """
    Update metadata for multiple tasks and re-predict if tasks are in queue.

    For tasks that are currently assigned to an instance (status PENDING or RUNNING):
    1. Save old metadata and prediction
    2. Remove old predicted value from instance queue
    3. Update metadata in registry
    4. Re-predict with new metadata
    5. If prediction fails: rollback metadata, restore queue
    6. If prediction succeeds: add new prediction to queue, update task prediction fields

    COMPLETED and FAILED tasks are silently skipped (no error).

    Args:
        request: Batch of task metadata updates

    Returns:
        TaskUpdateMetadataResponse with results for each task
    """
    results = []
    succeeded = 0
    failed = 0
    skipped = 0

    for update in request.updates:
        try:
            result = await _update_single_task_metadata(
                task_id=update.task_id,
                new_metadata=update.metadata,
            )
            results.append(result)

            if result.success:
                if result.message.startswith("Skipped"):
                    skipped += 1
                else:
                    succeeded += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Error updating metadata for task {update.task_id}: {e}")
            results.append(TaskUpdateMetadataResult(
                task_id=update.task_id,
                success=False,
                message=f"Internal error: {str(e)}",
                queue_updated=False,
            ))
            failed += 1

    return TaskUpdateMetadataResponse(
        success=failed == 0,
        message=f"Updated {succeeded}/{len(request.updates)} tasks ({skipped} skipped, {failed} failed)",
        total=len(request.updates),
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        results=results,
    )


async def _update_single_task_metadata(
    task_id: str,
    new_metadata: dict,
) -> TaskUpdateMetadataResult:
    """
    Update metadata for a single task and handle queue updates if needed.

    Implements rollback on prediction failure.

    Args:
        task_id: Task ID to update
        new_metadata: New metadata dictionary

    Returns:
        TaskUpdateMetadataResult with operation status
    """
    # 1. Get the task
    task = await task_registry.get(task_id)
    if not task:
        return TaskUpdateMetadataResult(
            task_id=task_id,
            success=False,
            message="Task not found",
            queue_updated=False,
        )

    # 2. Skip COMPLETED and FAILED tasks (silently)
    if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        return TaskUpdateMetadataResult(
            task_id=task_id,
            success=True,
            message=f"Skipped: task is in {task.status.value} state",
            queue_updated=False,
        )

    # Store old values for potential rollback
    old_metadata = task.metadata.copy() if task.metadata else {}
    old_prediction_ms = task.predicted_time_ms
    old_error_margin_ms = task.predicted_error_margin_ms
    old_quantiles = task.predicted_quantiles.copy() if task.predicted_quantiles else None

    # 3. Check if task is in a queue (needs queue update)
    needs_queue_update = (
        task.assigned_instance != "" and
        task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
    )

    # 4. If no queue update needed, just update metadata
    if not needs_queue_update:
        try:
            await task_registry.update_metadata(task_id, new_metadata)
            return TaskUpdateMetadataResult(
                task_id=task_id,
                success=True,
                message="Metadata updated (task not in queue)",
                queue_updated=False,
                old_prediction_ms=old_prediction_ms,
            )
        except KeyError:
            return TaskUpdateMetadataResult(
                task_id=task_id,
                success=False,
                message="Task not found during update",
                queue_updated=False,
            )

    # 5. Handle queue update with rollback support
    try:
        # 5a. Get the instance for prediction
        instance = await instance_registry.get(task.assigned_instance)
        if not instance:
            logger.warning(
                f"Instance {task.assigned_instance} not found for task {task_id}, "
                "skipping queue update"
            )
            # Just update metadata without queue update
            await task_registry.update_metadata(task_id, new_metadata)
            return TaskUpdateMetadataResult(
                task_id=task_id,
                success=True,
                message="Metadata updated (instance not found, queue update skipped)",
                queue_updated=False,
                old_prediction_ms=old_prediction_ms,
            )

        # 5b. Subtract old prediction from queue
        if old_prediction_ms is not None:
            await _subtract_task_from_queue_info(
                instance_id=task.assigned_instance,
                predicted_time_ms=old_prediction_ms,
                predicted_error_margin_ms=old_error_margin_ms,
                predicted_quantiles=old_quantiles,
            )

        # 5c. Update metadata in registry
        await task_registry.update_metadata(task_id, new_metadata)

        # 5d. Get new prediction with updated metadata
        try:
            # Determine prediction type based on current strategy
            strategy_name = scheduling_strategy.__class__.__name__
            if strategy_name == "MinimumExpectedTimeStrategy":
                prediction_type = "expect_error"
                quantiles = None
            else:
                prediction_type = "quantile"
                quantiles = getattr(scheduling_strategy, 'quantiles', [0.5, 0.9, 0.95, 0.99])

            predictions = await predictor_client.predict(
                model_id=task.model_id,
                metadata=new_metadata,
                instances=[instance],
                prediction_type=prediction_type,
                quantiles=quantiles,
            )

            if not predictions:
                raise ValueError("No predictions returned from predictor")

            new_pred = predictions[0]
            new_prediction_ms = new_pred.predicted_time_ms

        except Exception as pred_error:
            # Prediction failed - rollback
            logger.error(f"Prediction failed for task {task_id}, rolling back: {pred_error}")

            # Rollback metadata
            await task_registry.update_metadata(task_id, old_metadata)

            # Rollback queue (re-add old prediction)
            if old_prediction_ms is not None:
                await _add_task_to_queue_info(
                    instance_id=task.assigned_instance,
                    predicted_time_ms=old_prediction_ms,
                    predicted_error_margin_ms=old_error_margin_ms,
                    predicted_quantiles=old_quantiles,
                )

            return TaskUpdateMetadataResult(
                task_id=task_id,
                success=False,
                message=f"Prediction failed, rolled back: {str(pred_error)}",
                queue_updated=False,
                old_prediction_ms=old_prediction_ms,
            )

        # 5e. Add new prediction to queue
        await _add_task_to_queue_info(
            instance_id=task.assigned_instance,
            predicted_time_ms=new_pred.predicted_time_ms,
            predicted_error_margin_ms=new_pred.error_margin_ms,
            predicted_quantiles=new_pred.quantiles,
        )

        # 5f. Update task's prediction fields
        await task_registry.update_prediction(
            task_id=task_id,
            predicted_time_ms=new_pred.predicted_time_ms,
            predicted_error_margin_ms=new_pred.error_margin_ms,
            predicted_quantiles=new_pred.quantiles,
        )

        logger.info(
            f"Updated metadata and re-predicted task {task_id}: "
            f"{old_prediction_ms:.2f}ms -> {new_prediction_ms:.2f}ms"
            if old_prediction_ms else f"New prediction: {new_prediction_ms:.2f}ms"
        )

        return TaskUpdateMetadataResult(
            task_id=task_id,
            success=True,
            message="Metadata updated with queue recalculation",
            queue_updated=True,
            old_prediction_ms=old_prediction_ms,
            new_prediction_ms=new_prediction_ms,
        )

    except Exception as e:
        # General error - attempt rollback
        logger.error(f"Error updating task {task_id}: {e}")

        try:
            # Attempt to rollback metadata
            await task_registry.update_metadata(task_id, old_metadata)
        except Exception:
            pass  # Best effort rollback

        return TaskUpdateMetadataResult(
            task_id=task_id,
            success=False,
            message=f"Update failed: {str(e)}",
            queue_updated=False,
            old_prediction_ms=old_prediction_ms,
        )


@app.post("/task/repredict", response_model=TaskRepredictResponse)
async def repredict_all_tasks():
    """
    Re-predict all pending/running tasks and update queue information.

    This endpoint triggers re-prediction for all tasks that are currently
    assigned to an instance (status PENDING or RUNNING) WITHOUT modifying
    their metadata.

    Use case: When the predictor model has been updated or retrained,
    use this endpoint to refresh all queue predictions.

    Flow for each eligible task:
    1. Subtract old prediction from instance queue
    2. Re-predict with existing metadata
    3. Add new prediction to instance queue
    4. Update task's prediction fields

    On prediction failure for a task:
    - Queue values are restored
    - Task is counted as failed
    - Processing continues with other tasks

    COMPLETED and FAILED tasks are skipped.
    Tasks not assigned to an instance are skipped.

    Returns:
        TaskRepredictResponse with summary counts
    """
    # Get all tasks from registry
    all_tasks, total_count = await task_registry.list_all()

    repredicted = 0
    failed = 0
    skipped = 0
    eligible_count = 0

    for task in all_tasks:
        try:
            # Skip COMPLETED and FAILED tasks
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                skipped += 1
                continue

            # Skip tasks not in a queue (no assigned instance)
            if not task.assigned_instance:
                skipped += 1
                continue

            # This task is eligible for re-prediction
            eligible_count += 1

            # Re-predict this task (without changing metadata)
            success = await _repredict_single_task(task)

            if success:
                repredicted += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Error re-predicting task {task.task_id}: {e}")
            failed += 1

    return TaskRepredictResponse(
        success=failed == 0,
        message=f"Re-predicted {repredicted}/{eligible_count} eligible tasks ({skipped} skipped, {failed} failed)",
        total_tasks=total_count,
        eligible_tasks=eligible_count,
        repredicted=repredicted,
        failed=failed,
        skipped=skipped,
    )


async def _repredict_single_task(task) -> bool:
    """
    Re-predict a single task without changing metadata.

    Args:
        task: TaskRecord to re-predict

    Returns:
        True if re-prediction succeeded, False otherwise
    """
    task_id = task.task_id

    # Store old prediction values for potential rollback
    old_prediction_ms = task.predicted_time_ms
    old_error_margin_ms = task.predicted_error_margin_ms
    old_quantiles = task.predicted_quantiles.copy() if task.predicted_quantiles else None

    try:
        # 1. Get the instance
        instance = await instance_registry.get(task.assigned_instance)
        if not instance:
            logger.warning(
                f"Instance {task.assigned_instance} not found for task {task_id}, "
                "skipping re-prediction"
            )
            return False

        # 2. Subtract old prediction from queue
        if old_prediction_ms is not None:
            await _subtract_task_from_queue_info(
                instance_id=task.assigned_instance,
                predicted_time_ms=old_prediction_ms,
                predicted_error_margin_ms=old_error_margin_ms,
                predicted_quantiles=old_quantiles,
            )

        # 3. Get new prediction with existing metadata (no change)
        try:
            # Determine prediction type based on current strategy
            strategy_name = scheduling_strategy.__class__.__name__
            if strategy_name == "MinimumExpectedTimeStrategy":
                prediction_type = "expect_error"
                quantiles = None
            else:
                prediction_type = "quantile"
                quantiles = getattr(scheduling_strategy, 'quantiles', [0.5, 0.9, 0.95, 0.99])

            predictions = await predictor_client.predict(
                model_id=task.model_id,
                metadata=task.metadata,  # Use existing metadata (no change)
                instances=[instance],
                prediction_type=prediction_type,
                quantiles=quantiles,
            )

            if not predictions:
                raise ValueError("No predictions returned from predictor")

            new_pred = predictions[0]

        except Exception as pred_error:
            # Prediction failed - rollback queue
            logger.error(f"Re-prediction failed for task {task_id}: {pred_error}")

            # Rollback queue (re-add old prediction)
            if old_prediction_ms is not None:
                await _add_task_to_queue_info(
                    instance_id=task.assigned_instance,
                    predicted_time_ms=old_prediction_ms,
                    predicted_error_margin_ms=old_error_margin_ms,
                    predicted_quantiles=old_quantiles,
                )

            return False

        # 4. Add new prediction to queue
        await _add_task_to_queue_info(
            instance_id=task.assigned_instance,
            predicted_time_ms=new_pred.predicted_time_ms,
            predicted_error_margin_ms=new_pred.error_margin_ms,
            predicted_quantiles=new_pred.quantiles,
        )

        # 5. Update task's prediction fields
        await task_registry.update_prediction(
            task_id=task_id,
            predicted_time_ms=new_pred.predicted_time_ms,
            predicted_error_margin_ms=new_pred.error_margin_ms,
            predicted_quantiles=new_pred.quantiles,
        )

        logger.debug(
            f"Re-predicted task {task_id}: "
            f"{old_prediction_ms:.2f}ms -> {new_pred.predicted_time_ms:.2f}ms"
            if old_prediction_ms else f"New prediction: {new_pred.predicted_time_ms:.2f}ms"
        )

        return True

    except Exception as e:
        # General error - attempt to restore queue
        logger.error(f"Error re-predicting task {task_id}: {e}")

        try:
            # Attempt to restore queue
            if old_prediction_ms is not None:
                await _add_task_to_queue_info(
                    instance_id=task.assigned_instance,
                    predicted_time_ms=old_prediction_ms,
                    predicted_error_margin_ms=old_error_margin_ms,
                    predicted_quantiles=old_quantiles,
                )
        except Exception:
            pass  # Best effort rollback

        return False


@app.get("/task/schedule_info", response_model=TaskScheduleInfoResponse)
async def get_task_schedule_info(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    instance_id: Optional[str] = Query(None, description="Filter by instance ID"),
    status: Optional[str] = Query(None, description="Filter by task status (pending, running, completed, failed)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of tasks to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """
    Get scheduling information for all tasks.

    Returns which instance each task was scheduled to, along with task status
    and submission time. Supports filtering by model_id, instance_id, and status.

    Args:
        model_id: Optional filter by model ID
        instance_id: Optional filter by instance ID
        status: Optional filter by task status
        limit: Maximum number of tasks to return (default 100, max 1000)
        offset: Pagination offset (default 0)

    Returns:
        TaskScheduleInfoResponse with list of task scheduling information
    """
    # Parse status filter if provided
    status_filter = None
    if status:
        try:
            status_filter = TaskStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={"success": False, "error": f"Invalid status: {status}"}
            )

    # Get tasks from registry with filters
    tasks, total = await task_registry.list_all(
        status=status_filter,
        model_id=model_id,
        instance_id=instance_id,
        limit=limit,
        offset=offset,
    )

    # Convert to schedule info format
    task_infos = [
        TaskScheduleInfo(
            task_id=task.task_id,
            model_id=task.model_id,
            status=task.status,
            assigned_instance=task.assigned_instance,
            submitted_at=task.submitted_at,
        )
        for task in tasks
    ]

    return TaskScheduleInfoResponse(
        success=True,
        count=len(task_infos),
        total=total,
        tasks=task_infos,
    )


@app.post("/callback/task_result", response_model=TaskResultCallbackResponse)
async def callback_task_result(request: TaskResultCallbackRequest):
    """
    Callback endpoint for instances to report task completion.

    This endpoint is called by instances when a task completes or fails.
    The instance sends the task result, and the scheduler updates its state
    and notifies WebSocket subscribers.

    Args:
        request: Task result callback data

    Returns:
        TaskResultCallbackResponse with acknowledgment

    Raises:
        HTTPException 404: If task not found
        HTTPException 400: If task status is invalid
    """
    # Validate task exists
    task = await task_registry.get(request.task_id)

    #TODO: Resotre it after experiment
    if background_scheduler.scheduling_strategy.__class__.__name__ == "PowerOfTwoStrategy":
        request.execution_time_ms = 1.0
    if not task:
        error_msg = "Task not found"
        logger.error(f"[callback_task_result] {error_msg} | task_id={request.task_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    # Validate status
    if request.status not in ("completed", "failed"):
        error_msg = f"Invalid status: {request.status}. Must be 'completed' or 'failed'"
        logger.error(f"[callback_task_result] {error_msg} | task_id={request.task_id} | status={request.status}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg,
            },
        )

    # Handle task result via task dispatcher
    await task_dispatcher.handle_task_result(
        task_id=request.task_id,
        status=request.status,
        result=request.result,
        error=request.error,
        execution_time_ms=request.execution_time_ms,
    )

    return TaskResultCallbackResponse(
        success=True,
        message="Task result received successfully",
    )


@app.websocket("/task/get_result")
async def websocket_get_result(websocket: WebSocket):
    """
    WebSocket endpoint for real-time task result delivery.

    Clients can subscribe to multiple task IDs and receive results
    as soon as tasks complete or fail.

    Includes keepalive ping mechanism to prevent connection timeout.

    Args:
        websocket: WebSocket connection
    """
    await websocket.accept()

    # Register connection
    await websocket_manager.connect(websocket)

    # Keepalive configuration
    PING_INTERVAL = 10  # Send ping every 10 seconds

    async def send_keepalive():
        """Send periodic ping messages to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL)
                ping_msg = WSPingMessage(timestamp=asyncio.get_event_loop().time())
                await websocket.send_json(ping_msg.model_dump())
                logger.debug("Sent keepalive ping to WebSocket client")
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass
        except Exception as e:
            logger.warning(f"Keepalive task error: {e}")

    # Start keepalive task
    keepalive_task = asyncio.create_task(send_keepalive())

    try:
        while True:
            # Receive message from client (handle both text and ping/pong frames)
            message = await websocket.receive()

            # Handle WebSocket protocol-level ping/pong
            if message["type"] == "websocket.ping":
                # Respond to protocol-level ping with pong
                await websocket.send({"type": "websocket.pong", "bytes": message.get("bytes", b"")})
                logger.debug("Responded to protocol-level ping")
                continue

            elif message["type"] == "websocket.pong":
                # Received protocol-level pong, just log
                logger.debug("Received protocol-level pong")
                continue

            elif message["type"] == "websocket.disconnect":
                # Client disconnected
                logger.debug("Client initiated disconnect")
                break

            elif message["type"] != "websocket.receive":
                # Unknown message type
                logger.warning(f"Unknown WebSocket message type: {message['type']}")
                continue

            # Parse JSON data from text message
            try:
                data = json.loads(message.get("text", "{}"))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON message: {e}")
                error_msg = WSErrorMessage(error="Invalid JSON format")
                await websocket.send_json(error_msg.model_dump())
                continue

            # Parse message type
            message_type = data.get("type")

            if message_type == WSMessageType.PONG:
                # Client responded to our application-level ping, just log and continue
                logger.debug("Received application-level pong from WebSocket client")
                continue

            elif message_type == WSMessageType.PING:
                # Client sent application-level ping, respond with pong
                pong_msg = WSPongMessage(timestamp=asyncio.get_event_loop().time())
                await websocket.send_json(pong_msg.model_dump())
                logger.debug("Sent application-level pong response to WebSocket client")
                continue

            elif message_type == WSMessageType.SUBSCRIBE:
                # Parse task_ids
                task_ids = data.get("task_ids", [])

                if not isinstance(task_ids, list):
                    error_msg = WSErrorMessage(
                        error="task_ids must be a list of strings"
                    )
                    await websocket.send_json(error_msg.model_dump())
                    continue

                # Subscribe to tasks
                await websocket_manager.subscribe(websocket, task_ids)

                # For each task_id, check if already completed and send result immediately
                for task_id in task_ids:
                    task = await task_registry.get(task_id)
                    if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                        # Send result immediately
                        result_msg = WSTaskResultMessage(
                            task_id=task.task_id,
                            status=task.status,
                            result=task.result,
                            error=task.error,
                            timestamps=task.get_timestamps(),
                            execution_time_ms=task.execution_time_ms,
                        )
                        await websocket.send_json(result_msg.model_dump())

                # Send acknowledgment
                subscribed = await websocket_manager.get_subscribed_tasks(websocket)
                ack_msg = WSAckMessage(
                    message=f"Subscribed to {len(task_ids)} tasks",
                    subscribed_tasks=subscribed,
                )
                await websocket.send_json(ack_msg.model_dump())

            elif message_type == WSMessageType.UNSUBSCRIBE:
                # Parse task_ids
                task_ids = data.get("task_ids", [])

                if not isinstance(task_ids, list):
                    error_msg = WSErrorMessage(
                        error="task_ids must be a list of strings"
                    )
                    await websocket.send_json(error_msg.model_dump())
                    continue

                # Unsubscribe from tasks
                await websocket_manager.unsubscribe(websocket, task_ids)

                # Send acknowledgment
                subscribed = await websocket_manager.get_subscribed_tasks(websocket)
                ack_msg = WSAckMessage(
                    message=f"Unsubscribed from {len(task_ids)} tasks",
                    subscribed_tasks=subscribed,
                )
                await websocket.send_json(ack_msg.model_dump())

            else:
                # Unknown message type
                error_msg = WSErrorMessage(
                    error=f"Unknown message type: {message_type}"
                )
                await websocket.send_json(error_msg.model_dump())

    except WebSocketDisconnect:
        # Clean up subscriptions
        await websocket_manager.disconnect(websocket)
        logger.debug(f"WebSocket client disconnected")

    except Exception as e:
        # Log the error
        logger.error(f"WebSocket error: {e}", exc_info=True)
        # Send error message to client
        try:
            error_msg = WSErrorMessage(error=f"Server error: {str(e)}")
            await websocket.send_json(error_msg.model_dump())
        except Exception:
            pass

        # Clean up
        await websocket_manager.disconnect(websocket)

    finally:
        # Cancel keepalive task
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass
        logger.debug("WebSocket keepalive task cancelled")


# ============================================================================
# Strategy Management Helpers
# ============================================================================


async def reinitialize_instance_queues(
    strategy_name: str,
    quantiles: Optional[List[float]] = None
) -> int:
    """
    Reinitialize all instance queue info to match the new strategy type.

    Args:
        strategy_name: Name of the new scheduling strategy
        quantiles: Custom quantiles for probabilistic strategy (optional)

    Returns:
        Number of instances whose queue info was reinitialized
    """
    # Determine the queue info type for the new strategy
    if strategy_name == "min_time" or strategy_name == "po2" or strategy_name == "severless":
        queue_info_type = "expect_error"
    elif strategy_name == "probabilistic":
        queue_info_type = "probabilistic"
    else:  # round_robin
        queue_info_type = "probabilistic"  # Default to probabilistic for round_robin

    # Update the global queue_info_type FIRST to ensure consistency
    # This prevents race conditions where new instances might be registered
    # during the reinitialization process
    instance_registry._queue_info_type = queue_info_type

    # Update stored quantiles configuration if custom quantiles provided
    if quantiles:
        instance_registry._quantiles = sorted(set(quantiles))

    # Get all registered instances
    all_instances = await instance_registry.list_all()

    # Reinitialize queue info for each instance
    for instance in all_instances:
        if queue_info_type == "expect_error":
            # Initialize with InstanceQueueExpectError
            new_queue_info = InstanceQueueExpectError(
                instance_id=instance.instance_id,
                expected_time_ms=0.0,
                error_margin_ms=0.0,
            )
        else:  # probabilistic
            # Initialize with InstanceQueueProbabilistic
            # Use custom quantiles if provided, otherwise use default
            if quantiles:
                # Ensure quantiles are sorted and unique
                sorted_quantiles = sorted(set(quantiles))
                values = [0.0] * len(sorted_quantiles)
            else:
                # Default quantiles for distribution representation
                sorted_quantiles = [0.5, 0.9, 0.95, 0.99]
                values = [0.0, 0.0, 0.0, 0.0]

            new_queue_info = InstanceQueueProbabilistic(
                instance_id=instance.instance_id,
                quantiles=sorted_quantiles,
                values=values,
            )

        # Update the queue info in the registry
        await instance_registry.update_queue_info(instance.instance_id, new_queue_info)

    return len(all_instances)


def get_current_strategy_info() -> StrategyInfo:
    """
    Get information about the current scheduling strategy.

    Returns:
        StrategyInfo with strategy name and parameters
    """
    # Determine strategy name from the strategy instance
    strategy_class_name = scheduling_strategy.__class__.__name__

    if strategy_class_name == "MinimumExpectedTimeStrategy":
        strategy_name = "min_time"
        parameters = {}
    elif strategy_class_name == "ProbabilisticSchedulingStrategy":
        strategy_name = "probabilistic"
        # Get target_quantile from strategy instance
        target_quantile = getattr(scheduling_strategy, 'target_quantile', 0.9)
        parameters = {"target_quantile": target_quantile}
    elif strategy_class_name == "RoundRobinStrategy":
        strategy_name = "round_robin"
        parameters = {}
    elif strategy_class_name == "PowerOfTwoStrategy":
        strategy_name = "po2"
        parameters = {}
    else:
        strategy_name = "unknown"
        parameters = {}

    return StrategyInfo(
        strategy_name=strategy_name,
        parameters=parameters,
    )


# ============================================================================
# Strategy Management Endpoints
# ============================================================================


@app.get("/strategy/get", response_model=StrategyGetResponse)
async def get_strategy_endpoint():
    """
    Get the current scheduling strategy and its parameters.

    Returns:
        StrategyGetResponse with current strategy information
    """
    strategy_info = get_current_strategy_info()

    return StrategyGetResponse(
        success=True,
        strategy_info=strategy_info,
    )


@app.post("/strategy/set", response_model=StrategySetResponse)
async def set_strategy_endpoint(request: StrategySetRequest):
    """
    Set the scheduling strategy for the scheduler.

    This endpoint:
    1. Validates that no tasks are currently running
    2. Clears all tasks from the task queue
    3. Reinitializes instance queue info to match the new strategy
    4. Switches to the new scheduling strategy

    Args:
        request: Strategy configuration including name and parameters

    Returns:
        StrategySetResponse with operation details and new strategy info

    Raises:
        HTTPException 400: If tasks are currently running
        HTTPException 500: If strategy initialization fails
    """
    global scheduling_strategy

    random.seed(42)

    # Check if there are any running tasks
    running_count = await task_registry.get_count_by_status(TaskStatus.RUNNING)
    if running_count > 0:
        error_msg = f"Cannot switch strategy while {running_count} task(s) are running. Please wait for tasks to complete or fail them first."
        logger.error(f"[set_strategy] {error_msg} | strategy={request.strategy_name.value} | running_count={running_count}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg,
            },
        )

    # Clear all tasks from the task queue
    cleared_count = await task_registry.clear_all()
    logger.info(f"Cleared {cleared_count} tasks before switching strategy")

    logger.info(f"Switching the scheduling strategy to {request.strategy_name.value}, with the quantiles: {request.quantiles}")

    # Reinitialize instance queues to match the new strategy
    reinitialized_count = await reinitialize_instance_queues(
        request.strategy_name.value,
        quantiles=request.quantiles
    )
    logger.info(f"Reinitialized {reinitialized_count} instance queues for strategy '{request.strategy_name.value}'")

    # Create new scheduling strategy instance
    try:
        # Get target_quantile from request, or use default
        target_quantile = request.target_quantile if request.target_quantile is not None else 0.9

        new_strategy = get_strategy(
            strategy_name=request.strategy_name.value,
            predictor_client=predictor_client,
            instance_registry=instance_registry,
            target_quantile=target_quantile,
        )
        logger.info(f"Created new scheduling strategy: {request.strategy_name.value}")
    except Exception as e:
        logger.error(f"Failed to initialize strategy '{request.strategy_name.value}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Failed to initialize strategy: {str(e)}",
            },
        )

    # Update global scheduling strategy
    scheduling_strategy = new_strategy

    # Update BackgroundScheduler to use the new strategy
    background_scheduler.scheduling_strategy = new_strategy

    central_queue.set_scheduling_strategy(new_strategy)

    # Update config (in-memory only, not persisted)
    config.scheduling.default_strategy = request.strategy_name.value

    # Get the new strategy info
    strategy_info = get_current_strategy_info()

    random.seed(42)
    np.random.seed(42)
    
    logger.success(f"Successfully switched to strategy '{request.strategy_name.value}'")

    return StrategySetResponse(
        success=True,
        message=f"Successfully switched to '{request.strategy_name.value}' strategy",
        cleared_tasks=cleared_count,
        reinitialized_instances=reinitialized_count,
        strategy_info=strategy_info,
    )


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify scheduler status.

    Returns:
        HealthResponse with status and statistics

    Returns 503 if service is unhealthy
    """
    try:
        # Check if predictor service is reachable
        # TODO: Make this optional or configurable
        # predictor_healthy = await predictor_client.health_check()
        # if not predictor_healthy:
        #     raise Exception("Predictor service unavailable")

        # Collect statistics
        stats = HealthStats(
            total_instances=await instance_registry.get_total_count(),
            active_instances=await instance_registry.get_active_count(),
            total_tasks=await task_registry.get_total_count(),
            pending_tasks=await task_registry.get_count_by_status(TaskStatus.PENDING),
            running_tasks=await task_registry.get_count_by_status(TaskStatus.RUNNING),
            completed_tasks=await task_registry.get_count_by_status(TaskStatus.COMPLETED),
            failed_tasks=await task_registry.get_count_by_status(TaskStatus.FAILED),
        )

        return HealthResponse(
            success=True,
            status="healthy",
            timestamp=datetime.now().isoformat() + "Z",
            version="1.0.0",
            stats=stats,
        )

    except Exception as e:
        # Log the health check failure
        error_msg = str(e)
        logger.error(f"[health_check] Health check failed | error={error_msg}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "status": "unhealthy",
                "error": error_msg,
                "timestamp": datetime.now().isoformat() + "Z",
            },
        )



# ============================================================================
# Lifecycle Events (now handled by lifespan context manager)
# ============================================================================
