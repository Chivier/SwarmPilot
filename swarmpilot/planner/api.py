"""FastAPI application for the Planner service.

This module provides the main FastAPI application for the Planner service,
which handles model deployment optimization using PyLet cluster management.
"""

import random
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from . import __version__
from .available_instance_store import (
    AvailableInstance,
    get_available_instance_store,
)
from .config import config
from .core.swarm_optimizer import (
    IntegerProgrammingOptimizer,
    SimulatedAnnealingOptimizer,
)
from .logging_config import setup_logging
from .models import (
    InstanceDrainRequest,
    InstanceDrainResponse,
    InstanceDrainStatusResponse,
    InstanceRegisterRequest,
    InstanceRegisterResponse,
    InstanceRemoveRequest,
    InstanceRemoveResponse,
    InstanceStatus,
    PlannerInput,
    PlannerOutput,
    SchedulerDeregisterRequest,
    SchedulerDeregisterResponse,
    SchedulerListResponse,
    SchedulerRegisterRequest,
    SchedulerRegisterResponse,
    TaskResubmitRequest,
    TaskResubmitResponse,
)
from .scheduler_registry import get_scheduler_registry

try:
    from .pylet.deployment_service import (
        create_pylet_service,
        get_pylet_service_optional,
    )
    from .pylet_api import router as pylet_router
except ImportError:
    create_pylet_service = None  # type: ignore[assignment]
    get_pylet_service_optional = lambda: None  # type: ignore[assignment]
    pylet_router = None  # type: ignore[assignment]

np.random.seed(42)
random.seed(42)

# Configure logging
setup_logging()


# Validate configuration on startup
try:
    config.validate()
    logger.info("Configuration validated successfully")
    if config.scheduler_url:
        logger.info(f"Default scheduler URL: {config.scheduler_url}")
except ValueError as e:
    logger.error(f"Configuration validation failed: {e}")
    raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Initialize PyLet if enabled
    pylet_service = None
    if config.pylet_enabled and create_pylet_service is not None:
        try:
            logger.info(
                f"Initializing PyLet service (head={config.pylet_head_url}, "
                f"backend={config.pylet_backend}, "
                f"reuse_cluster={config.pylet_reuse_cluster})"
            )
            pylet_service = create_pylet_service(
                pylet_head_url=config.pylet_head_url,
                scheduler_url=config.scheduler_url or "http://localhost:8001",
                default_backend=config.pylet_backend,
                default_gpu_count=config.pylet_gpu_count,
                default_cpu_count=config.pylet_cpu_count,
                deploy_timeout=config.pylet_deploy_timeout,
                drain_timeout=config.pylet_drain_timeout,
                custom_command=config.pylet_custom_command,
                reuse_cluster=config.pylet_reuse_cluster,
            )
            logger.info("PyLet service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PyLet service: {e}")
            logger.warning("Continuing without PyLet - endpoints will be unavailable")
    else:
        logger.info("PyLet is disabled")

    yield

    # Shutdown: Close PyLet service
    if pylet_service is not None:
        logger.info("Closing PyLet service")
        pylet_service.close()
        logger.info("PyLet service closed")


# Create FastAPI app
app = FastAPI(
    title="Planner Service",
    description="Model deployment optimization service for SwarmPilot",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Include PyLet router (only if pylet SDK is available)
if pylet_router is not None:
    app.include_router(pylet_router, prefix="/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}",
        },
    )


@app.get("/v1/health")
async def health_check():
    """Health check endpoint for monitoring and load balancing.

    Returns:
        Health status with timestamp
    """
    return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/v1/info")
async def service_info():
    """Get service information and capabilities.

    Returns:
        Service metadata including version and supported algorithms
    """
    # Get PyLet status
    pylet_service = get_pylet_service_optional()
    pylet_status = {
        "enabled": config.pylet_enabled,
        "initialized": pylet_service is not None and pylet_service.initialized,
    }
    if pylet_service and pylet_service.initialized:
        pylet_status["active_instances"] = len(pylet_service.get_active_instances())
        pylet_status["current_state"] = pylet_service.get_current_state()

    # Get scheduler registry info
    registry = get_scheduler_registry()
    scheduler_registry_info = {
        "total": len(registry),
        "models": registry.get_registered_models(),
    }

    return {
        "service": "planner",
        "version": __version__,
        "algorithms": ["simulated_annealing", "integer_programming"],
        "objective_methods": [
            "relative_error",
            "ratio_difference",
            "weighted_squared",
        ],
        "description": "Model deployment optimization service",
        "available_instances": get_available_instance_store().available_instances,
        "pylet": pylet_status,
        "scheduler_registry": scheduler_registry_info,
    }


@app.post("/v1/plan", response_model=PlannerOutput)
async def plan_deployment(input_data: PlannerInput):
    """Compute optimal deployment plan without execution.

    This endpoint runs the optimization algorithm to find the best model
    deployment configuration but does not deploy to any instances.

    Args:
        input_data: Optimization parameters and configuration

    Returns:
        PlannerOutput: Optimal deployment plan with score and statistics

    Raises:
        HTTPException: If optimization fails or parameters are invalid
    """
    try:
        # Log planning request
        logger.info(
            f"[PLAN_REQUEST] endpoint=/plan M={input_data.M} N={input_data.N} "
            f"algorithm={input_data.algorithm} objective={input_data.objective_method} "
            f"change_factor={input_data.a} target={input_data.target}"
        )

        # Convert inputs to numpy arrays
        B = np.array(input_data.B)
        initial = np.array(input_data.initial)
        target = np.array(input_data.target)

        # Select optimizer based on algorithm
        if input_data.algorithm == "simulated_annealing":
            optimizer = SimulatedAnnealingOptimizer(
                M=input_data.M,
                N=input_data.N,
                B=B,
                initial=initial,
                a=input_data.a,
                target=target,
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=input_data.objective_method,
                initial_temp=input_data.initial_temp,
                final_temp=input_data.final_temp,
                cooling_rate=input_data.cooling_rate,
                max_iterations=input_data.max_iterations,
                iterations_per_temp=input_data.iterations_per_temp,
                verbose=input_data.verbose,
            )

        elif input_data.algorithm == "integer_programming":
            optimizer = IntegerProgrammingOptimizer(
                M=input_data.M,
                N=input_data.N,
                B=B,
                initial=initial,
                a=input_data.a,
                target=target,
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=input_data.objective_method,
                solver_name=input_data.solver_name,
                time_limit=input_data.time_limit,
                verbose=input_data.verbose,
            )

        else:
            error_msg = f"Unknown algorithm: {input_data.algorithm}"
            client_msg = error_msg
            logger.error(
                f"/plan request failed: {error_msg}. Returning HTTP 400. Client will receive: {client_msg}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=client_msg
            )

        # Compute service capacity and changes
        service_capacity = optimizer.compute_service_capacity(deployment)
        changes_count = optimizer.compute_changes(deployment)

        result = PlannerOutput(
            deployment=deployment.tolist(),
            score=float(score),
            stats=stats,
            service_capacity=service_capacity.tolist(),
            changes_count=int(changes_count),
        )

        logger.info(
            f"Optimization completed: score={score:.4f}, changes={changes_count}"
        )
        return result

    except ImportError as e:
        client_msg = f"Algorithm dependency not available: {str(e)}"
        logger.error(
            f"/plan request failed - ImportError: {e}. Returning HTTP 500. Client will receive: {client_msg}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=client_msg
        )
    except ValueError as e:
        client_msg = f"Invalid input: {str(e)}"
        logger.error(
            f"/plan request failed - ValueError: {e}. Returning HTTP 400. Client will receive: {client_msg}",
            exc_info=True,
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=client_msg)
    except Exception as e:
        client_msg = f"Optimization failed: {str(e)}"
        logger.error(
            f"/plan request failed - Unexpected error: {e}. Returning HTTP 500. Client will receive: {client_msg}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=client_msg
        )


# ============================================================================
# Scheduler Registry Endpoints (PYLET-024)
# These endpoints allow schedulers to register/deregister with the planner,
# enabling per-model scheduler routing.
# ============================================================================


@app.post("/v1/scheduler/register", response_model=SchedulerRegisterResponse)
async def register_scheduler(request: SchedulerRegisterRequest):
    """Register a scheduler for a specific model.

    Schedulers call this on startup to advertise their URL and the model
    they handle. If a scheduler for the same model was already registered,
    it is replaced.

    Args:
        request: Scheduler registration details.

    Returns:
        SchedulerRegisterResponse with registration status.
    """
    registry = get_scheduler_registry()
    replaced = registry.register(
        model_id=request.model_id,
        scheduler_url=request.scheduler_url,
        metadata=request.metadata,
    )

    return SchedulerRegisterResponse(
        success=True,
        message=(
            f"Scheduler for {request.model_id} registered at "
            f"{request.scheduler_url}"
        ),
        replaced_previous=replaced,
    )


@app.post(
    "/v1/scheduler/deregister", response_model=SchedulerDeregisterResponse
)
async def deregister_scheduler(request: SchedulerDeregisterRequest):
    """Deregister a scheduler for a specific model.

    Schedulers call this on graceful shutdown.

    Args:
        request: Scheduler deregistration details.

    Returns:
        SchedulerDeregisterResponse with deregistration status.
    """
    registry = get_scheduler_registry()
    found = registry.deregister(request.model_id)

    if found:
        return SchedulerDeregisterResponse(
            success=True,
            message=f"Scheduler for {request.model_id} deregistered",
        )
    else:
        return SchedulerDeregisterResponse(
            success=False,
            message=f"No scheduler registered for {request.model_id}",
        )


@app.get("/v1/scheduler/list", response_model=SchedulerListResponse)
async def list_schedulers():
    """List all registered schedulers.

    Returns:
        SchedulerListResponse with all registered schedulers.
    """
    registry = get_scheduler_registry()
    schedulers = registry.list_all()
    return SchedulerListResponse(
        schedulers=schedulers,
        total=len(schedulers),
    )


@app.get("/v1/scheduler/{model_id}")
async def get_scheduler(model_id: str):
    """Get the scheduler registered for a specific model.

    Args:
        model_id: Model identifier.

    Returns:
        Scheduler info or 404 if not registered.
    """
    registry = get_scheduler_registry()
    info = registry.get_scheduler_info(model_id)

    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No scheduler registered for model: {model_id}",
        )

    return info


# Legacy /deploy and /deploy/migration endpoints have been removed.
# Use the PyLet endpoints (/deploy, /optimize) for deployment.


@app.post("/v1/instance/register", response_model=InstanceRegisterResponse)
async def register_available_instance(request: InstanceRegisterRequest):
    """Register an available instance to the planner's available instance store.

    This endpoint has the same parameters as the scheduler's /instance/register
    but stores instances for migration-based redeployment instead of task scheduling.

    Args:
        request: Instance registration details (instance_id, model_id, endpoint, platform_info)

    Returns:
        InstanceRegisterResponse with registration status

    Raises:
        HTTPException: If registration fails
    """
    try:
        logger.info(
            f"Registering available instance: {request.instance_id} "
            f"for model {request.model_id} at {request.endpoint}"
        )

        # Get the available instance store
        instance_store = get_available_instance_store()

        # Create AvailableInstance and add to store
        available_instance = AvailableInstance(
            model_id=request.model_id, endpoint=request.endpoint
        )

        await instance_store.add_available_instance(available_instance)

        logger.info(
            f"Successfully registered instance {request.instance_id} "
            f"for model {request.model_id}"
        )

        return InstanceRegisterResponse(
            success=True,
            message=f"Instance {request.instance_id} registered successfully for model {request.model_id}",
        )

    except Exception as e:
        client_msg = f"Failed to register instance: {str(e)}"
        logger.error(
            f"/instance/register failed - Error: {e}. Returning HTTP 500. Client will receive: {client_msg}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=client_msg
        )


# ============================================================================
# Legacy endpoints removed. Use PyLet API instead:
# - /status - Get cluster status and active instances
# - /deploy - Deploy instances to target state
# - /scale - Scale a specific model
# - /migrate - Migrate an instance
# - /optimize - Run optimizer and deploy via PyLet
# ============================================================================


# ============================================================================
# Dummy Endpoints (compatible with Scheduler interface)
# These endpoints allow instances registered to Planner to properly deregister
# without requiring actual Scheduler functionality.
# ============================================================================


@app.post("/v1/instance/drain", response_model=InstanceDrainResponse)
async def dummy_drain_instance(request: InstanceDrainRequest):
    """Dummy drain endpoint for instances registered to Planner.

    Always returns success since Planner doesn't manage task queues.
    This allows instances registered to Planner to complete their
    deregister flow without errors.

    Args:
        request: Instance drain request with instance_id

    Returns:
        InstanceDrainResponse with success status
    """
    logger.info(f"[Dummy] Drain requested for instance: {request.instance_id}")
    return InstanceDrainResponse(
        success=True,
        message=f"Instance {request.instance_id} drain acknowledged (Planner dummy)",
        instance_id=request.instance_id,
        status=InstanceStatus.DRAINING,
        pending_tasks=0,
        running_tasks=0,
        estimated_completion_time_ms=0.0,
    )


@app.get("/v1/instance/drain/status", response_model=InstanceDrainStatusResponse)
async def dummy_drain_status(instance_id: str):
    """Dummy drain status endpoint for instances registered to Planner.

    Always returns can_remove=True since Planner doesn't manage task queues.

    Args:
        instance_id: Instance ID to check drain status

    Returns:
        InstanceDrainStatusResponse with can_remove=True
    """
    logger.info(f"[Dummy] Drain status check for instance: {instance_id}")
    return InstanceDrainStatusResponse(
        success=True,
        instance_id=instance_id,
        status=InstanceStatus.REMOVING,
        pending_tasks=0,
        running_tasks=0,
        can_remove=True,
        drain_initiated_at=None,
    )


@app.post("/v1/instance/remove", response_model=InstanceRemoveResponse)
async def dummy_remove_instance(request: InstanceRemoveRequest):
    """Dummy remove endpoint for instances registered to Planner.

    Always returns success. This allows instances registered to Planner
    to complete their deregister flow without errors.

    Args:
        request: Instance remove request with instance_id

    Returns:
        InstanceRemoveResponse with success status
    """
    logger.info(f"[Dummy] Remove requested for instance: {request.instance_id}")
    return InstanceRemoveResponse(
        success=True,
        message=f"Instance {request.instance_id} removed (Planner dummy)",
        instance_id=request.instance_id,
    )


@app.post("/v1/task/resubmit", response_model=TaskResubmitResponse)
async def dummy_resubmit_task(request: TaskResubmitRequest):
    """Dummy task resubmit endpoint for instances registered to Planner.

    Always returns success. Planner doesn't manage task queues,
    so resubmission is a no-op.

    Args:
        request: Task resubmit request with task_id and original_instance_id

    Returns:
        TaskResubmitResponse with success status
    """
    logger.info(
        f"[Dummy] Task resubmit requested: task={request.task_id}, "
        f"original_instance={request.original_instance_id}"
    )
    return TaskResubmitResponse(
        success=True,
        message=f"Task {request.task_id} resubmit acknowledged (Planner dummy)",
    )


# ============================================================================
# Timeline Tracking Endpoints
# These endpoints provide access to instance count timeline data
# ============================================================================


@app.get("/v1/timeline")
async def get_instance_timeline():
    """Get the instance count timeline.

    Returns all recorded migration events with instance counts per model.
    Each entry includes timestamp, event type, instance counts, and metrics.

    Returns:
        Dictionary with success status and list of timeline entries
    """
    from .instance_timeline_tracker import get_timeline_tracker

    tracker = get_timeline_tracker()
    entries = tracker.get_entries()
    logger.info(f"Timeline requested: {len(entries)} entries")
    return {"success": True, "entry_count": len(entries), "entries": entries}


@app.post("/v1/timeline/clear")
async def clear_instance_timeline():
    """Clear the instance count timeline.

    Should be called at the start of a new experiment to ensure
    clean timeline data.

    Returns:
        Dictionary with success status and confirmation message
    """
    from .instance_timeline_tracker import get_timeline_tracker

    tracker = get_timeline_tracker()
    tracker.clear()
    logger.info("Timeline cleared via API")
    return {"success": True, "message": "Timeline cleared successfully"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
