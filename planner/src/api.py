"""FastAPI application for the Planner service."""

from datetime import datetime, timezone
from typing import Optional, Dict
import numpy as np
import random
import asyncio
import time
from contextlib import asynccontextmanager

np.random.seed(42)
random.seed(42)

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from .models import (
    PlannerInput,
    PlannerOutput,
    DeploymentInput,
    DeploymentOutput,
    DeploymentStatus,
    InstanceRegisterRequest,
    InstanceRegisterResponse,
    MigrationOutput,
    SubmitTargetRequest,
    SubmitTargetResponse,
)
from .deployment_service import ModelMapper, InstanceDeployer, InstanceMigrator
from .core.swarm_optimizer import (
    SimulatedAnnealingOptimizer,
    IntegerProgrammingOptimizer,
)
from . import __version__
from .logging_config import setup_logging
from .config import config
from .available_instance_store import get_available_instance_store, AvailableInstance
from typing import List

# Configure logging
setup_logging()

# Global state for target distribution and model mapping
# These are set when /deploy or /deploy/migration is called
_stored_model_mapping: Optional[Dict[str, int]] = None
_stored_reverse_mapping: Optional[Dict[int, str]] = None
_current_target: Optional[List[float]] = None

# Global state for auto-optimization
_submitted_models: set = set()  # Track which models have submitted targets
_last_target_update: float = 0.0  # Timestamp of last target update
_auto_optimize_running: bool = False  # Flag to prevent concurrent optimizations
_stored_deployment_input: Optional["DeploymentInput"] = None  # Stored for auto-optimization
_auto_optimize_task: Optional[asyncio.Task] = None  # Background task handle


async def _auto_optimize_loop():
    """Background loop that checks conditions and triggers optimization."""
    global _auto_optimize_running

    logger.info(f"Auto-optimization loop started (interval={config.auto_optimize_interval}s, cooldown={config.auto_optimize_cooldown}s)")

    while True:
        try:
            await asyncio.sleep(config.auto_optimize_interval)

            # Skip if feature disabled (in case it's changed at runtime)
            if not config.auto_optimize_enabled:
                continue

            # Skip if no deployment has been made yet
            if _stored_model_mapping is None or _stored_deployment_input is None:
                logger.debug("Auto-optimization skipped: no deployment configured yet")
                continue

            # Skip if optimization is already running
            if _auto_optimize_running:
                logger.debug("Auto-optimization skipped: previous run still in progress")
                continue

            # Check if all models have submitted targets
            if len(_submitted_models) < len(_stored_model_mapping):
                logger.debug(f"Auto-optimization skipped: {len(_submitted_models)}/{len(_stored_model_mapping)} models submitted")
                continue

            # Check cooldown period
            time_since_update = time.time() - _last_target_update
            if _last_target_update == 0 or time_since_update < config.auto_optimize_cooldown:
                logger.debug(f"Auto-optimization skipped: cooldown not met ({time_since_update:.1f}s < {config.auto_optimize_cooldown}s)")
                continue

            # All conditions met, trigger optimization
            logger.info(f"Auto-optimization triggered: all {len(_submitted_models)} models submitted")
            await _trigger_optimization()

        except asyncio.CancelledError:
            logger.info("Auto-optimization loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in auto-optimization loop: {e}", exc_info=True)
            # Continue running despite errors


async def _trigger_optimization():
    """Execute the optimization with current targets."""
    global _auto_optimize_running, _submitted_models, _stored_model_mapping, _stored_reverse_mapping, _current_target

    _auto_optimize_running = True

    try:
        # Create a copy of the deployment input with current targets
        deployment_input = _stored_deployment_input.model_copy(deep=True)
        deployment_input.planner_input.target = _current_target.copy()

        logger.info(f"Running auto-optimization with targets: {_current_target}")

        # Import here to avoid circular import
        from .deployment_service import ModelMapper, InstanceMigrator

        # Run optimization similar to deploy_with_migration
        current_models = [inst.current_model for inst in deployment_input.instances]
        endpoints = [inst.endpoint for inst in deployment_input.instances]

        # Use existing mapping
        mapper = ModelMapper()
        model_mapping = _stored_model_mapping
        reverse_mapping = _stored_reverse_mapping

        # Map names to IDs
        initial_ids = mapper.map_names_to_ids(current_models, model_mapping)
        planner_params = deployment_input.planner_input.model_copy(deep=True)
        planner_params.initial = initial_ids

        # Run optimization
        B = np.array(planner_params.B)
        initial = np.array(planner_params.initial)
        target = np.array(planner_params.target)

        if planner_params.algorithm == "simulated_annealing":
            optimizer = SimulatedAnnealingOptimizer(
                M=planner_params.M,
                N=planner_params.N,
                B=B,
                initial=initial,
                a=0.1,
                target=target
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=planner_params.objective_method,
                initial_temp=planner_params.initial_temp,
                final_temp=planner_params.final_temp,
                cooling_rate=planner_params.cooling_rate,
                max_iterations=planner_params.max_iterations,
                iterations_per_temp=planner_params.iterations_per_temp,
                verbose=planner_params.verbose
            )
        else:
            optimizer = IntegerProgrammingOptimizer(
                M=planner_params.M,
                N=planner_params.N,
                B=B,
                initial=initial,
                a=planner_params.a,
                target=target
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=planner_params.objective_method,
                solver_name=planner_params.solver_name,
                time_limit=planner_params.time_limit,
                verbose=planner_params.verbose
            )

        changes_count = optimizer.compute_changes(deployment)

        # Get target models and perform migration
        deployment_target_models = [reverse_mapping[idx] for idx in deployment]
        instance_store = get_available_instance_store()

        pending_change_original = []
        pending_change_original_model = []
        pending_change_target = []

        for idx, (cur, target_model) in enumerate(zip(current_models, deployment_target_models)):
            if cur != target_model:
                pending_change_original.append(endpoints[idx])
                pending_change_original_model.append(cur)
                target_instance = await instance_store.fetch_one_available_instance(target_model)
                if not target_instance:
                    raise ValueError(f"No available target endpoint for model {target_model}")
                pending_change_target.append(target_instance.endpoint)

        # Perform migration if needed
        if pending_change_original:
            scheduler_mapping = config.get_scheduler_url(deployment_input.scheduler_mapping)
            migrator = InstanceMigrator(
                timeout=config.instance_timeout,
                scheduler_mapping=scheduler_mapping,
                max_retries=config.instance_max_retries,
                retry_delay=config.instance_retry_delay
            )
            migration_status = await migrator.migration_instances(
                pending_change_original,
                pending_change_target
            )

            # Add original endpoints back to store
            for model_id, endpoint in zip(pending_change_original_model, pending_change_original):
                await instance_store.add_available_instance(
                    AvailableInstance(
                        model_id=model_id,
                        endpoint=endpoint
                    )
                )

            failed = [s for s in migration_status if not s.success]
            logger.info(
                f"Auto-optimization completed: score={score:.4f}, "
                f"changes={changes_count}, failed={len(failed)}"
            )
        else:
            logger.info(f"Auto-optimization completed: score={score:.4f}, no changes needed")

        # Reset submitted models for next cycle
        _submitted_models.clear()

    except Exception as e:
        logger.error(f"Auto-optimization failed: {e}", exc_info=True)
        # Don't clear submitted_models on failure - allow retry next cycle

    finally:
        _auto_optimize_running = False


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
    global _auto_optimize_task

    # Startup
    if config.auto_optimize_enabled:
        logger.info(f"Starting auto-optimization loop (interval={config.auto_optimize_interval}s)")
        _auto_optimize_task = asyncio.create_task(_auto_optimize_loop())
    else:
        logger.info("Auto-optimization is disabled")

    yield

    # Shutdown
    if _auto_optimize_task is not None:
        logger.info("Stopping auto-optimization loop")
        _auto_optimize_task.cancel()
        try:
            await _auto_optimize_task
        except asyncio.CancelledError:
            pass
        logger.info("Auto-optimization loop stopped")


# Create FastAPI app
app = FastAPI(
    title="Planner Service",
    description="Model deployment optimization service for SwarmPilot",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}"
        }
    )


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancing.

    Returns:
        Health status with timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/info")
async def service_info():
    """
    Get service information and capabilities.

    Returns:
        Service metadata including version and supported algorithms
    """
    return {
        "service": "planner",
        "version": __version__,
        "algorithms": ["simulated_annealing", "integer_programming"],
        "objective_methods": ["relative_error", "ratio_difference", "weighted_squared"],
        "description": "Model deployment optimization service",
        "available_instances": get_available_instance_store().available_instances
    }


@app.post("/plan", response_model=PlannerOutput)
async def plan_deployment(input_data: PlannerInput):
    """
    Compute optimal deployment plan without execution.

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
        logger.info(f"Received /plan request: M={input_data.M}, N={input_data.N}, "
                   f"algorithm={input_data.algorithm}")

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
                target=target
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=input_data.objective_method,
                initial_temp=input_data.initial_temp,
                final_temp=input_data.final_temp,
                cooling_rate=input_data.cooling_rate,
                max_iterations=input_data.max_iterations,
                iterations_per_temp=input_data.iterations_per_temp,
                verbose=input_data.verbose
            )

        elif input_data.algorithm == "integer_programming":
            optimizer = IntegerProgrammingOptimizer(
                M=input_data.M,
                N=input_data.N,
                B=B,
                initial=initial,
                a=input_data.a,
                target=target
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=input_data.objective_method,
                solver_name=input_data.solver_name,
                time_limit=input_data.time_limit,
                verbose=input_data.verbose
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown algorithm: {input_data.algorithm}"
            )

        # Compute service capacity and changes
        service_capacity = optimizer.compute_service_capacity(deployment)
        changes_count = optimizer.compute_changes(deployment)

        result = PlannerOutput(
            deployment=deployment.tolist(),
            score=float(score),
            stats=stats,
            service_capacity=service_capacity.tolist(),
            changes_count=int(changes_count)
        )

        logger.info(f"Optimization completed: score={score:.4f}, changes={changes_count}")
        return result

    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Algorithm dependency not available: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@app.post("/deploy/migration", response_model=MigrationOutput)
async def deploy_with_migration(input_data: DeploymentInput):
    """
    Compute optimal deployment plan and execute it across instances with migration mode.

    Workflow:
    1. Extract model names from instances
    2. Create model name → ID mapping
    3. Run optimization algorithm
    4. Map result IDs back to model names
    5. Deploy to instances concurrently with migration mode
    6. Return results with detailed status

    Args:
        input_data: Deployment configuration with instances and optimization parameters

    Returns:
        DeploymentOutput: Optimization results plus deployment execution status

    Raises:
        HTTPException: If optimization or deployment fails
    """
    try:
        logger.info(f"Received /deploy/migration request: {len(input_data.instances)} instances, "
                   f"algorithm={input_data.planner_input.algorithm}")

        # Step 1: Extract model names from instances
        current_models = [inst.current_model for inst in input_data.instances]
        endpoints = [inst.endpoint for inst in input_data.instances]

        logger.info(f"Current models: {current_models}")

        # Step 2: Create model name → ID mapping
        mapper = ModelMapper()
        model_mapping = mapper.create_mapping(current_models)
        reverse_mapping = {v: k for k, v in model_mapping.items()}

        logger.info(f"Model mapping: {model_mapping}")

        # Store mapping for submit_target use
        global _stored_model_mapping, _stored_reverse_mapping, _current_target, _stored_deployment_input, _submitted_models
        _stored_model_mapping = model_mapping
        _stored_reverse_mapping = reverse_mapping
        _current_target = [0.0] * len(model_mapping)
        _stored_deployment_input = input_data.model_copy(deep=True)  # Store for auto-optimization
        _submitted_models.clear()  # Reset submitted models for new deployment
        logger.info(f"Stored model mapping for submit_target: {len(model_mapping)} models")

        # Update planner_input.initial with mapped IDs
        try:
            initial_ids = mapper.map_names_to_ids(current_models, model_mapping)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model mapping failed: {str(e)}"
            )

        # Override the initial field in planner_input
        planner_params = input_data.planner_input.model_copy(deep=True)
        planner_params.initial = initial_ids

        # Step 3: Run optimization
        B = np.array(planner_params.B)
        initial = np.array(planner_params.initial)
        target = np.array(planner_params.target)

        if planner_params.algorithm == "simulated_annealing":
            optimizer = SimulatedAnnealingOptimizer(
                M=planner_params.M,
                N=planner_params.N,
                B=B,
                initial=initial,
                a=planner_params.a,
                target=target
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=planner_params.objective_method,
                initial_temp=planner_params.initial_temp,
                final_temp=planner_params.final_temp,
                cooling_rate=planner_params.cooling_rate,
                max_iterations=planner_params.max_iterations,
                iterations_per_temp=planner_params.iterations_per_temp,
                verbose=planner_params.verbose
            )

        elif planner_params.algorithm == "integer_programming":
            optimizer = IntegerProgrammingOptimizer(
                M=planner_params.M,
                N=planner_params.N,
                B=B,
                initial=initial,
                a=planner_params.a,
                target=target
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=planner_params.objective_method,
                solver_name=planner_params.solver_name,
                time_limit=planner_params.time_limit,
                verbose=planner_params.verbose
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown algorithm: {planner_params.algorithm}"
            )

        service_capacity = optimizer.compute_service_capacity(deployment)
        changes_count = optimizer.compute_changes(deployment)

        # Get the target instance list for all instances
        # We need to find the changed instance here
        instance_store = get_available_instance_store()
        deployment_target_models = [reverse_mapping[idx] for idx in deployment]
        pending_change_original_model = []
        pending_change_original = []
        pending_change_target = []
        for idx, (cur, target_model) in enumerate(zip(current_models, deployment_target_models)):
            if cur != target_model:
                pending_change_original.append(endpoints[idx])
                pending_change_original_model.append(cur)
                target_instance = await instance_store.fetch_one_available_instance(target_model)
                if not target_instance:
                    raise ValueError(f"No available target endpoint for model {target_model}")
                pending_change_target.append(target_instance.endpoint)
                
            

        logger.info(f"Optimization completed: score={score:.4f}, changes={changes_count}")

        # Step 4: Map result IDs back to model names
        try:
            target_models = mapper.map_ids_to_names(deployment.tolist(), reverse_mapping)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to map deployment IDs to names: {str(e)}"
            )

        logger.info(f"Target models: {target_models}")

        # Step 5: Deploy to instances
        # Use scheduler_url from request or fall back to config default
        scheduler_mapping = config.get_scheduler_url(input_data.scheduler_mapping)
        logger.debug(f"Using scheduler URL: {scheduler_mapping}")

        # Perform migration for instances that need to change
        migration_status = []
        if pending_change_original:
            migrator = InstanceMigrator(
                timeout=config.instance_timeout,
                scheduler_mapping=scheduler_mapping,
                max_retries=config.instance_max_retries,
                retry_delay=config.instance_retry_delay
            )
            migration_status = await migrator.migration_instances(
                pending_change_original,
                pending_change_target
            )
            logger.info(f"Migration completed for {len(pending_change_original)} instances")

        # Append original endpoint to the storage for future use
        for model_id, endpoint in zip(pending_change_original_model, pending_change_original):
            await instance_store.add_available_instance(
                AvailableInstance(
                    model_id=model_id,
                    endpoint=endpoint
                )
            )

        # Step 6: Aggregate results
        failed_instances = [
            status.instance_index
            for status in migration_status
            if not status.success
        ]
        overall_success = len(failed_instances) == 0

        if not overall_success:
            logger.warning(f"Deployment partially failed: {len(failed_instances)} instances failed")
        else:
            logger.info("Deployment completed successfully for all instances")

        result = MigrationOutput(
            deployment=deployment.tolist(),
            score=float(score),
            stats=stats,
            service_capacity=service_capacity.tolist(),
            changes_count=int(changes_count),
            deployment_status=migration_status,
            success=overall_success,
            failed_instances=failed_instances
        )

        return result

    except HTTPException:
        raise
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Algorithm dependency not available: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deployment failed: {str(e)}"
        )

@app.post("/deploy", response_model=DeploymentOutput)
async def deploy_with_optimization(input_data: DeploymentInput):
    """
    Compute optimal deployment plan and execute it across instances.

    Workflow:
    1. Extract model names from instances
    2. Create model name → ID mapping
    3. Run optimization algorithm
    4. Map result IDs back to model names
    5. Deploy to instances concurrently
    6. Return results with detailed status

    Args:
        input_data: Deployment configuration with instances and optimization parameters

    Returns:
        DeploymentOutput: Optimization results plus deployment execution status

    Raises:
        HTTPException: If optimization or deployment fails
    """
    try:
        logger.info(f"Received /deploy request: {len(input_data.instances)} instances, "
                   f"algorithm={input_data.planner_input.algorithm}")

        # Step 1: Extract model names from instances
        current_models = [inst.current_model for inst in input_data.instances]
        endpoints = [inst.endpoint for inst in input_data.instances]

        logger.info(f"Current models: {current_models}")

        # Step 2: Create model name → ID mapping
        mapper = ModelMapper()
        model_mapping = mapper.create_mapping(current_models)
        reverse_mapping = {v: k for k, v in model_mapping.items()}

        logger.info(f"Model mapping: {model_mapping}")

        # Store mapping for submit_target use
        global _stored_model_mapping, _stored_reverse_mapping, _current_target, _stored_deployment_input, _submitted_models
        _stored_model_mapping = model_mapping
        _stored_reverse_mapping = reverse_mapping
        _current_target = [0.0] * len(model_mapping)
        _stored_deployment_input = input_data.model_copy(deep=True)  # Store for auto-optimization
        _submitted_models.clear()  # Reset submitted models for new deployment
        logger.info(f"Stored model mapping for submit_target: {len(model_mapping)} models")

        # Update planner_input.initial with mapped IDs
        try:
            initial_ids = mapper.map_names_to_ids(current_models, model_mapping)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model mapping failed: {str(e)}"
            )

        # Override the initial field in planner_input
        planner_params = input_data.planner_input.model_copy(deep=True)
        planner_params.initial = initial_ids

        # Step 3: Run optimization
        B = np.array(planner_params.B)
        initial = np.array(planner_params.initial)
        target = np.array(planner_params.target)

        if planner_params.algorithm == "simulated_annealing":
            optimizer = SimulatedAnnealingOptimizer(
                M=planner_params.M,
                N=planner_params.N,
                B=B,
                initial=initial,
                a=planner_params.a,
                target=target
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=planner_params.objective_method,
                initial_temp=planner_params.initial_temp,
                final_temp=planner_params.final_temp,
                cooling_rate=planner_params.cooling_rate,
                max_iterations=planner_params.max_iterations,
                iterations_per_temp=planner_params.iterations_per_temp,
                verbose=planner_params.verbose
            )

        elif planner_params.algorithm == "integer_programming":
            optimizer = IntegerProgrammingOptimizer(
                M=planner_params.M,
                N=planner_params.N,
                B=B,
                initial=initial,
                a=planner_params.a,
                target=target
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=planner_params.objective_method,
                solver_name=planner_params.solver_name,
                time_limit=planner_params.time_limit,
                verbose=planner_params.verbose
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown algorithm: {planner_params.algorithm}"
            )

        service_capacity = optimizer.compute_service_capacity(deployment)
        changes_count = optimizer.compute_changes(deployment)

        logger.info(f"Optimization completed: score={score:.4f}, changes={changes_count}")

        # Step 4: Map result IDs back to model names
        try:
            target_models = mapper.map_ids_to_names(deployment.tolist(), reverse_mapping)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to map deployment IDs to names: {str(e)}"
            )

        logger.info(f"Target models: {target_models}")

        # Step 5: Deploy to instances
        # Use scheduler_url from request or fall back to config default
        # Note: DeploymentInput uses scheduler_mapping, get first URL if available
        first_scheduler = None
        if input_data.scheduler_mapping:
            first_scheduler = next(iter(input_data.scheduler_mapping.values()), None)
        scheduler_url = config.get_scheduler_url(first_scheduler)
        logger.debug(f"Using scheduler URL: {scheduler_url}")

        deployer = InstanceDeployer(
            timeout=config.instance_timeout,
            scheduler_url=scheduler_url,
            max_retries=config.instance_max_retries,
            retry_delay=config.instance_retry_delay
        )
        deployment_statuses = await deployer.deploy_to_instances(
            endpoints=endpoints,
            target_models=target_models,
            previous_models=current_models
        )

        # Step 6: Aggregate results
        failed_instances = [
            status.instance_index
            for status in deployment_statuses
            if not status.success
        ]
        overall_success = len(failed_instances) == 0

        if not overall_success:
            logger.warning(f"Deployment partially failed: {len(failed_instances)} instances failed")
        else:
            logger.info("Deployment completed successfully for all instances")

        result = DeploymentOutput(
            deployment=deployment.tolist(),
            score=float(score),
            stats=stats,
            service_capacity=service_capacity.tolist(),
            changes_count=int(changes_count),
            deployment_status=deployment_statuses,
            success=overall_success,
            failed_instances=failed_instances
        )

        return result

    except HTTPException:
        raise
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Algorithm dependency not available: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deployment failed: {str(e)}"
        )


@app.post("/instance/register", response_model=InstanceRegisterResponse)
async def register_available_instance(request: InstanceRegisterRequest):
    """
    Register an available instance to the planner's available instance store.

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
            model_id=request.model_id,
            endpoint=request.endpoint
        )

        await instance_store.add_available_instance(available_instance)

        logger.info(
            f"Successfully registered instance {request.instance_id} "
            f"for model {request.model_id}"
        )

        return InstanceRegisterResponse(
            success=True,
            message=f"Instance {request.instance_id} registered successfully for model {request.model_id}"
        )

    except Exception as e:
        logger.error(f"Failed to register instance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register instance: {str(e)}"
        )

@app.get("/migration/info")
def get_migration_info() -> Dict[str, AvailableInstance]:
    """
    this endpoint is show current available instances for migration
    """
    instance_store = get_available_instance_store()

    return instance_store.available_instances


@app.post("/submit_target", response_model=SubmitTargetResponse)
async def submit_target(request: SubmitTargetRequest):
    """
    Submit queue length from scheduler to update target distribution.

    This endpoint receives queue length data from each scheduler and accumulates
    it into the target distribution array. Only effective after /deploy or
    /deploy/migration has been called to establish the model mapping.

    Args:
        request: Queue length data with model_id and value

    Returns:
        SubmitTargetResponse with update status and current target
    """
    global _current_target, _stored_model_mapping, _submitted_models, _last_target_update

    # No error if mapping doesn't exist, just no effect
    if _stored_model_mapping is None:
        logger.info("submit_target called but no mapping exists yet")
        return SubmitTargetResponse(
            success=True,
            message="No active mapping. Call /deploy or /deploy/migration first.",
            current_target=None
        )

    # Check if model_id exists in mapping
    if request.model_id not in _stored_model_mapping:
        logger.info(f"submit_target: model_id {request.model_id} not in mapping")
        return SubmitTargetResponse(
            success=True,
            message=f"Model {request.model_id} not in current mapping",
            current_target=_current_target
        )

    # Update target at corresponding position
    idx = _stored_model_mapping[request.model_id]
    _current_target[idx] = request.value

    # Track submitted models and update timestamp
    _submitted_models.add(request.model_id)
    _last_target_update = time.time()

    logger.info(f"Updated target[{idx}] for model {request.model_id} to {request.value} ({len(_submitted_models)}/{len(_stored_model_mapping)} models submitted)")

    return SubmitTargetResponse(
        success=True,
        message=f"Updated target[{idx}] for {request.model_id} ({len(_submitted_models)}/{len(_stored_model_mapping)} submitted)",
        current_target=_current_target
    )


@app.get("/target")
async def get_target():
    """
    Get current accumulated target distribution.

    Returns the current target array and model mapping set by /deploy or
    /deploy/migration, updated by /submit_target calls.

    Returns:
        Dictionary with target array and model mapping
    """
    return {
        "target": _current_target,
        "model_mapping": _stored_model_mapping,
        "reverse_mapping": _stored_reverse_mapping
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
