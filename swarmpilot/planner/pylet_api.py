"""PyLet API endpoints for the Planner service (PYLET-014).

This module provides FastAPI endpoints for PyLet-based instance management.
It integrates with the optimizer to provide automated scaling and deployment.

Endpoints:
    GET  /status - Get PyLet service status
    POST /deploy - Run optimizer and deploy result via PyLet (same input as /plan)
    POST /deploy_manually - Deploy instances to manually specified target state
    POST /scale  - Scale a specific model
    POST /migrate - Migrate an instance
    POST /optimize - Run optimizer and deploy via PyLet (simplified input)
    POST /terminate-all - Terminate all PyLet instances
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from .config import config
from .core.swarm_optimizer import (
    IntegerProgrammingOptimizer,
    SimulatedAnnealingOptimizer,
)
from .models import (
    PyLetDeploymentInput,
    PyLetDeploymentOutput,
    PyLetDeployWithPlanInput,
    PyLetInstanceStatus,
    PyLetMigrateInput,
    PyLetMigrateOutput,
    PyLetOptimizeInput,
    PyLetOptimizeOutput,
    PyLetScaleInput,
    PyLetScaleOutput,
    PyLetStatusOutput,
)
from .pylet.deployment_service import get_pylet_service_optional
from .scheduler_registry import get_scheduler_registry
from .services.model_validation import ModelValidationService

# Create router for PyLet endpoints (top-level API)
router = APIRouter(tags=["pylet"])


def _instance_to_status(instance) -> PyLetInstanceStatus:
    """Convert ManagedInstance to PyLetInstanceStatus."""
    return PyLetInstanceStatus(
        pylet_id=instance.pylet_id,
        instance_id=instance.instance_id,
        model_id=instance.model_id,
        endpoint=instance.endpoint,
        status=instance.status.value,
        error=instance.error,
    )


def _ensure_pylet_enabled():
    """Ensure PyLet is enabled and initialized.

    Raises:
        HTTPException: If PyLet is not enabled or not initialized.
    """
    if not config.pylet_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PyLet is not enabled. Set PYLET_ENABLED=true",
        )

    service = get_pylet_service_optional()
    if service is None or not service.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PyLet service not initialized",
        )

    return service


@router.get("/status", response_model=PyLetStatusOutput)
async def pylet_status():
    """Get PyLet service status.

    Returns:
        PyLetStatusOutput with current service state.
    """
    service = get_pylet_service_optional()

    if service is None or not service.initialized:
        return PyLetStatusOutput(
            enabled=config.pylet_enabled,
            initialized=False,
            current_state={},
            total_instances=0,
            active_instances=[],
        )

    current_state = service.get_current_state()
    active = service.get_active_instances()

    return PyLetStatusOutput(
        enabled=True,
        initialized=True,
        current_state=current_state,
        total_instances=len(active),
        active_instances=[_instance_to_status(i) for i in active],
    )


@router.post("/deploy_manually", response_model=PyLetDeploymentOutput)
async def pylet_deploy_manually(input_data: PyLetDeploymentInput):
    """Deploy instances to target state via PyLet (manual specification).

    This endpoint reconciles the current state to the target state,
    adding or removing instances as needed. Use this when you have
    a pre-computed target_state dict.

    For automatic optimization and deployment, use /deploy instead.

    Args:
        input_data: Target state configuration with {model_id: count}.

    Returns:
        PyLetDeploymentOutput with deployment results.
    """
    service = _ensure_pylet_enabled()

    logger.info(
        f"PyLet deploy_manually request: target_state={input_data.target_state}"
    )

    try:
        result = service.apply_deployment(
            target_state=input_data.target_state,
            wait_for_ready=input_data.wait_for_ready,
        )

        active_statuses = [_instance_to_status(i) for i in result.active_instances]

        return PyLetDeploymentOutput(
            success=result.success,
            added_count=result.total_added,
            removed_count=result.total_removed,
            active_instances=active_statuses,
            failed_adds=(
                result.deployment_result.failed_adds if result.deployment_result else []
            ),
            failed_removes=(
                result.deployment_result.failed_removes
                if result.deployment_result
                else []
            ),
            error=result.error,
        )

    except Exception as e:
        logger.error(f"PyLet deploy_manually failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deployment failed: {str(e)}",
        )


@router.post("/deploy", response_model=PyLetOptimizeOutput)
async def pylet_deploy(input_data: PyLetDeployWithPlanInput):
    """Run optimization algorithm and deploy result via PyLet.

    This endpoint accepts the same parameters as /plan, runs the optimization
    algorithm to compute the optimal deployment, then deploys the result
    via PyLet.

    Args:
        input_data: Optimization parameters (same as /plan) plus model_ids mapping.

    Returns:
        PyLetOptimizeOutput with optimization results and deployment status.
    """
    service = _ensure_pylet_enabled()

    # Log planning request
    logger.info(
        f"[PLAN_REQUEST] endpoint=/deploy M={input_data.M} N={input_data.N} "
        f"algorithm={input_data.algorithm} objective={input_data.objective_method} "
        f"change_factor={input_data.a} target={input_data.target}"
    )

    # Validate model_ids against scheduler registry (PYLET-024)
    validator = ModelValidationService(get_scheduler_registry())
    validation = validator.validate_models(input_data.model_ids)
    if not validation.valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation.message,
        )

    try:
        # Convert inputs to numpy arrays
        B = np.array(input_data.B)
        initial = (
            np.array(input_data.initial)
            if input_data.initial is not None
            else np.array([-1] * input_data.M)
        )
        target = np.array(input_data.target)

        # Select optimizer based on algorithm (same logic as /plan)
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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown algorithm: {input_data.algorithm}",
            )

        # Convert deployment array to target_state dict using model_ids
        target_state: dict[str, int] = {}
        for idx in deployment:
            if idx >= 0:
                model_id = input_data.model_ids[idx]
                target_state[model_id] = target_state.get(model_id, 0) + 1

        logger.info(
            f"Optimization completed: score={score:.4f}, target_state={target_state}"
        )

        # Apply deployment via PyLet
        deploy_result = service.apply_deployment(
            target_state=target_state,
            wait_for_ready=input_data.wait_for_ready,
        )

        # Compute additional metrics
        service_capacity = optimizer.compute_service_capacity(deployment)
        changes_count = optimizer.compute_changes(deployment)

        active_statuses = [
            _instance_to_status(i) for i in deploy_result.active_instances
        ]

        return PyLetOptimizeOutput(
            deployment=deployment.tolist(),
            score=float(score),
            service_capacity=service_capacity.tolist(),
            changes_count=int(changes_count),
            stats=stats,
            deployment_success=deploy_result.success,
            added_count=deploy_result.total_added,
            removed_count=deploy_result.total_removed,
            active_instances=active_statuses,
            error=deploy_result.error,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"PyLet deploy failed - ValueError: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        )
    except Exception as e:
        logger.error(f"PyLet deploy failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deployment failed: {str(e)}",
        )


@router.post("/scale", response_model=PyLetScaleOutput)
async def pylet_scale(input_data: PyLetScaleInput):
    """Scale a specific model to target count.

    Args:
        input_data: Scale configuration.

    Returns:
        PyLetScaleOutput with scaling results.
    """
    service = _ensure_pylet_enabled()

    logger.info(
        f"PyLet scale request: model={input_data.model_id}, "
        f"target={input_data.target_count}"
    )

    try:
        # Get previous count
        previous = len(service.get_instances_by_model(input_data.model_id))

        result = service.scale_model(
            model_id=input_data.model_id,
            target_count=input_data.target_count,
            wait_for_ready=input_data.wait_for_ready,
        )

        active_statuses = [_instance_to_status(i) for i in result.active_instances]
        current = len(result.active_instances)

        return PyLetScaleOutput(
            success=result.success,
            model_id=input_data.model_id,
            previous_count=previous,
            current_count=current,
            added=result.total_added,
            removed=result.total_removed,
            active_instances=active_statuses,
            error=result.error,
        )

    except Exception as e:
        logger.error(f"PyLet scale failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scale failed: {str(e)}",
        )


@router.post("/migrate", response_model=PyLetMigrateOutput)
async def pylet_migrate(input_data: PyLetMigrateInput):
    """Migrate an instance (cancel and resubmit).

    Args:
        input_data: Migration configuration.

    Returns:
        PyLetMigrateOutput with migration results.
    """
    service = _ensure_pylet_enabled()

    logger.info(
        f"PyLet migrate request: pylet_id={input_data.pylet_id}, "
        f"target_model={input_data.target_model_id}"
    )

    try:
        result = service.migrate_model(
            pylet_id=input_data.pylet_id,
            target_model_id=input_data.target_model_id,
        )

        # Get the new instance info if migration succeeded
        new_pylet_id = None
        endpoint = None
        model_id = input_data.target_model_id or "unknown"

        if result.migration_result and result.migration_result.operations:
            op = result.migration_result.operations[0]
            new_pylet_id = op.new_pylet_id
            model_id = op.model_id

            # Get endpoint from active instances
            for inst in result.active_instances:
                if inst.pylet_id == new_pylet_id:
                    endpoint = inst.endpoint
                    break

        return PyLetMigrateOutput(
            success=result.success,
            old_pylet_id=input_data.pylet_id,
            new_pylet_id=new_pylet_id,
            model_id=model_id,
            endpoint=endpoint,
            error=result.error,
        )

    except Exception as e:
        logger.error(f"PyLet migrate failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Migration failed: {str(e)}",
        )


@router.post("/optimize", response_model=PyLetOptimizeOutput)
async def pylet_optimize(input_data: PyLetOptimizeInput):
    """Run optimizer and deploy via PyLet.

    This endpoint runs the optimization algorithm to compute optimal
    model distribution, then deploys the result via PyLet.

    Args:
        input_data: Optimization and deployment configuration.

    Returns:
        PyLetOptimizeOutput with optimization and deployment results.
    """
    service = _ensure_pylet_enabled()

    # Log planning request
    logger.info(
        f"[PLAN_REQUEST] endpoint=/optimize "
        f"model_ids={input_data.model_ids} target={input_data.target} "
        f"algorithm={input_data.algorithm} objective={input_data.objective_method} "
        f"change_factor={input_data.a}"
    )

    try:
        # Build arrays for optimizer
        N = len(input_data.model_ids)
        M = len(input_data.B)

        B = np.array(input_data.B)
        target = np.array(input_data.target)

        # For PyLet, the "initial" represents current model distribution
        # We'll compute optimal counts per model
        if input_data.algorithm == "simulated_annealing":
            optimizer = SimulatedAnnealingOptimizer(
                M=M,
                N=N,
                B=B,
                initial=np.array([-1] * M),  # Start fresh
                a=input_data.a,
                target=target,
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=input_data.objective_method,
            )

        elif input_data.algorithm == "integer_programming":
            optimizer = IntegerProgrammingOptimizer(
                M=M,
                N=N,
                B=B,
                initial=np.array([-1] * M),
                a=input_data.a,
                target=target,
            )

            deployment, score, stats = optimizer.optimize(
                objective_method=input_data.objective_method,
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown algorithm: {input_data.algorithm}",
            )

        # Convert deployment array to model counts
        target_state: dict[str, int] = {}
        for idx in deployment:
            if idx >= 0:
                model_id = input_data.model_ids[idx]
                target_state[model_id] = target_state.get(model_id, 0) + 1

        logger.info(f"Computed target state: {target_state}")

        # Apply deployment
        deploy_result = service.apply_deployment(
            target_state=target_state,
            wait_for_ready=input_data.wait_for_ready,
        )

        service_capacity = optimizer.compute_service_capacity(deployment)
        changes_count = optimizer.compute_changes(deployment)

        active_statuses = [
            _instance_to_status(i) for i in deploy_result.active_instances
        ]

        return PyLetOptimizeOutput(
            deployment=deployment.tolist(),
            score=float(score),
            service_capacity=service_capacity.tolist(),
            changes_count=int(changes_count),
            stats=stats,
            deployment_success=deploy_result.success,
            added_count=deploy_result.total_added,
            removed_count=deploy_result.total_removed,
            active_instances=active_statuses,
            error=deploy_result.error,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PyLet optimize failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}",
        )


@router.post("/terminate-all")
async def pylet_terminate_all(wait_for_drain: bool = False):
    """Terminate all PyLet-managed instances.

    Args:
        wait_for_drain: Whether to wait for drain before termination.

    Returns:
        Dict with termination results.
    """
    service = _ensure_pylet_enabled()

    logger.info(f"PyLet terminate-all request (wait_for_drain={wait_for_drain})")

    try:
        results = service.terminate_all(wait_for_drain=wait_for_drain)

        succeeded = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)

        return {
            "success": failed == 0,
            "total": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "details": results,
        }

    except Exception as e:
        logger.error(f"PyLet terminate-all failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Termination failed: {str(e)}",
        )
