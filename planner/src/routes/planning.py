"""Planning endpoint for computing optimal deployment plans."""

import numpy as np
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from ..core.swarm_optimizer import (
    IntegerProgrammingOptimizer,
    SimulatedAnnealingOptimizer,
)
from ..models import PlannerInput, PlannerOutput

router = APIRouter(tags=["planning"])


@router.post("/plan", response_model=PlannerOutput)
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
        logger.info(
            f"Received /plan request: M={input_data.M}, N={input_data.N}, "
            f"algorithm={input_data.algorithm}"
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=client_msg
        )
    except Exception as e:
        client_msg = f"Optimization failed: {str(e)}"
        logger.error(
            f"/plan request failed - Unexpected error: {e}. Returning HTTP 500. Client will receive: {client_msg}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=client_msg
        )
