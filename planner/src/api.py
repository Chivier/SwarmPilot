"""FastAPI application for the Planner service."""

from datetime import datetime, timezone
from typing import Optional
import numpy as np

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from .models import (
    PlannerInput,
    PlannerOutput,
    DeploymentInput,
    DeploymentOutput,
    DeploymentStatus,
)
from .deployment_service import ModelMapper, InstanceDeployer
from .core.swarm_optimizer import (
    SimulatedAnnealingOptimizer,
    IntegerProgrammingOptimizer,
)
from . import __version__
from .logging_config import setup_logging
from .config import config

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

# Create FastAPI app
app = FastAPI(
    title="Planner Service",
    description="Model deployment optimization service for SwarmPilot",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
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
        scheduler_url = config.get_scheduler_url(input_data.scheduler_url)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
