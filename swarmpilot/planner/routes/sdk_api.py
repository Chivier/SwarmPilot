"""SDK deployment management endpoints for the Planner service.

This module provides a simplified REST interface for deploying,
scaling, and managing model services via PyLet. The endpoints are
designed for SDK and CLI consumers who need straightforward
deployment primitives.

Endpoints:
    POST /serve       - Deploy a model service
    POST /run         - Start a custom workload
    POST /register    - Register model requirements
    POST /deploy      - Trigger optimized deployment
    GET  /registered  - List registered models
    GET  /instances   - List all instances
    GET  /instances/{name} - Get single instance
    POST /scale       - Scale model replicas
    POST /terminate   - Terminate instances
    GET  /schedulers  - Get scheduler mapping
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from ..config import config
from ..models.sdk_api import (
    DeployResponse,
    InstanceDetailResponse,
    RegisteredModelsResponse,
    RegisterRequest,
    RunRequest,
    RunResponse,
    ScaleRequest,
    ScaleResponse,
    SchedulerMapResponse,
    ServeRequest,
    ServeResponse,
    TerminateRequest,
    TerminateResponse,
)
from ..scheduler_registry import get_scheduler_registry

try:
    from ..pylet.deployment_service import get_pylet_service_optional
except ImportError:
    get_pylet_service_optional = lambda: None  # type: ignore[assignment]

router = APIRouter(tags=["sdk"])

# Thread-safe store for registered model requirements.
_registered_models: dict[str, RegisterRequest] = {}
_registered_models_lock = threading.Lock()


def _ensure_pylet_enabled():
    """Ensure PyLet is enabled and initialized.

    Returns:
        The initialized PyLetDeploymentService.

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


def _sanitize_name(model: str) -> str:
    """Sanitize a model name for use as a deployment name.

    Replaces '/' with '-' and strips leading dashes.

    Args:
        model: Raw model name (e.g. 'Qwen/Qwen3-0.6B').

    Returns:
        Sanitized deployment name (e.g. 'Qwen-Qwen3-0.6B').
    """
    return model.replace("/", "-").lstrip("-")


def _resolve_scheduler(scheduler: str | None, model: str) -> str | None:
    """Resolve the scheduler URL for a model.

    If no scheduler is registered for the model, attempts to find an
    idle scheduler (one whose instances have all been removed) and
    reassign it to the new model.

    Args:
        scheduler: 'auto' to look up in SchedulerRegistry,
            None to skip, or a URL string.
        model: Model identifier for registry lookup.

    Returns:
        Resolved scheduler URL or None.
    """
    if scheduler is None:
        return None
    if scheduler != "auto":
        return scheduler

    registry = get_scheduler_registry()
    url = registry.get_scheduler_url(model)
    if url is not None:
        return url

    # No scheduler for this model — look for an idle one to reassign
    url = _try_reassign_idle_scheduler(model)
    return url


def _try_reassign_idle_scheduler(model: str) -> str | None:
    """Find an idle scheduler and reassign it to the given model.

    A scheduler is idle when the planner has no managed instances
    for its current model_id. On success the scheduler's model_id
    is updated via its ``/v1/model/reassign`` API and the
    SchedulerRegistry is remapped.

    Args:
        model: Target model identifier.

    Returns:
        Scheduler URL if reassignment succeeded, None otherwise.
    """
    from ..pylet.instance_manager import get_instance_manager
    from ..pylet.scheduler_client import SchedulerClient

    registry = get_scheduler_registry()

    try:
        manager = get_instance_manager()
    except RuntimeError:
        return None

    # Check each registered scheduler for idleness
    for info in registry.list_all():
        old_model = info.model_id
        # Count managed instances for this scheduler's model
        instances = [
            inst
            for inst in manager.instances.values()
            if inst.model_id == old_model
        ]
        if len(instances) > 0:
            continue

        # This scheduler has no instances — try to reassign it
        logger.info(
            f"Found idle scheduler for {old_model} at "
            f"{info.scheduler_url}, reassigning to {model}"
        )
        client = SchedulerClient(info.scheduler_url)
        try:
            if client.reassign_model(model):
                registry.reassign(old_model, model)
                return info.scheduler_url
        finally:
            client.close()

    logger.warning(
        f"No idle scheduler available for model {model}"
    )
    return None


def _instance_to_detail(instance) -> InstanceDetailResponse:
    """Convert a ManagedInstance to InstanceDetailResponse.

    Args:
        instance: A ManagedInstance from the instance manager.

    Returns:
        InstanceDetailResponse with the instance details.
    """
    return InstanceDetailResponse(
        pylet_id=instance.pylet_id,
        instance_id=instance.instance_id,
        model_id=instance.model_id,
        endpoint=instance.endpoint,
        status=instance.status.value,
        gpu_count=instance.gpu_count,
        error=instance.error,
    )


@router.post("/serve", response_model=ServeResponse)
async def serve_model(request: ServeRequest):
    """Deploy a model service.

    If ``model_or_command`` contains a ``/``, it is treated as a
    model name and a ``vllm serve`` command is generated
    automatically.  Otherwise it is passed through as a literal
    command.

    Args:
        request: Serve deployment configuration.

    Returns:
        ServeResponse with deployment results.
    """
    service = _ensure_pylet_enabled()

    model_or_cmd = request.model_or_command
    # Auto-generate vllm serve command for model names
    if "/" in model_or_cmd:
        model = model_or_cmd
    else:
        model = model_or_cmd

    name = request.name or _sanitize_name(model)
    scheduler_url = _resolve_scheduler(request.scheduler, model)

    logger.info(
        f"[SDK] serve name={name} model={model} "
        f"replicas={request.replicas} scheduler={scheduler_url}"
    )

    try:
        result = service.apply_deployment(
            target_state={model: request.replicas},
            wait_for_ready=True,
        )

        instance_ids = [inst.pylet_id for inst in result.active_instances]

        return ServeResponse(
            success=result.success,
            name=name,
            model=model,
            replicas=request.replicas,
            instances=instance_ids,
            scheduler_url=scheduler_url,
            error=result.error,
        )
    except Exception as e:
        logger.error(f"[SDK] serve failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Serve failed: {str(e)}",
        )


@router.post("/run", response_model=RunResponse)
async def run_workload(request: RunRequest):
    """Start a custom workload.

    Deploys the given command directly via PyLet without scheduler
    registration.

    Args:
        request: Run configuration with command.

    Returns:
        RunResponse with deployment results.
    """
    service = _ensure_pylet_enabled()

    name = request.name or f"run-{id(request) % 100000:05d}"

    logger.info(
        f"[SDK] run name={name} command={request.command!r} "
        f"replicas={request.replicas}"
    )

    try:
        result = service.apply_deployment(
            target_state={request.command: request.replicas},
            wait_for_ready=True,
        )

        instance_ids = [inst.pylet_id for inst in result.active_instances]

        return RunResponse(
            success=result.success,
            name=name,
            command=request.command,
            replicas=request.replicas,
            instances=instance_ids,
            error=result.error,
        )
    except Exception as e:
        logger.error(f"[SDK] run failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Run failed: {str(e)}",
        )


@router.post("/register")
async def register_model(request: RegisterRequest):
    """Register model requirements for optimized deployment.

    Stores model requirements in a thread-safe registry.
    Overwrites any existing entry for the same model.

    Args:
        request: Model registration details.

    Returns:
        Dict with registration status.
    """
    with _registered_models_lock:
        _registered_models[request.model] = request

    logger.info(
        f"[SDK] register model={request.model} "
        f"replicas={request.replicas} gpu={request.gpu_count}"
    )

    return {"status": "registered", "model": request.model}


@router.post("/deploy", response_model=DeployResponse)
async def deploy_registered():
    """Trigger optimized deployment of registered models.

    Collects all models from the register store and deploys them
    via PyLet. Requires PyLet to be enabled.

    Returns:
        DeployResponse with deployment results.
    """
    service = _ensure_pylet_enabled()

    with _registered_models_lock:
        models = dict(_registered_models)

    if not models:
        return DeployResponse(
            success=True,
            deployed_models=[],
            total_instances=0,
            error=None,
        )

    # Build target state from registered models
    target_state: dict[str, int] = {
        reg.model: reg.replicas for reg in models.values()
    }

    logger.info(f"[SDK] deploy target_state={target_state}")

    try:
        result = service.apply_deployment(
            target_state=target_state,
            wait_for_ready=True,
        )

        return DeployResponse(
            success=result.success,
            deployed_models=list(target_state.keys()),
            total_instances=len(result.active_instances),
            error=result.error,
        )
    except Exception as e:
        logger.error(f"[SDK] deploy failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deploy failed: {str(e)}",
        )


@router.get("/registered", response_model=RegisteredModelsResponse)
async def list_registered():
    """List all registered model requirements.

    Returns:
        RegisteredModelsResponse with registered models.
    """
    with _registered_models_lock:
        models = dict(_registered_models)

    return RegisteredModelsResponse(
        models=models,
        total=len(models),
    )


@router.get(
    "/instances",
    response_model=list[InstanceDetailResponse],
)
async def list_instances():
    """List all managed instances.

    Returns:
        List of InstanceDetailResponse for active instances.
    """
    service = _ensure_pylet_enabled()

    active = service.get_active_instances()
    return [_instance_to_detail(inst) for inst in active]


@router.get(
    "/instances/{name}",
    response_model=InstanceDetailResponse,
)
async def get_instance(name: str):
    """Get a single instance by deployment name or pylet_id.

    Searches active instances by pylet_id, instance_id, or
    model_id and returns the first match.

    Args:
        name: Instance identifier (pylet_id, instance_id,
            or model_id).

    Returns:
        InstanceDetailResponse for the matched instance.

    Raises:
        HTTPException: 404 if no matching instance found.
    """
    service = _ensure_pylet_enabled()

    active = service.get_active_instances()
    for inst in active:
        if name in (
            inst.pylet_id,
            inst.instance_id,
            inst.model_id,
        ):
            return _instance_to_detail(inst)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Instance not found: {name}",
    )


@router.post("/scale", response_model=ScaleResponse)
async def scale_model(request: ScaleRequest):
    """Scale model replicas to target count.

    Args:
        request: Scale configuration with model and target
            replica count.

    Returns:
        ScaleResponse with scaling results.
    """
    service = _ensure_pylet_enabled()

    logger.info(
        f"[SDK] scale model={request.model} replicas={request.replicas}"
    )

    try:
        previous = len(service.get_instances_by_model(request.model))

        result = service.scale_model(
            model_id=request.model,
            target_count=request.replicas,
            wait_for_ready=True,
        )

        current = len(result.active_instances)

        return ScaleResponse(
            success=result.success,
            model=request.model,
            previous_count=previous,
            current_count=current,
            error=result.error,
        )
    except Exception as e:
        logger.error(f"[SDK] scale failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scale failed: {str(e)}",
        )


@router.post("/terminate", response_model=TerminateResponse)
async def terminate_instances(request: TerminateRequest):
    """Terminate instances by name, model, or all.

    Exactly one of ``name``, ``model``, or ``all`` should be
    specified.

    Args:
        request: Termination criteria.

    Returns:
        TerminateResponse with termination results.
    """
    service = _ensure_pylet_enabled()

    logger.info(
        f"[SDK] terminate name={request.name} "
        f"model={request.model} all={request.all}"
    )

    try:
        if request.all:
            results = service.terminate_all()
            count = sum(1 for v in results.values() if v)
            return TerminateResponse(
                success=True,
                terminated_count=count,
                message=f"Terminated {count} instances",
            )

        if request.model:
            instances = service.get_instances_by_model(request.model)
            pylet_ids = [i.pylet_id for i in instances]
            mgr = service.instance_manager
            results = mgr.terminate_instances(pylet_ids)
            count = sum(1 for v in results.values() if v)
            return TerminateResponse(
                success=True,
                terminated_count=count,
                message=(
                    f"Terminated {count} instances for model {request.model}"
                ),
            )

        if request.name:
            # Try to match by pylet_id or instance_id
            active = service.get_active_instances()
            matched = [
                i
                for i in active
                if request.name in (i.pylet_id, i.instance_id, i.model_id)
            ]
            if not matched:
                return TerminateResponse(
                    success=False,
                    terminated_count=0,
                    message=(f"No instances found matching '{request.name}'"),
                )
            pylet_ids = [i.pylet_id for i in matched]
            mgr = service.instance_manager
            results = mgr.terminate_instances(pylet_ids)
            count = sum(1 for v in results.values() if v)
            return TerminateResponse(
                success=True,
                terminated_count=count,
                message=(
                    f"Terminated {count} instances matching '{request.name}'"
                ),
            )

        return TerminateResponse(
            success=False,
            terminated_count=0,
            message=(
                "No termination criteria specified. "
                "Set 'name', 'model', or 'all'."
            ),
        )

    except Exception as e:
        logger.error(f"[SDK] terminate failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terminate failed: {str(e)}",
        )


@router.get("/schedulers", response_model=SchedulerMapResponse)
async def get_schedulers():
    """Get the scheduler-to-model mapping.

    Returns:
        SchedulerMapResponse with model-to-scheduler URL map.
    """
    registry = get_scheduler_registry()
    all_info = registry.list_all()
    mapping = {info.model_id: info.scheduler_url for info in all_info}
    return SchedulerMapResponse(
        schedulers=mapping,
        total=len(mapping),
    )
