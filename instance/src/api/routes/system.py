"""System API routes.

Provides health check and system information endpoints.
"""

import time
from datetime import UTC, datetime

import psutil
from fastapi import APIRouter

from src.api.schemas import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HealthResponse,
    InferenceServerInfo,
    MemoryInfo,
    SystemInfo,
    SystemResources,
)

router = APIRouter(tags=["system"])

# Instance startup time for uptime calculation
_STARTUP_TIME = time.time()

# Instance ID (would typically come from config or environment)
_INSTANCE_ID = "inst_default"


def get_instance_id() -> str:
    """Get the instance identifier.

    Returns:
        Instance ID string.
    """
    return _INSTANCE_ID


def get_uptime_seconds() -> int:
    """Get instance uptime in seconds.

    Returns:
        Number of seconds since instance started.
    """
    return int(time.time() - _STARTUP_TIME)


def get_gpu_info() -> list[dict]:
    """Get GPU information.

    Returns:
        List of GPU info dictionaries. Empty list if no GPUs available.
    """
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_gb": float(parts[2]) / 1024,
                        "memory_used_gb": float(parts[3]) / 1024,
                        "utilization_percent": float(parts[4]),
                        "temperature_celsius": int(parts[5]),
                    }
                )
        return gpus
    except Exception:
        return []


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check instance health status.

    Returns:
        HealthResponse with status and current timestamp.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


@router.get("/info", response_model=SystemInfo)
async def system_info() -> SystemInfo:
    """Get system information and resource usage.

    Returns:
        SystemInfo with instance details, resources, and capabilities.
    """
    # CPU info
    cpu = CPUInfo(
        cores=psutil.cpu_count(),
        usage_percent=psutil.cpu_percent(),
    )

    # Memory info
    mem = psutil.virtual_memory()
    memory = MemoryInfo(
        total_gb=mem.total / (1024**3),
        used_gb=mem.used / (1024**3),
        usage_percent=mem.percent,
    )

    # Disk info
    disk_usage = psutil.disk_usage("/")
    disk = DiskInfo(
        total_gb=disk_usage.total / (1024**3),
        used_gb=disk_usage.used / (1024**3),
        available_gb=disk_usage.free / (1024**3),
        min_free_gb=10.0,  # Default minimum free space
        usage_percent=disk_usage.percent,
    )

    # GPU info
    gpu_data = get_gpu_info()
    gpus = [GPUInfo(**gpu) for gpu in gpu_data]

    # Build response
    return SystemInfo(
        instance_id=get_instance_id(),
        uptime_seconds=get_uptime_seconds(),
        supported_model_types=["llm"],
        inference_server=InferenceServerInfo(
            type="vllm",
            version="0.6.0",  # Would come from actual vLLM installation
        ),
        resources=SystemResources(
            cpu=cpu,
            memory=memory,
            disk=disk,
            gpu=gpus,
        ),
    )
