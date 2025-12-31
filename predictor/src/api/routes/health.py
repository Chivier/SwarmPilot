"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import status
from fastapi.responses import JSONResponse

from src.api import dependencies
from src.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint.

    Returns service health status and checks if storage is accessible.

    Returns:
        HealthResponse with status or error if storage unavailable.
    """
    try:
        storage_info = dependencies.storage.get_storage_info()

        if not storage_info["is_accessible"]:
            error_detail = {
                "status": "unhealthy",
                "reason": (
                    f"Storage directory not accessible: "
                    f"{storage_info['storage_dir']}"
                ),
            }
            dependencies._log_error(
                error_context="Health check - storage not accessible",
                error_detail=error_detail,
                include_traceback=False,
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=error_detail,
            )

        return HealthResponse(status="healthy")

    except Exception as e:
        error_detail = {
            "status": "unhealthy",
            "reason": f"Health check failed: {str(e)}",
        }
        dependencies._log_error(
            error_context="Health check - unexpected exception",
            error_detail=error_detail,
            exception=e,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_detail,
        )
