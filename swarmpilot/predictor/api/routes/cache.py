"""Cache management endpoints."""

from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from swarmpilot.predictor.api import dependencies

router = APIRouter()


@router.get("/stats", tags=["Cache"])
async def get_cache_stats():
    """Get model cache statistics.

    Returns information about cache performance including hit rate,
    current size, and total hits/misses.

    Returns:
        JSONResponse with cache statistics.
    """
    try:
        stats = dependencies.model_cache.get_stats()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "cache_stats": stats,
                "message": "Cache statistics retrieved successfully",
            },
        )
    except (RuntimeError, AttributeError) as e:
        error_detail = {
            "error": "Failed to get cache stats",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="Cache stats retrieval failed",
            error_detail=error_detail,
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        ) from e
    except Exception as e:
        error_detail = {
            "error": "Failed to get cache stats",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="Cache stats retrieval failed",
            error_detail=error_detail,
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        ) from e


@router.post("/clear", tags=["Cache"])
async def clear_cache():
    """Clear all cached models.

    Useful for freeing memory or forcing model reloads.
    This does not affect stored models on disk.

    Returns:
        JSONResponse confirming cache cleared.
    """
    try:
        dependencies.model_cache.clear()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Model cache cleared successfully",
            },
        )
    except RuntimeError as e:
        error_detail = {
            "error": "Failed to clear cache",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="Cache clear operation failed",
            error_detail=error_detail,
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        ) from e
    except Exception as e:
        error_detail = {
            "error": "Failed to clear cache",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="Cache clear operation failed",
            error_detail=error_detail,
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        ) from e
