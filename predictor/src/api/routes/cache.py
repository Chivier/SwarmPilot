"""Cache management endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import status
from fastapi.responses import JSONResponse

from src.api import dependencies
from src.api.routes.helpers import handle_library_exception

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
        # Use library API for cache stats
        stats = dependencies.predictor_api.get_cache_stats()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "cache_stats": stats,
                "message": "Cache statistics retrieved successfully",
            },
        )
    except Exception as e:
        raise handle_library_exception(e, "get cache stats")


@router.post("/clear", tags=["Cache"])
async def clear_cache():
    """Clear all cached models.

    Useful for freeing memory or forcing model reloads.
    This does not affect stored models on disk.

    Returns:
        JSONResponse confirming cache cleared.
    """
    try:
        # Use library API for cache clear
        dependencies.predictor_api.clear_cache()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Model cache cleared successfully",
            },
        )
    except Exception as e:
        raise handle_library_exception(e, "clear cache")
