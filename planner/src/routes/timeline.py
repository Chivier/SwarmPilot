"""Timeline endpoints for tracking instance state changes."""

from fastapi import APIRouter
from loguru import logger

router = APIRouter(tags=["timeline"])


@router.get("/timeline")
async def get_instance_timeline():
    """Get the instance count timeline.

    Returns all recorded migration events with instance counts per model.
    Each entry includes timestamp, event type, instance counts, and metrics.

    Returns:
        Dictionary with success status and list of timeline entries
    """
    from ..instance_timeline_tracker import get_timeline_tracker

    tracker = get_timeline_tracker()
    entries = tracker.get_entries()
    logger.info(f"Timeline requested: {len(entries)} entries")
    return {"success": True, "entry_count": len(entries), "entries": entries}


@router.post("/timeline/clear")
async def clear_instance_timeline():
    """Clear the instance count timeline.

    Should be called at the start of a new experiment to ensure
    clean timeline data.

    Returns:
        Dictionary with success status and confirmation message
    """
    from ..instance_timeline_tracker import get_timeline_tracker

    tracker = get_timeline_tracker()
    tracker.clear()
    logger.info("Timeline cleared via API")
    return {"success": True, "message": "Timeline cleared successfully"}
