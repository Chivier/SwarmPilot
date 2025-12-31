"""Background scheduler for continuous task processing.

This module provides backward compatibility by re-exporting
from src.services.background_scheduler.
"""

from src.services.background_scheduler import BackgroundScheduler

__all__ = ["BackgroundScheduler"]
