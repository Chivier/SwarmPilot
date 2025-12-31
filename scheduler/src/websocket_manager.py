"""WebSocket connection manager.

This module provides backward compatibility by re-exporting
from src.services.websocket_manager.
"""

from src.services.websocket_manager import ConnectionManager

__all__ = ["ConnectionManager"]
